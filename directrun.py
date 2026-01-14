import os
import glob
import math
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Utils
# ----------------------------
def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_fiber_txt(path: str) -> Dict[str, np.ndarray]:
    """
    Read whitespace-separated txt with header:
    x y z strain u v w

    Returns dict with:
      coords: (P,3) float32
      strain: (P,1) float32
      disp: (P,3) float32  (u,v,w)
    """
    # Robust load: read header then numeric
    # np.genfromtxt can handle whitespace + header names.
    data = np.genfromtxt(path, names=True, dtype=np.float64, encoding=None)

    # Names must include these columns
    required = ["x", "y", "z", "strain", "u", "v", "w"]
    for k in required:
        if k not in data.dtype.names:
            raise ValueError(f"File {path} missing column '{k}'. Found: {data.dtype.names}")

    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    z = data["z"].astype(np.float32)
    strain = data["strain"].astype(np.float32)[:, None]
    disp = np.stack(
        [data["u"].astype(np.float32),
         data["v"].astype(np.float32),
         data["w"].astype(np.float32)],
        axis=1
    )
    coords = np.stack([x, y, z], axis=1).astype(np.float32)

    return {"coords": coords, "strain": strain, "disp": disp}


class Normalizer:
    """Per-channel (mean,std) normalizer for numpy/tensor."""
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-12):
        self.mean = mean.astype(np.float32)
        self.std = (std.astype(np.float32) + eps)

    def encode_np(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def decode_np(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def encode_t(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def decode_t(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        return x * std + mean


# ----------------------------
# 2D Basis for Plate (Physics-Encoded Decoder)
# ----------------------------
def build_plate_basis(
    coords_xy: torch.Tensor,
    W: float,
    L: float,
    M: int,
    N: int,
    clamp_y: bool = True
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    2D basis expansion on rectangle:
      x in [0,W], y in [0,L]

    We hard-encode clamped boundary at y=L using gate(y)=(1-y/L)^2:
      w(x,L)=0 and approx dw/dy(x,L)=0

    Basis:
      gate(y) * cos(m*pi*x/W) * cos(n*pi*y/L)   for m=0..M, n=0..N
      gate(y) * sin(m*pi*x/W) * cos(n*pi*y/L)   for m=1..M, n=0..N

    Return:
      Phi: (P,K) basis matrix at P query points
      meta: dict with (m,n,is_sin) for each basis term
    """
    x = coords_xy[:, 0] / W
    y = coords_xy[:, 1] / L

    if clamp_y:
        gate = (1.0 - y).clamp(min=0.0) ** 2
    else:
        gate = torch.ones_like(y)

    terms: List[torch.Tensor] = []
    m_list, n_list, is_sin_list = [], [], []

    # cos-cos
    for m in range(M + 1):
        cos_mx = torch.cos(m * math.pi * x) if m > 0 else torch.ones_like(x)
        for n in range(N + 1):
            cos_ny = torch.cos(n * math.pi * y) if n > 0 else torch.ones_like(y)
            terms.append(gate * cos_mx * cos_ny)
            m_list.append(m); n_list.append(n); is_sin_list.append(0)

    # sin-cos
    for m in range(1, M + 1):
        sin_mx = torch.sin(m * math.pi * x)
        for n in range(N + 1):
            cos_ny = torch.cos(n * math.pi * y) if n > 0 else torch.ones_like(y)
            terms.append(gate * sin_mx * cos_ny)
            m_list.append(m); n_list.append(n); is_sin_list.append(1)

    Phi = torch.stack(terms, dim=1)  # (P,K)
    meta = {
        "m": torch.tensor(m_list, dtype=torch.long),
        "n": torch.tensor(n_list, dtype=torch.long),
        "is_sin": torch.tensor(is_sin_list, dtype=torch.long),
    }
    return Phi, meta


def make_hf_weight(meta: Dict[str, torch.Tensor]) -> torch.Tensor:
    """High-frequency penalty weight per basis term ~ (m^2+n^2) normalized to [0,1]."""
    m = meta["m"].float()
    n = meta["n"].float()
    w = (m * m + n * n)
    w = w / w.max().clamp(min=1.0)
    return w


# ----------------------------
# Encoder (order-robust PointNet-style)
# ----------------------------
class PointNetEncoder(nn.Module):
    """
    Order-invariant encoder for set of tokens (x_norm, y_norm, strain_norm, ...).
    This avoids "sequence adjacency" artifacts when your txt actually concatenates multiple curves.
    """
    def __init__(self, d_in: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,P,F)
        return z: (B,d_model)
        """
        h = self.mlp1(x)                    # (B,P,d)
        g = h.mean(dim=1, keepdim=True)     # (B,1,d) global
        h2 = self.mlp2(torch.cat([h, g.expand_as(h)], dim=-1))
        z = h2.mean(dim=1)                  # (B,d)
        return z


# ----------------------------
# Full Model: Encoder -> Basis Coeffs -> Displacement at query points
# ----------------------------
class PlateBasisNet(nn.Module):
    def __init__(
        self,
        Phi: torch.Tensor,                  # (P,K)
        meta: Dict[str, torch.Tensor],
        in_feature_dim: int,
        d_model: int = 128,
        out_channels: int = 3,              # predict (u,v,w)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("Phi_T", Phi.T.contiguous())  # (K,P)
        self.K = Phi.shape[1]
        self.P = Phi.shape[0]
        self.out_channels = out_channels

        self.encoder = PointNetEncoder(in_feature_dim, d_model=d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_channels * self.K),
        )

        hf_w = make_hf_weight(meta)  # (K,)
        self.register_buffer("hf_w", hf_w)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feats: (B,P,F)
        returns:
          pred_disp_norm: (B,P,3)  (normalized displacement)
          coeffs: (B,3,K)
        """
        z = self.encoder(feats)  # (B,d)
        coeffs = self.head(z).view(-1, self.out_channels, self.K)  # (B,3,K)
        pred_cp = torch.einsum("bck,kp->bcp", coeffs, self.Phi_T)  # (B,3,P)
        pred = pred_cp.permute(0, 2, 1).contiguous()               # (B,P,3)
        return pred, coeffs

    def hf_regularizer(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Penalize high-frequency coefficients to stabilize the ill-posed inverse mapping.
        coeffs: (B,3,K)
        """
        w = self.hf_w[None, None, :]  # (1,1,K)
        return (coeffs * coeffs * w).mean()


# ----------------------------
# Dataset (many files, fixed coordinates)
# ----------------------------
class FiberDataset(Dataset):
    """
    Each file is one sample.
    Coordinates must match exactly (fixed fiber layout).
    """
    def __init__(self, file_paths: List[str], cache_in_memory: bool = True):
        super().__init__()
        if len(file_paths) == 0:
            raise ValueError("No txt files found.")
        self.file_paths = file_paths
        self.cache_in_memory = cache_in_memory

        first = read_fiber_txt(file_paths[0])
        self.coords = first["coords"]  # (P,3)
        self.P = self.coords.shape[0]

        self._cache: Optional[List[Dict[str, np.ndarray]]] = None
        if cache_in_memory:
            self._cache = []
            for fp in file_paths:
                d = read_fiber_txt(fp)
                self._check_coords(d["coords"], fp)
                self._cache.append(d)

    def _check_coords(self, coords: np.ndarray, fp: str) -> None:
        if coords.shape != self.coords.shape:
            raise ValueError(f"Point count mismatch in {fp}: {coords.shape} vs {self.coords.shape}")
        if not np.allclose(coords, self.coords, atol=1e-6):
            raise ValueError(
                f"Coordinates mismatch in {fp}.\n"
                f"All samples must have same coordinate layout & ordering."
            )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._cache is not None:
            d = self._cache[idx]
        else:
            d = read_fiber_txt(self.file_paths[idx])
            self._check_coords(d["coords"], self.file_paths[idx])

        strain = torch.from_numpy(d["strain"])  # (P,1)
        disp = torch.from_numpy(d["disp"])      # (P,3)
        return {"strain": strain, "disp": disp}


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def compute_stats(ds: FiberDataset, indices: List[int]) -> Dict[str, np.ndarray]:
    strains = []
    disps = []
    for i in indices:
        item = ds[i]
        strains.append(item["strain"].numpy())
        disps.append(item["disp"].numpy())
    strain_all = np.concatenate(strains, axis=0)  # (sumP,1)
    disp_all = np.concatenate(disps, axis=0)      # (sumP,3)

    return {
        "strain_mean": strain_all.mean(axis=0),
        "strain_std": strain_all.std(axis=0),
        "disp_mean": disp_all.mean(axis=0),
        "disp_std": disp_all.std(axis=0),
    }


@dataclass
class TrainConfig:
    data_dir: str = "data"
    out_dir: str = "runs/plate_basis"
    seed: int = 42

    # basis size
    M: int = 4
    N: int = 12
    clamp_y: bool = True

    # train split
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # training
    batch_size: int = 16
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_hf: float = 1e-3  # coeff smoothness
    huber_delta: float = 1.0

    # model
    d_model: int = 128
    dropout: float = 0.1

    # export grid (for "whole plate")
    export_grid_nx: int = 61   # e.g., 0..0.30 step 0.005 -> 61
    export_grid_ny: int = 201  # e.g., 0..1.00 step 0.005 -> 201


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


@torch.no_grad()
def eval_epoch(
    model: PlateBasisNet,
    loader: DataLoader,
    strain_norm: Normalizer,
    disp_norm: Normalizer,
    coord_feat: torch.Tensor,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n_batches = 0

    for batch in loader:
        strain = batch["strain"].to(device)  # (B,P,1)
        disp = batch["disp"].to(device)      # (B,P,3)

        strain_n = strain_norm.encode_t(strain)
        disp_n = disp_norm.encode_t(disp)

        B = strain.shape[0]
        coord = coord_feat[None, :, :].expand(B, -1, -1).to(device)  # (B,P,2)
        feats = torch.cat([coord, strain_n], dim=-1)                  # (B,P,3)

        pred_n, coeffs = model(feats)
        loss_disp = F.huber_loss(pred_n, disp_n, delta=1.0)
        loss = loss_disp + 0.0 * model.hf_regularizer(coeffs)  # no hf in eval
        loss_sum += float(loss.item())
        n_batches += 1

    return {"loss": loss_sum / max(1, n_batches)}


def train(cfg: TrainConfig) -> None:
    seed_all(cfg.seed)
    ensure_dir(cfg.out_dir)

    # collect txt files
    if os.path.isfile(cfg.data_dir) and cfg.data_dir.endswith(".txt"):
        files = [cfg.data_dir]
    else:
        files = sorted(glob.glob(os.path.join(cfg.data_dir, "*.txt")))

    if len(files) == 0:
        raise RuntimeError(f"No .txt files found under {cfg.data_dir}")

    print(f"[INFO] Found {len(files)} samples")

    ds = FiberDataset(files, cache_in_memory=True)

    # split
    train_idx, val_idx, test_idx = split_indices(len(ds), cfg.val_ratio, cfg.test_ratio, cfg.seed)
    if len(train_idx) == 0:
        raise RuntimeError("Train split is empty. Provide more files or reduce val/test ratio.")
    print(f"[INFO] split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # coordinate normalization from fixed coords
    coords = ds.coords  # (P,3) numpy
    x = coords[:, 0]
    y = coords[:, 1]
    W = float(x.max())
    L = float(y.max())
    coords_xy_t = torch.from_numpy(np.stack([x, y], axis=1)).float()

    coord_feat = torch.stack([coords_xy_t[:, 0] / W, coords_xy_t[:, 1] / L], dim=1)  # (P,2)

    # basis (query points = all fiber points)
    Phi, meta = build_plate_basis(coords_xy_t, W=W, L=L, M=cfg.M, N=cfg.N, clamp_y=cfg.clamp_y)

    # stats (train only)
    stats = compute_stats(ds, train_idx)
    strain_normalizer = Normalizer(stats["strain_mean"], stats["strain_std"])
    disp_normalizer = Normalizer(stats["disp_mean"], stats["disp_std"])

    # loaders (SubsetSampler style)
    def make_loader(indices: List[int], shuffle: bool) -> DataLoader:
        subset = torch.utils.data.Subset(ds, indices)
        return DataLoader(subset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False) if len(val_idx) > 0 else None
    test_loader = make_loader(test_idx, shuffle=False) if len(test_idx) > 0 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # model
    model = PlateBasisNet(
        Phi=Phi,
        meta=meta,
        in_feature_dim=3,  # (x_norm,y_norm,strain_norm)
        d_model=cfg.d_model,
        out_channels=3,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_path = os.path.join(cfg.out_dir, "best.pt")

    # save config + normals
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2, ensure_ascii=False)
    np.savez(
        os.path.join(cfg.out_dir, "normalizers.npz"),
        strain_mean=stats["strain_mean"], strain_std=stats["strain_std"],
        disp_mean=stats["disp_mean"], disp_std=stats["disp_std"],
        W=np.array([W], dtype=np.float32),
        L=np.array([L], dtype=np.float32),
    )

    # training loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            strain = batch["strain"].to(device)  # (B,P,1)
            disp = batch["disp"].to(device)      # (B,P,3)

            strain_n = strain_normalizer.encode_t(strain)
            disp_n = disp_normalizer.encode_t(disp)

            B = strain.shape[0]
            coord = coord_feat[None, :, :].expand(B, -1, -1).to(device)
            feats = torch.cat([coord, strain_n], dim=-1)  # (B,P,3)

            pred_n, coeffs = model(feats)

            loss_disp = F.huber_loss(pred_n, disp_n, delta=cfg.huber_delta)
            loss_hf = model.hf_regularizer(coeffs)
            loss = loss_disp + cfg.lambda_hf * loss_hf

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            loss_sum += float(loss.item())
            n_batches += 1

        train_loss = loss_sum / max(1, n_batches)

        # val
        if val_loader is not None:
            val_metrics = eval_epoch(model, val_loader, strain_normalizer, disp_normalizer, coord_feat, device)
            val_loss = val_metrics["loss"]
        else:
            val_loss = train_loss

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "W": W, "L": L,
                    "M": cfg.M, "N": cfg.N, "clamp_y": cfg.clamp_y,
                },
                best_path
            )

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:04d}] train={train_loss:.6f}  val={val_loss:.6f}  best_val={best_val:.6f}")

    print(f"[INFO] Training done. Best checkpoint: {best_path}")

    # test
    if test_loader is not None:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_metrics = eval_epoch(model, test_loader, strain_normalizer, disp_normalizer, coord_feat, device)
        print(f"[TEST] loss={test_metrics['loss']:.6f}")

    # export one example grid using first train sample
    export_grid_example(cfg, best_path, ds, strain_normalizer, disp_normalizer, device)


@torch.no_grad()
def export_grid_example(
    cfg: TrainConfig,
    ckpt_path: str,
    ds: FiberDataset,
    strain_norm: Normalizer,
    disp_norm: Normalizer,
    device: torch.device
) -> None:
    """
    Use the trained model to export u,v,w on a dense grid (whole plate).
    We run inference on sample 0.
    """
    out_csv = os.path.join(cfg.out_dir, "grid_pred_case0.csv")

    ckpt = torch.load(ckpt_path, map_location=device)
    W = float(ckpt["W"]); L = float(ckpt["L"])
    M = int(ckpt["M"]); N = int(ckpt["N"]); clamp_y = bool(ckpt["clamp_y"])

    # build grid points
    xs = np.linspace(0.0, W, cfg.export_grid_nx, dtype=np.float32)
    ys = np.linspace(0.0, L, cfg.export_grid_ny, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    grid_xy = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)  # (Q,2)
    grid_xy_t = torch.from_numpy(grid_xy).float()

    # basis on grid
    Phi_grid, meta = build_plate_basis(grid_xy_t, W=W, L=L, M=M, N=N, clamp_y=clamp_y)

    # model (same weights, but swap Phi_T)
    # rebuild a fresh model with grid Phi
    model_grid = PlateBasisNet(
        Phi=Phi_grid,
        meta=meta,
        in_feature_dim=3,
        d_model=cfg.d_model,
        out_channels=3,
        dropout=cfg.dropout,
    ).to(device)
    model_grid.load_state_dict(ckpt["model"], strict=False)  # strict=False because Phi differs but weights match
    #sd = ckpt["model"]
    #sd = {k: v for k, v in sd.items() if not k.startswith("Phi_")}  # 或至少排除 "Phi_T"
    #missing, unexpected = model_grid.load_state_dict(sd, strict=False)
    #print("missing:", missing)
    #print("unexpected:", unexpected)

    #TODO: FIX THIS PROBLEM
    # load one sample
    item = ds[0]
    strain = item["strain"].unsqueeze(0).to(device)  # (1,P,1)

    # coords features come from grid now for decoding, but encoder must see sensor coords:
    # We'll compute sensor coord_feat from ds.coords. 
    coords = ds.coords
    x = coords[:, 0]; y = coords[:, 1]
    coord_feat_sensor = torch.from_numpy(np.stack([x / W, y / L], axis=1)).float().to(device)  # (P,2)

    strain_n = strain_norm.encode_t(strain)  # (1,P,1)
    feats = torch.cat([coord_feat_sensor.unsqueeze(0), strain_n], dim=-1)  # (1,P,3)

    # forward: pred on GRID because model_grid has Phi_grid
    pred_grid_n, _ = model_grid(feats)              # (1,Q,3) normalized
    pred_grid = disp_norm.decode_t(pred_grid_n)[0]  # (Q,3) unnorm

    pred_grid = pred_grid.detach().cpu().numpy()
    out = np.concatenate([grid_xy, pred_grid], axis=1)  # x,y,u,v,w

    header = "x,y,u,v,w"
    np.savetxt(out_csv, out, delimiter=",", header=header, comments="")
    print(f"[INFO] Exported whole-plate grid prediction to: {out_csv}")


if __name__ == "__main__":
    # You can edit config here (or extend to argparse if you want).
    cfg = TrainConfig(
        data_dir="data",          # or a single file: "merged_fibers.txt"
        out_dir="runs/plate_basis",
        seed=42,
        M=4, N=12,
        clamp_y=True,
        batch_size=16,
        epochs=100,
        lr=1e-3,
        weight_decay=1e-4,
        lambda_hf=1e-3,
        d_model=128,
        dropout=0.1,
        export_grid_nx=61,
        export_grid_ny=201,
    )
    train(cfg)
