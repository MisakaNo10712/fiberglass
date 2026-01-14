import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# 直接复用你现有的 model.py 里这些定义
from model import read_fiber_txt, Normalizer, build_plate_basis, PlateBasisNet


@torch.no_grad()
def main(
    run_dir: str = "runs/plate_basis",
    sample_txt: str = "merged_fibers.txt",   # 用来提供传感器点coords和strain
    ckpt_name: str = "best.pt",
    save_dir: str = None,
):
    if save_dir is None:
        save_dir = run_dir

    ckpt_path = os.path.join(run_dir, ckpt_name)
    cfg_path = os.path.join(run_dir, "config.json")
    norm_path = os.path.join(run_dir, "normalizers.npz")

    assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
    assert os.path.exists(cfg_path), f"Missing config: {cfg_path}"
    assert os.path.exists(norm_path), f"Missing normalizers: {norm_path}"
    assert os.path.exists(sample_txt), f"Missing sample txt: {sample_txt}"

    # 读取训练配置（为了拿 d_model/dropout 和 export_grid_nx/ny）
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    nx = int(cfg.get("export_grid_nx", 61))
    ny = int(cfg.get("export_grid_ny", 201))
    d_model = int(cfg.get("d_model", 128))
    dropout = float(cfg.get("dropout", 0.1))

    # 读取 normalizer
    nz = np.load(norm_path)
    strain_norm = Normalizer(nz["strain_mean"], nz["strain_std"])
    disp_norm = Normalizer(nz["disp_mean"], nz["disp_std"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    # 读取 checkpoint（含 W,L,M,N,clamp_y 和模型权重）
    ckpt = torch.load(ckpt_path, map_location=device)
    W = float(ckpt["W"])
    L = float(ckpt["L"])
    M = int(ckpt["M"])
    N = int(ckpt["N"])
    clamp_y = bool(ckpt["clamp_y"])

    # 读取一个样本（提供传感器点的 coords 和 strain）
    d = read_fiber_txt(sample_txt)
    coords = d["coords"]  # (P,3)
    strain = d["strain"]  # (P,1)

    # 传感器点坐标特征（x_norm,y_norm）
    x = coords[:, 0].astype(np.float32)
    y = coords[:, 1].astype(np.float32)
    coord_feat_sensor = np.stack([x / W, y / L], axis=1).astype(np.float32)  # (P,2)

    # 拼成 feats = [x_norm, y_norm, strain_norm]
    strain_t = torch.from_numpy(strain).unsqueeze(0).to(device)  # (1,P,1)
    strain_n = strain_norm.encode_t(strain_t)                    # (1,P,1)
    coord_t = torch.from_numpy(coord_feat_sensor).unsqueeze(0).to(device)  # (1,P,2)
    feats = torch.cat([coord_t, strain_n], dim=-1)               # (1,P,3)

    # 构建整板规则网格点 (Q,2)
    xs = np.linspace(0.0, W, nx, dtype=np.float32)
    ys = np.linspace(0.0, L, ny, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    grid_xy = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1).astype(np.float32)  # (Q,2)
    grid_xy_t = torch.from_numpy(grid_xy).float().to(device)

    # 网格点上的 basis
    Phi_grid, meta = build_plate_basis(grid_xy_t, W=W, L=L, M=M, N=N, clamp_y=clamp_y)

    # 构建“网格版”模型：Phi_T 是网格的 (K,Q)
    model_grid = PlateBasisNet(
        Phi=Phi_grid,
        meta=meta,
        in_feature_dim=3,
        d_model=d_model,
        out_channels=3,
        dropout=dropout,
    ).to(device)
    model_grid.eval()

    # 加载权重：跳过 Phi_*（解决 Phi_T shape mismatch）
    sd = ckpt["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("Phi_")}
    missing, unexpected = model_grid.load_state_dict(sd, strict=False)
    print("[INFO] load_state_dict missing:", missing)
    print("[INFO] load_state_dict unexpected:", unexpected)

    # 推理：输出 (1,Q,3) 归一化位移 -> 反归一化
    pred_grid_n, _ = model_grid(feats)          # (1,Q,3)
    pred_grid = disp_norm.decode_t(pred_grid_n)[0]  # (Q,3)
    pred = pred_grid.detach().cpu().numpy()

    # reshape 回 (ny,nx,3)
    pred = pred.reshape(ny, nx, 3)
    u = pred[:, :, 0]
    v = pred[:, :, 1]
    w = pred[:, :, 2]

    os.makedirs(save_dir, exist_ok=True)

    # --- 绘图：u/v/w 场（各自一张图） ---
    def plot_field(Z, title, fname):
        plt.figure()
        plt.contourf(gx, gy, Z, levels=60)
        plt.colorbar(label=title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.axis("equal")
        out_path = os.path.join(save_dir, fname)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print("[INFO] saved:", out_path)

    plot_field(u, "Predicted u", "pred_u.png")
    plot_field(v, "Predicted v", "pred_v.png")
    plot_field(w, "Predicted w", "pred_w.png")

    # --- 额外：画位移矢量场 (u,v) 的 quiver（稀疏采样避免太密） ---
    step = max(1, min(nx, ny) // 30)  # 大概 30x30 箭头
    plt.figure()
    plt.quiver(gx[::step, ::step], gy[::step, ::step],
               u[::step, ::step], v[::step, ::step])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted displacement vectors (u,v)")
    plt.axis("equal")
    out_path = os.path.join(save_dir, "pred_uv_quiver.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("[INFO] saved:", out_path)

    # 如果你想交互式弹窗看图，取消下面注释
    # plt.show()


if __name__ == "__main__":
    main(
        run_dir="runs/plate_basis",
        sample_txt="merged_fibers.txt",   # 改成你想用的某个样本txt
        ckpt_name="best.pt",
        save_dir="runs/plate_basis/viz"
    )
