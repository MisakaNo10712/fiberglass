#!/usr/bin/env python
"""
Train a coefficient regression model from fiber sequences.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root + src to path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from datasets import FiberSequenceDataset, fiber_sequence_collate
from models import MambaCoeffNet
from train import Trainer
from utils import setup_logger

logger = setup_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fiber sequence model")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|auto")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(config_device: Any, override: str | None) -> torch.device:
    device_value = override if override is not None else config_device
    if isinstance(device_value, dict):
        device_value = device_value.get("type", "auto")
    device_value = device_value or "auto"
    if device_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = config.get("data", {})
    dataset = FiberSequenceDataset(
        manifest_path=data_cfg.get("manifest"),
        samples_dir=data_cfg.get("samples_dir"),
        df_list=data_cfg.get("df_list") if data_cfg.get("use_dataframe") else None,
        stats_path=config.get("stats_path"),
    )
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=fiber_sequence_collate,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=fiber_sequence_collate,
    )
    return train_loader, val_loader


def build_model(config: dict[str, Any]) -> MambaCoeffNet:
    model_cfg = config.get("model", {})
    basis_cfg = config.get("basis", {})
    out_coeffs = int(model_cfg.get("out_coeffs", basis_cfg.get("M") * basis_cfg.get("N")))
    model_cfg["out_coeffs"] = out_coeffs
    return MambaCoeffNet(
        in_features=int(model_cfg.get("in_features", 5)),
        d_model=int(model_cfg.get("d_model", 128)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        out_coeffs=out_coeffs,
        encoder_type=str(model_cfg.get("encoder_type", "auto")),
    )


def maybe_plot_curves(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("matplotlib unavailable; skipping curve plot")
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row.get("train_loss", float("nan")) for row in history]
    val_loss = [row.get("val_loss", float("nan")) for row in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train")
    if any(np.isfinite(val_loss)):
        plt.plot(epochs, val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    if args.manifest is not None:
        config.setdefault("data", {})["manifest"] = args.manifest
    if args.data_dir is not None:
        config.setdefault("data", {})["samples_dir"] = args.data_dir
    if args.device is not None:
        config["device"] = args.device

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(config.get("device", "auto"), None)
    logger.info("Using device: %s", device)

    run_cfg = config.get("run", {})
    run_root = Path(run_cfg.get("save_dir", "runs"))
    run_name = run_cfg.get("name") or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    train_loader, val_loader = build_dataloaders(config)

    model = build_model(config)
    train_cfg = config.get("train", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    trainer = Trainer(model, optimizer, config, device)

    epochs = int(train_cfg.get("epochs", 10))
    best_loss = float("inf")
    checkpoint_path = run_dir / "checkpoint.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_one_epoch(train_loader)
        val_metrics = trainer.eval(val_loader)
        row = trainer.record_epoch(epoch, train_metrics, val_metrics)

        logger.info(
            "Epoch %d | train loss %.6f | val loss %.6f | rmse_kappa %.6f",
            epoch,
            row.get("train_loss", float("nan")),
            row.get("val_loss", float("nan")),
            row.get("train_rmse_kappa", float("nan")),
        )

        current_loss = row.get("val_loss", row.get("train_loss", float("inf")))
        if current_loss < best_loss:
            best_loss = current_loss
            trainer.save_checkpoint(checkpoint_path, epoch, row)

    trainer.save_history_csv(run_dir / "metrics.csv")
    maybe_plot_curves(trainer.history, run_dir / "curves.png")

    logger.info("Run artifacts saved to %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
