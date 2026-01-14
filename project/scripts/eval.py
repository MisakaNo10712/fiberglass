#!/usr/bin/env python
"""
Evaluate a trained checkpoint on fiber sequence data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import FiberSequenceDataset, fiber_sequence_collate
from models import MambaCoeffNet
from train import Trainer
from utils import setup_logger

logger = setup_logger("eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fiber sequence model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|auto")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(config_device: Any, override: str | None) -> torch.device:
    device_value = override if override is not None else config_device
    if isinstance(device_value, dict):
        device_value = device_value.get("type", "auto")
    device_value = device_value or "auto"
    if device_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def build_dataloader(config: dict[str, Any]) -> DataLoader:
    data_cfg = config.get("data", {})
    dataset = FiberSequenceDataset(
        manifest_path=data_cfg.get("manifest"),
        samples_dir=data_cfg.get("samples_dir"),
        df_list=data_cfg.get("df_list") if data_cfg.get("use_dataframe") else None,
    )
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=fiber_sequence_collate,
    )


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


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = (
        Path(args.config)
        if args.config is not None
        else checkpoint_path.parent / "config.yaml"
    )
    config = load_config(config_path)

    if args.manifest is not None:
        config.setdefault("data", {})["manifest"] = args.manifest
    if args.data_dir is not None:
        config.setdefault("data", {})["samples_dir"] = args.data_dir
    if args.device is not None:
        config["device"] = args.device

    device = resolve_device(config.get("device", "auto"), None)

    dataloader = build_dataloader(config)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = Trainer(model, optimizer, config, device)
    trainer.load_checkpoint(checkpoint_path)

    metrics = trainer.eval(dataloader)
    logger.info("Eval metrics: %s", metrics)

    output_path = checkpoint_path.parent / "eval_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
