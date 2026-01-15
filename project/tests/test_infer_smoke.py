from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models import MambaCoeffNet


def _load_infer_module():
    root = Path(__file__).resolve().parents[1]
    infer_path = root / "scripts" / "infer.py"
    spec = importlib.util.spec_from_file_location("infer", infer_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_txt(path: Path, data: np.ndarray) -> None:
    header = "x y z strain u v w"
    np.savetxt(path, data, header=header, comments="", fmt="%.8e")


def test_infer_smoke(tmp_path: Path) -> None:
    xs = np.linspace(0.0, 1.0, 3)
    ys = np.linspace(0.0, 1.0, 4)
    xv, yv = np.meshgrid(xs, ys)
    x = xv.ravel()
    y = yv.ravel()
    z = np.zeros_like(x)

    strain_top = 1e-6 * (x + y)
    strain_bot = strain_top + 2e-6
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    w = 1e-3 * (x - y)

    top = np.column_stack([x, y, z, strain_top, u, v, w])
    bot = np.column_stack([x, y, z, strain_bot, u, v, w])

    top_path = tmp_path / "top.txt"
    bot_path = tmp_path / "bot.txt"
    _write_txt(top_path, top)
    _write_txt(bot_path, bot)

    plate_cfg = {
        "plate": {"h": 0.01, "Lx": 1.0, "Ly": 1.0},
        "basis": {"M": 2, "N": 2},
        "grid": {"Nx": 8, "Ny": 6},
        "norm": {"x_scale": 1.0, "y_scale": 1.0},
    }
    plate_path = tmp_path / "plate.yaml"
    with plate_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(plate_cfg, handle, sort_keys=False)

    ckpt_path = tmp_path / "checkpoint.pt"
    model = MambaCoeffNet(
        in_features=5,
        out_coeffs=4,
        d_model=32,
        n_layers=0,
        dropout=0.0,
        encoder_type="cnn",
    )
    ckpt = {
        "model_state": model.state_dict(),
        "config": {
            "model": {
                "d_model": 32,
                "n_layers": 0,
                "dropout": 0.0,
                "encoder_type": "cnn",
            },
            "basis": {"M": 2, "N": 2},
        },
    }
    torch.save(ckpt, ckpt_path)

    infer = _load_infer_module()
    out_dir = tmp_path / "out"
    args = infer.parse_args(
        [
            "--top",
            str(top_path),
            "--bot",
            str(bot_path),
            "--plate",
            str(plate_path),
            "--ckpt",
            str(ckpt_path),
            "--out_dir",
            str(out_dir),
            "--device",
            "cpu",
            "--order_mode",
            "as_is",
            "--pair_method",
            "s",
        ]
    )
    infer.run_infer(args)

    assert (out_dir / "w_grid.npy").exists()
    assert (out_dir / "figures" / "surface_3d.png").exists()
    assert (out_dir / "figures" / "kappa_curve.png").exists()
    assert (out_dir / "summary.json").exists()
