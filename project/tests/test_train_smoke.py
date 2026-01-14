from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.basis.dct2 import w_from_coeff
from src.datasets import FiberSequenceDataset, fiber_sequence_collate
from src.models import MambaCoeffNet
from src.operators.curvature_projection import kappa_t_from_coeff
from src.train import Trainer


def _make_sample(L: int = 32, M: int = 4, N: int = 4, Lx: float = 1.0, Ly: float = 1.0):
    torch.manual_seed(0)
    a_true = torch.randn(M, N) * 0.1
    x = torch.linspace(0.05, Lx - 0.05, steps=L)
    y = torch.linspace(0.05, Ly - 0.05, steps=L)
    tx = torch.ones(L)
    ty = torch.zeros(L)

    kappa_t = kappa_t_from_coeff(a_true, x, y, tx, ty, Lx=Lx, Ly=Ly)
    w_points = w_from_coeff(a_true, x, y, Lx=Lx, Ly=Ly)

    df = pd.DataFrame(
        {
            "x": x.numpy(),
            "y": y.numpy(),
            "tx": tx.numpy(),
            "ty": ty.numpy(),
            "kappa_t": kappa_t.numpy(),
            "mask": np.ones(L, dtype=np.float32),
            "w": w_points.numpy(),
        }
    )
    return df


def test_train_overfit_smoke():
    df = _make_sample()
    dataset = FiberSequenceDataset(df_list=[df])
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=fiber_sequence_collate)

    config = {
        "seed": 0,
        "basis": {"M": 4, "N": 4, "Lx": 1.0, "Ly": 1.0},
        "model": {"in_features": 5, "d_model": 32, "n_layers": 2, "dropout": 0.0, "out_coeffs": 16},
        "train": {"batch_size": 1, "epochs": 25, "lr": 0.05, "weight_decay": 0.0},
        "loss": {"huber_delta": 1.0, "lambda_hf": 0.0, "lambda_w": 1.0, "lambda_bc": 0.0},
    }

    model = MambaCoeffNet(
        in_features=5,
        d_model=32,
        n_layers=2,
        dropout=0.0,
        out_coeffs=16,
        encoder_type="cnn",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model, optimizer, config, device=torch.device("cpu"))

    losses = []
    for _ in range(25):
        metrics = trainer.train_one_epoch(loader)
        losses.append(metrics["loss"])

    assert losses[-1] < losses[0] * 0.7
