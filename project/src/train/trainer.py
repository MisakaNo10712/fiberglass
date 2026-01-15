"""
Trainer for the fiber-sequence-to-coefficient regression loop.

工程背景:
- 输入序列 token X = [x, y, tx, ty, kappa_t]，模型预测系数 a (M*N)。
- 通过可微算子计算 κ_pred，并与测得 κ_meas 做一致性损失。
- 可选监督 w_points 与边界项，以及高频正则 hf_weights(a)。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.basis.dct2 import hf_weights, w_from_coeff
from src.losses.losses import total_loss
from src.operators.curvature_projection import kappa_t_from_coeff
from src.train.metrics import compute_metrics

__all__ = ["Trainer"]


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


class Trainer:
    """Lightweight trainer for training/evaluation and checkpointing."""

    def __init__(self, model, optimizer, config: dict, device: torch.device) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.history: list[dict[str, float]] = []

        basis_cfg = config.get("basis", {})
        self.M = int(basis_cfg.get("M"))
        self.N = int(basis_cfg.get("N"))
        self.Lx = float(basis_cfg.get("Lx"))
        self.Ly = float(basis_cfg.get("Ly"))
        self.hf_W = hf_weights(self.M, self.N, device=device, dtype=torch.float32)

        train_cfg = config.get("train", {})
        self.grad_clip = train_cfg.get("grad_clip")

        loss_cfg = config.get("loss", {})
        self.lambda_kappa = float(loss_cfg.get("lambda_kappa", 1.0))
        self.lambda_hf = float(loss_cfg.get("lambda_hf", 0.0))
        self.lambda_w = float(loss_cfg.get("lambda_w", 0.0))
        self.lambda_bc = float(loss_cfg.get("lambda_bc", 0.0))
        self.huber_delta = float(loss_cfg.get("huber_delta", 1.0))

        curriculum_cfg = config.get("curriculum", {})
        self.curriculum_enabled = bool(curriculum_cfg.get("enabled", False))
        self.stageA_steps = int(curriculum_cfg.get("stageA_steps", 0))
        self.stageB_warmup_steps = int(curriculum_cfg.get("stageB_warmup_steps", 0))
        self.global_step = 0

    def _move_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

    def _compute_kappa(
        self, a_reshaped: Tensor, x: Tensor, y: Tensor, tx: Tensor, ty: Tensor
    ) -> Tensor:
        if x.ndim == 2 and a_reshaped.ndim == 3 and x.shape[0] == a_reshaped.shape[0]:
            preds = []
            for i in range(a_reshaped.shape[0]):
                preds.append(
                    kappa_t_from_coeff(
                        a_reshaped[i], x[i], y[i], tx[i], ty[i], Lx=self.Lx, Ly=self.Ly
                    )
                )
            return torch.stack(preds, dim=0)
        return kappa_t_from_coeff(a_reshaped, x, y, tx, ty, Lx=self.Lx, Ly=self.Ly)

    def _compute_w(self, a_reshaped: Tensor, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim == 2 and a_reshaped.ndim == 3 and x.shape[0] == a_reshaped.shape[0]:
            preds = []
            for i in range(a_reshaped.shape[0]):
                preds.append(w_from_coeff(a_reshaped[i], x[i], y[i], Lx=self.Lx, Ly=self.Ly))
            return torch.stack(preds, dim=0)
        return w_from_coeff(a_reshaped, x, y, Lx=self.Lx, Ly=self.Ly)

    def _forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        X = batch["X"]
        mask = batch.get("mask")
        a = self.model(X, mask)
        if a.shape[-1] != self.M * self.N:
            raise ValueError("Model output coeff size does not match basis M*N.")
        a_reshaped = a.reshape(a.shape[0], self.M, self.N)

        x = batch.get("x", X[..., 0])
        y = batch.get("y", X[..., 1])
        tx = X[..., 2]
        ty = X[..., 3]

        kappa_pred = self._compute_kappa(a_reshaped, x, y, tx, ty)
        outputs = {"a": a, "a_reshaped": a_reshaped, "kappa_pred": kappa_pred}

        if "w_points" in batch or self.lambda_bc > 0:
            outputs["w_pred_points"] = self._compute_w(a_reshaped, x, y)

        return outputs

    def _boundary_mask(self, y: Tensor, mask: Tensor | None) -> Tensor:
        boundary = (y - self.Ly).abs() <= 1e-4
        if mask is None:
            return boundary
        return boundary & (mask > 0)

    def _step_loss(self, batch: dict[str, Tensor], outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        kappa_meas = batch["X"][..., 4]
        mask = batch.get("mask")
        w_true = batch.get("w_points")

        kappa_std = batch.get("kappa_std", 1.0)
        w_std = batch.get("w_std", 1.0)

        lambda_kappa, lambda_hf = self._scheduled_lambdas()

        bc_pred = None
        bc_mask = None
        if self.lambda_bc > 0 and outputs.get("w_pred_points") is not None:
            bc_pred = outputs["w_pred_points"]
            bc_mask = self._boundary_mask(batch.get("y", batch["X"][..., 1]), mask)

        return total_loss(
            outputs["kappa_pred"],
            kappa_meas,
            outputs["a_reshaped"],
            self.hf_W,
            mask=mask,
            w_pred_points=outputs.get("w_pred_points"),
            w_true_points=w_true,
            mask_w=mask,
            bc_pred=bc_pred,
            bc_mask=bc_mask,
            kappa_std=kappa_std,
            w_std=w_std,
            lambda_kappa=lambda_kappa,
            lambda_hf=lambda_hf,
            lambda_w=self.lambda_w,
            lambda_bc=self.lambda_bc,
            huber_delta=self.huber_delta,
        )

    def _scheduled_lambdas(self) -> tuple[float, float]:
        if not self.curriculum_enabled:
            return self.lambda_kappa, self.lambda_hf
        if self.global_step < self.stageA_steps:
            return 0.0, 0.0
        if self.stageB_warmup_steps <= 0:
            return self.lambda_kappa, self.lambda_hf
        progress = (self.global_step - self.stageA_steps + 1) / float(self.stageB_warmup_steps)
        progress = max(0.0, min(1.0, progress))
        return self.lambda_kappa * progress, self.lambda_hf * progress

    def train_one_epoch(self, dataloader) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        count = 0

        for batch in dataloader:
            batch = self._move_batch(batch)
            outputs = self._forward(batch)
            loss_dict = self._step_loss(batch, outputs)

            self.optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
            self.optimizer.step()
            self.global_step += 1

            metrics = compute_metrics(batch, outputs, self.config)
            batch_size = batch["X"].shape[0]
            count += batch_size
            for key, value in loss_dict.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size

        return {key: value / max(1, count) for key, value in totals.items()}

    def eval(self, dataloader) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch(batch)
                outputs = self._forward(batch)
                loss_dict = self._step_loss(batch, outputs)
                metrics = compute_metrics(batch, outputs, self.config)

                batch_size = batch["X"].shape[0]
                count += batch_size
                for key, value in loss_dict.items():
                    totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value) * batch_size

        return {key: value / max(1, count) for key, value in totals.items()}

    def record_epoch(
        self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float] | None = None
    ) -> dict[str, float]:
        row: dict[str, float] = {"epoch": float(epoch)}
        row.update(_prefix_metrics("train_", train_metrics))
        if val_metrics is not None:
            row.update(_prefix_metrics("val_", val_metrics))
        self.history.append(row)
        return row

    def save_history_csv(self, path: str | Path) -> None:
        if not self.history:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in self.history for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint
