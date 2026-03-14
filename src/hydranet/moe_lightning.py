"""Lightning-only helpers for student MoE training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

from .models.moe_student import MoEStudent, freeze_module


class MoESwitcherLightningModule(pl.LightningModule):
    """Lightning wrapper that trains the MoE switcher against routing targets."""

    def __init__(
        self,
        model: MoEStudent,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_encoder: bool = True,
        freeze_experts: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        if freeze_encoder:
            freeze_module(self.model.encoder)
        if freeze_experts:
            for expert in self.model.experts.values():
                freeze_module(expert)

        self.save_hyperparameters(ignore=["model"])

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        return self.model(images)

    def _shared_step(self, batch: Mapping[str, Any], stage: str) -> torch.Tensor:
        outputs = self.model(batch["image"])
        targets = batch["target"]
        logits = outputs["routing_logits"]
        probs = outputs["routing_probs"]
        loss = self.loss_fn(logits, targets)
        preds = (probs >= self.model.threshold).float()
        if preds.sum(dim=1).eq(0).any():
            fallback = torch.argmax(logits, dim=1)
            fallback_targets = torch.zeros_like(preds)
            fallback_targets.scatter_(1, fallback.unsqueeze(1), 1.0)
            zero_rows = preds.sum(dim=1, keepdim=True).eq(0)
            preds = torch.where(zero_rows, fallback_targets, preds)

        tp = (preds * targets).sum()
        fp = (preds * (1.0 - targets)).sum()
        fn = ((1.0 - preds) * targets).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=targets.size(0))
        self.log(f"{stage}_precision", precision, prog_bar=(stage == "val"), batch_size=targets.size(0))
        self.log(f"{stage}_recall", recall, batch_size=targets.size(0))
        self.log(f"{stage}_f1", f1, prog_bar=(stage == "val"), batch_size=targets.size(0))
        return loss

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = [param for param in self.model.switcher.parameters() if param.requires_grad]
        return torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)


def build_trainer(
    output_dir: str | Path,
    *,
    max_epochs: int,
    accelerator: str = "cpu",
    devices: Union[int, str, list[int]] = 1,
    precision: Optional[str] = None,
) -> pl.Trainer:
    trainer_kwargs: dict[str, Any] = {
        "default_root_dir": str(output_dir),
        "max_epochs": max_epochs,
        "accelerator": accelerator,
        "devices": devices,
        "logger": False,
        "enable_checkpointing": False,
        "callbacks": [EarlyStopping(monitor="val_loss", patience=3, mode="min")],
    }
    if precision is not None:
        trainer_kwargs["precision"] = precision
    return pl.Trainer(
        **trainer_kwargs,
    )


def seed_everything(seed: int, *, workers: bool = True) -> int:
    return pl.seed_everything(seed, workers=workers)
