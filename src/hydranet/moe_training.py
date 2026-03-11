"""Routerset-backed training and inference utilities for the student MoE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .loading import load_student_moe, save_student_moe_bundle
from .models.moe_student import MoEStudent, freeze_module

DEFAULT_ROUTERSET_EXPERTS = ("anomaly_detection", "fire", "worldfloods")


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def routerset_patch_token(row: Mapping[str, Any]) -> str:
    width = row["patch_width"] if row["patch_width"] is not None else "full"
    height = row["patch_height"] if row["patch_height"] is not None else "full"
    return f"{row['source_sample_id']}_{row['patch_x']}_{row['patch_y']}_{width}_{height}.npy"


def routerset_image_path(routerset_dir: str | Path, row: Mapping[str, Any]) -> Path:
    routerset_root = Path(routerset_dir)
    filename = routerset_patch_token(row)
    return routerset_root / "images" / row["source_dataset"] / row["source_split"] / filename


def is_routerset_source_compatible(source_dataset: str) -> bool:
    return source_dataset in set(DEFAULT_ROUTERSET_EXPERTS)


@dataclass
class RoutersetRecord:
    image_path: Path
    source_dataset: str
    source_sample_id: str
    source_split: str
    target: List[float]
    record_status: str
    label_names: List[str]


class RoutersetMoEDataset(Dataset):
    """Dataset adapter that turns routerset manifest rows into MoE routing targets."""

    def __init__(
        self,
        routerset_dir: str | Path,
        *,
        expert_names: Sequence[str],
        split: str = "train",
        target_size: int = 256,
        include_statuses: Sequence[str] = ("positive", "below_threshold", "explicit_negative"),
    ) -> None:
        self.routerset_dir = Path(routerset_dir)
        self.expert_names = list(expert_names)
        self.split = split
        self.target_size = int(target_size)
        self.include_statuses = set(include_statuses)
        self.records = self._load_records()

    def _load_records(self) -> List[RoutersetRecord]:
        manifest_path = self.routerset_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Routerset manifest not found at {manifest_path}")

        records: List[RoutersetRecord] = []
        expert_set = set(self.expert_names)
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            source_dataset = row["source_dataset"]
            if source_dataset not in expert_set:
                continue
            if source_dataset not in DEFAULT_ROUTERSET_EXPERTS:
                continue
            if row["source_split"] != self.split:
                continue
            if row["record_status"] not in self.include_statuses:
                continue

            image_path = routerset_image_path(self.routerset_dir, row)
            if not image_path.exists():
                continue

            target = [1.0 if name == source_dataset else 0.0 for name in self.expert_names]
            records.append(
                RoutersetRecord(
                    image_path=image_path,
                    source_dataset=source_dataset,
                    source_sample_id=row["source_sample_id"],
                    source_split=row["source_split"],
                    target=target,
                    record_status=row["record_status"],
                    label_names=list(row["label_names"]),
                )
            )

        if not records:
            raise ValueError(
                f"No routerset records matched split={self.split!r} and experts={self.expert_names!r}."
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path) -> torch.Tensor:
        array = np.load(path)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3D tensor in {path}, got shape {array.shape}")
        if array.shape[0] != 8:
            raise ValueError(f"Expected channel-first 8-channel data in {path}, got shape {array.shape}")
        image = torch.from_numpy(array).float()
        if image.shape[1:] != (self.target_size, self.target_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return image

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        return {
            "image": self._load_image(record.image_path),
            "target": torch.tensor(record.target, dtype=torch.float32),
            "expert_name": record.source_dataset,
            "source_sample_id": record.source_sample_id,
            "image_path": str(record.image_path),
            "record_status": record.record_status,
            "label_names": list(record.label_names),
        }

    def summary(self) -> dict[str, Any]:
        counts: Dict[str, int] = {name: 0 for name in self.expert_names}
        for record in self.records:
            counts[record.source_dataset] += 1
        return {
            "split": self.split,
            "num_records": len(self.records),
            "experts": list(self.expert_names),
            "counts_by_expert": counts,
            "target_size": self.target_size,
        }


class RoutersetMoEDataModule(pl.LightningDataModule):
    """LightningDataModule that reads local routerset routing targets."""

    def __init__(
        self,
        routerset_dir: str | Path,
        *,
        expert_names: Sequence[str],
        batch_size: int = 4,
        num_workers: int = 0,
        target_size: int = 256,
    ) -> None:
        super().__init__()
        self.routerset_dir = Path(routerset_dir)
        self.expert_names = list(expert_names)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.target_size = int(target_size)
        self.train_dataset: Optional[RoutersetMoEDataset] = None
        self.val_dataset: Optional[RoutersetMoEDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = RoutersetMoEDataset(
                self.routerset_dir,
                expert_names=self.expert_names,
                split="train",
                target_size=self.target_size,
            )
            self.val_dataset = RoutersetMoEDataset(
                self.routerset_dir,
                expert_names=self.expert_names,
                split="validation",
                target_size=self.target_size,
            )
        elif stage in (None, "predict", "validate"):
            self.val_dataset = RoutersetMoEDataset(
                self.routerset_dir,
                expert_names=self.expert_names,
                split="validation",
                target_size=self.target_size,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


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


def create_moe_run_config(
    *,
    routerset_dir: str | Path,
    output_dir: str | Path,
    expert_names: Optional[Sequence[str]] = None,
    training: str = "finetuning",
    n_shots: int = 5000,
    target_size: int = 256,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> dict[str, Any]:
    experts = list(expert_names or DEFAULT_ROUTERSET_EXPERTS)
    return {
        "routerset_dir": str(routerset_dir),
        "output_dir": str(output_dir),
        "expert_names": experts,
        "training": training,
        "n_shots": int(n_shots),
        "target_size": int(target_size),
        "threshold": float(threshold),
        "top_k": top_k if top_k is not None else len(experts),
        "timestamp": utc_timestamp(),
    }


def capture_baseline_summary(model: MoEStudent, output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    summary = {
        "encoder_source_task": model.encoder_source_task,
        "threshold": model.threshold,
        "top_k": model.top_k,
        "num_experts": model.num_experts,
        "experts": {
            name: {
                "output_channels": model.experts[name].output_channels,
                "parameters": sum(param.numel() for param in model.experts[name].parameters()),
            }
            for name in model.expert_names
        },
        "encoder_parameters": sum(param.numel() for param in model.encoder.parameters()),
        "switcher_parameters": sum(param.numel() for param in model.switcher.parameters()),
    }
    return save_json(Path(output_dir) / "baseline_summary.json", summary)


def build_routerset_moe(
    *,
    routerset_dir: str | Path,
    expert_names: Optional[Sequence[str]] = None,
    training: str = "finetuning",
    n_shots: int = 5000,
    weights_dir: Optional[str] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    auto_load_weights: bool = True,
) -> MoEStudent:
    _ = routerset_dir
    experts = list(expert_names or DEFAULT_ROUTERSET_EXPERTS)
    return load_student_moe(
        expert_tasks=experts,
        training=training,
        n_shots=n_shots,
        weights_dir=weights_dir,
        threshold=threshold,
        top_k=top_k,
        auto_load_weights=auto_load_weights,
    )


def write_routing_predictions(
    model: MoEStudent,
    dataloader: DataLoader,
    output_path: str | Path,
) -> Path:
    lines: List[str] = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            outputs = model(images)
            probs = outputs["routing_probs"].cpu().tolist()
            active = outputs["active_experts"]
            for index, sample_id in enumerate(batch["source_sample_id"]):
                record = {
                    "source_sample_id": sample_id,
                    "expert_name": batch["expert_name"][index],
                    "active_experts": active[index],
                    "routing_probs": {
                        name: float(probs[index][expert_index])
                        for expert_index, name in enumerate(model.expert_names)
                    },
                }
                lines.append(json.dumps(record))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def train_switcher(
    *,
    routerset_dir: str | Path,
    output_dir: str | Path,
    expert_names: Optional[Sequence[str]] = None,
    weights_dir: Optional[str] = None,
    training: str = "finetuning",
    n_shots: int = 5000,
    batch_size: int = 4,
    num_workers: int = 0,
    max_epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    target_size: int = 256,
    seed: int = 42,
    auto_load_weights: bool = True,
) -> dict[str, Any]:
    pl.seed_everything(seed, workers=True)
    output_dir = ensure_dir(output_dir)
    expert_names = list(expert_names or DEFAULT_ROUTERSET_EXPERTS)

    model = build_routerset_moe(
        routerset_dir=routerset_dir,
        expert_names=expert_names,
        training=training,
        n_shots=n_shots,
        weights_dir=weights_dir,
        threshold=threshold,
        top_k=top_k,
        auto_load_weights=auto_load_weights,
    )
    datamodule = RoutersetMoEDataModule(
        routerset_dir,
        expert_names=expert_names,
        batch_size=batch_size,
        num_workers=num_workers,
        target_size=target_size,
    )
    lightning_module = MoESwitcherLightningModule(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    config = create_moe_run_config(
        routerset_dir=routerset_dir,
        output_dir=output_dir,
        expert_names=expert_names,
        training=training,
        n_shots=n_shots,
        target_size=target_size,
        threshold=threshold,
        top_k=top_k,
    )
    save_json(Path(output_dir) / "config.json", config)
    capture_baseline_summary(model, output_dir)

    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(lightning_module, datamodule=datamodule)
    metrics = {key: float(value) for key, value in trainer.callback_metrics.items()}
    save_json(Path(output_dir) / "metrics.json", metrics)

    bundle_path = save_student_moe_bundle(model, Path(output_dir) / "student_moe_bundle.pt", metadata=config)
    datamodule.setup("fit")
    prediction_path = write_routing_predictions(model, datamodule.val_dataloader(), Path(output_dir) / "routing_predictions.jsonl")

    summary = {
        "bundle_path": str(bundle_path),
        "metrics_path": str(Path(output_dir) / "metrics.json"),
        "prediction_path": str(prediction_path),
        "config_path": str(Path(output_dir) / "config.json"),
    }
    save_json(Path(output_dir) / "summary.json", summary)
    return summary
