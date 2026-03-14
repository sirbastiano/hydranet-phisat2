"""Routerset-backed training, validation, and release helpers for student MoE models."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from csv import DictReader
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .loading import (
    _resolve_student_checkpoint_path,
    load_student_moe,
    load_student_moe_bundle,
    save_student_moe_bundle,
)
from .models.moe_student import MoEStudent

DEFAULT_ROUTERSET_EXPERTS = (
    "anomaly_detection",
    "burned_area",
    "fire",
    "lc",
    "roads",
    "worldfloods",
)
DEFAULT_ROUTERSET_TARGET_SIZE = 256
DEFAULT_RELEASE_ROOT = "outputs/phidranet"
DEFAULT_ROUTERSET_DATASET_SUBDIR = "multilabel_dataset"
DEFAULT_ROUTERSET_MANIFEST = "multilabel_dataset/manifest.jsonl"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60
DEFAULT_RUNTIME_SUBDIR = "runtime"
FULL_TRAINING_CONTRACT_VERSION = 1
FULL_TRAINING_STAGES = (
    "preflight",
    "startup_gate",
    "training",
    "export",
    "smoke",
)
RUN_ARTIFACT_FILENAMES = {
    "runtime_environment": "runtime_environment.json",
    "startup_log": "startup_log.txt",
    "startup_stage": "startup_stage.json",
    "preflight_report": "preflight_report.json",
    "startup_gate": "startup_gate.json",
    "config": "config.json",
    "dataset_report": "dataset_report.json",
    "checkpoint_report": "checkpoint_report.json",
    "baseline_summary": "baseline_summary.json",
    "metrics": "metrics.json",
    "bundle": "student_moe_bundle.pt",
    "routing_predictions": "routing_predictions.jsonl",
    "summary": "summary.json",
}
RELEASE_ARTIFACT_FILENAMES = {
    "bundle": "student_moe_bundle.pt",
    "config": "config.json",
    "metrics": "metrics.json",
    "baseline_summary": "baseline_summary.json",
    "routing_predictions": "routing_predictions.jsonl",
    "dataset_report": "dataset_report.json",
    "checkpoint_report": "checkpoint_report.json",
    "release_manifest": "release_manifest.json",
    "deployment_readme": "DEPLOY.md",
}


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Union[str, Path]) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: Union[str, Path], payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_text(path: Union[str, Path], text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def build_run_contract(
    *,
    output_dir: Union[str, Path],
    completed_stages: Sequence[str],
    release_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    unique_completed_stages = [stage for stage in FULL_TRAINING_STAGES if stage in set(completed_stages)]
    next_stage = next((stage for stage in FULL_TRAINING_STAGES if stage not in unique_completed_stages), None)
    contract: dict[str, Any] = {
        "version": FULL_TRAINING_CONTRACT_VERSION,
        "canonical_stages": list(FULL_TRAINING_STAGES),
        "completed_stages": unique_completed_stages,
        "next_stage": next_stage,
        "run_dir": str(output_dir),
        "run_artifacts": {name: str(output_dir / filename) for name, filename in RUN_ARTIFACT_FILENAMES.items()},
        "expected_artifact_names": {
            "run_dir": dict(RUN_ARTIFACT_FILENAMES),
            "release_dir": dict(RELEASE_ARTIFACT_FILENAMES),
        },
    }
    if release_dir is not None:
        release_dir = Path(release_dir)
        contract["release_dir"] = str(release_dir)
        contract["release_artifacts"] = {
            name: str(release_dir / filename) for name, filename in RELEASE_ARTIFACT_FILENAMES.items()
        }
    return contract


def validate_run_contract_artifacts(contract: Mapping[str, Any], *, required_sections: Sequence[str]) -> None:
    for section in required_sections:
        artifact_map = contract.get(section, {})
        missing = [
            name for name, artifact_path in artifact_map.items()
            if name != "summary" and not Path(str(artifact_path)).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing required {section} artifacts for full-training contract: {', '.join(sorted(missing))}."
            )


def completed_stages_for_failed_run(output_dir: Union[str, Path]) -> list[str]:
    output_dir = Path(output_dir)
    completed: list[str] = []
    if (output_dir / "preflight_report.json").exists():
        completed.append("preflight")
    startup_gate_path = output_dir / "startup_gate.json"
    if startup_gate_path.exists():
        gate_report = json.loads(startup_gate_path.read_text(encoding="utf-8"))
        probe_status = (
            gate_report.get("lightning_probe", {}).get("status")
            if isinstance(gate_report.get("lightning_probe"), dict)
            else None
        )
        if probe_status in {"ok", "skipped"}:
            completed.append("startup_gate")
    if (output_dir / "metrics.json").exists():
        completed.append("training")
    if (output_dir / "student_moe_bundle.pt").exists():
        completed.append("export")
    return completed


def normalize_trainer_devices(devices: Union[str, int, Sequence[int]]) -> Union[str, int, list[int]]:
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return [int(device) for device in devices]
    text = str(devices).strip()
    if not text:
        return 1
    if "," in text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    if text.isdigit():
        return int(text)
    return text


def resolve_runtime_root(
    *,
    output_dir: Union[str, Path],
    runtime_root: Optional[Union[str, Path]] = None,
) -> Path:
    if runtime_root is not None:
        return ensure_dir(runtime_root)
    return ensure_dir(Path(output_dir) / DEFAULT_RUNTIME_SUBDIR)


def configure_local_runtime_environment(
    *,
    output_dir: Union[str, Path],
    runtime_root: Optional[Union[str, Path]] = None,
    weights_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    resolved_runtime_root = resolve_runtime_root(output_dir=output_dir, runtime_root=runtime_root)
    cache_root = ensure_dir(resolved_runtime_root / "cache")
    hf_home = ensure_dir(cache_root / "huggingface")
    hf_hub_cache = ensure_dir(hf_home / "hub")
    hf_datasets_cache = ensure_dir(hf_home / "datasets")
    torch_home = ensure_dir(cache_root / "torch")
    tmp_dir = ensure_dir(resolved_runtime_root / "tmp")
    pycache_dir = ensure_dir(cache_root / "pycache")
    mplconfig_dir = ensure_dir(cache_root / "matplotlib")
    resolved_weights_dir = ensure_dir(weights_dir or (resolved_runtime_root / "weights"))

    env_updates = {
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str(hf_hub_cache),
        "HF_DATASETS_CACHE": str(hf_datasets_cache),
        "TORCH_HOME": str(torch_home),
        "XDG_CACHE_HOME": str(cache_root),
        "TMPDIR": str(tmp_dir),
        "TMP": str(tmp_dir),
        "TEMP": str(tmp_dir),
        "PYTHONPYCACHEPREFIX": str(pycache_dir),
        "MPLCONFIGDIR": str(mplconfig_dir),
        "HF_HUB_DISABLE_XET": "1",
        "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
    }
    for key, value in env_updates.items():
        os.environ[key] = value

    return {
        "runtime_root": str(resolved_runtime_root),
        "weights_dir": str(resolved_weights_dir),
        "cache_root": str(cache_root),
        "hf_home": str(hf_home),
        "hf_hub_cache": str(hf_hub_cache),
        "hf_datasets_cache": str(hf_datasets_cache),
        "torch_home": str(torch_home),
        "tmp_dir": str(tmp_dir),
        "pycache_dir": str(pycache_dir),
        "mplconfig_dir": str(mplconfig_dir),
        "env": env_updates,
    }


@dataclass
class StartupRecorder:
    output_dir: Path
    current_stage: str = "init"
    failure_stage: Optional[str] = None

    def __post_init__(self) -> None:
        self.output_dir = ensure_dir(self.output_dir)
        self.log_path = self.output_dir / "startup_log.txt"
        self.stage_path = self.output_dir / "startup_stage.json"

    def mark(self, stage: str, **payload: Any) -> None:
        self.current_stage = stage
        entry = {"timestamp": utc_timestamp(), "stage": stage, **payload}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        save_json(
            self.stage_path,
            {
                "current_stage": self.current_stage,
                "failure_stage": self.failure_stage,
                "last_entry": entry,
            },
        )

    def fail(self, stage: str, error: BaseException) -> None:
        self.failure_stage = stage
        self.mark(
            stage,
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback="".join(traceback.format_exception(type(error), error, error.__traceback__)),
        )


def resolve_routerset_dataset_root(routerset_dir: Union[str, Path]) -> Path:
    return Path(routerset_dir) / DEFAULT_ROUTERSET_DATASET_SUBDIR


def default_rebuilt_manifest_path(routerset_dir: Union[str, Path]) -> Path:
    return resolve_routerset_dataset_root(routerset_dir) / "manifest_moe_train.jsonl"


def routerset_patch_token(row: Mapping[str, Any]) -> str:
    width = row["patch_width"] if row["patch_width"] is not None else "full"
    height = row["patch_height"] if row["patch_height"] is not None else "full"
    return f"{row['source_sample_id']}_{row['patch_x']}_{row['patch_y']}_{height}_{width}.npy"


def routerset_image_path(routerset_dir: Union[str, Path], row: Mapping[str, Any]) -> Path:
    routerset_root = resolve_routerset_dataset_root(routerset_dir)
    filename = routerset_patch_token(row)
    return routerset_root / "images" / row["source_dataset"] / row["source_split"] / filename


def resolve_routerset_manifest_path(
    routerset_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
) -> Path:
    if manifest_path is not None:
        return Path(manifest_path)
    return Path(routerset_dir) / DEFAULT_ROUTERSET_MANIFEST


def load_routerset_manifest_rows(
    routerset_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
) -> List[dict[str, Any]]:
    path = resolve_routerset_manifest_path(routerset_dir, manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Routerset manifest not found at {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _routerset_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("source_sample_id", "")),
        int(row.get("patch_x") or 0),
        int(row.get("patch_y") or 0),
        str(row.get("patch_width")),
        str(row.get("patch_height")),
    )


def rebuild_routerset_split_manifest(
    routerset_dir: Union[str, Path],
    *,
    expert_names: Sequence[str],
    manifest_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    rows = load_routerset_manifest_rows(routerset_dir, manifest_path)
    expert_set = set(expert_names)
    rebuilt_rows: List[dict[str, Any]] = []

    for expert_name in expert_names:
        expert_rows = [dict(row) for row in rows if row["source_dataset"] == expert_name]
        if not expert_rows:
            continue

        for row in expert_rows:
            row["moe_split"] = row.get("moe_split", row["source_split"])

        positive_rows = sorted(
            [row for row in expert_rows if row["record_status"] == "positive"],
            key=_routerset_row_sort_key,
        )
        if len(positive_rows) < 2:
            raise ValueError(
                f"Cannot rebuild routerset split for {expert_name!r}; need at least two positive rows, found {len(positive_rows)}."
            )

        train_positive = [row for row in positive_rows if row.get("moe_split", row["source_split"]) == "train"]
        val_positive = [row for row in positive_rows if row.get("moe_split", row["source_split"]) == "validation"]
        if train_positive and val_positive:
            rebuilt_rows.extend(expert_rows)
            continue

        val_quota = max(1, int(round(len(positive_rows) * 0.2)))
        val_quota = min(val_quota, len(positive_rows) - 1)
        validation_keys = {
            (
                row["source_sample_id"],
                row.get("patch_x"),
                row.get("patch_y"),
                row.get("patch_width"),
                row.get("patch_height"),
            )
            for row in positive_rows[:val_quota]
        }

        for row in rows:
            if row["source_dataset"] != expert_name:
                continue
            candidate = dict(row)
            candidate["moe_split"] = candidate.get("moe_split", candidate["source_split"])
            key = (
                candidate["source_sample_id"],
                candidate.get("patch_x"),
                candidate.get("patch_y"),
                candidate.get("patch_width"),
                candidate.get("patch_height"),
            )
            if candidate["record_status"] == "positive":
                candidate["moe_split"] = "validation" if key in validation_keys else "train"
            rebuilt_rows.append(candidate)

    if not rebuilt_rows:
        rebuilt_rows = [dict(row) for row in rows if row["source_dataset"] in expert_set]
        for row in rebuilt_rows:
            row["moe_split"] = row.get("moe_split", row["source_split"])

    passthrough_rows = [dict(row) for row in rows if row["source_dataset"] not in expert_set]
    for row in passthrough_rows:
        row["moe_split"] = row.get("moe_split", row["source_split"])
    final_rows = rebuilt_rows + passthrough_rows
    final_rows.sort(
        key=lambda row: (
            str(row["source_dataset"]),
            _routerset_row_sort_key(row),
            str(row.get("moe_split", row["source_split"])),
        )
    )

    destination = Path(output_path) if output_path is not None else default_rebuilt_manifest_path(routerset_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(json.dumps(row, sort_keys=True) for row in final_rows) + "\n", encoding="utf-8")
    return destination


def _normalize_to_channel_first(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {array.shape}")

    if array.shape[-1] <= 16 and array.shape[-1] < min(array.shape[0], array.shape[1]):
        return np.moveaxis(array, -1, 0)

    if array.shape[0] <= 16 and array.shape[0] < min(array.shape[1], array.shape[2]):
        return array

    if array.shape[-1] <= 16:
        return np.moveaxis(array, -1, 0)

    if array.shape[0] <= 16:
        return array

    raise ValueError(f"Could not infer channel axis for shape {array.shape}")


def normalize_routerset_array(
    array: np.ndarray,
    *,
    source_dataset: str,
    target_channels: int = 8,
) -> torch.Tensor:
    """Normalize any routerset source tensor into channel-first 8-channel float32."""
    current = np.asarray(array)
    current = _normalize_to_channel_first(current)
    current = current.astype(np.float32, copy=False)

    # Deterministic v1 adapter rules by source family.
    if source_dataset in {"lc", "roads"} and current.shape[0] >= target_channels:
        current = current[:target_channels]
    elif current.shape[0] > target_channels:
        current = current[:target_channels]
    elif current.shape[0] < target_channels:
        padding = np.zeros(
            (target_channels - current.shape[0], current.shape[1], current.shape[2]),
            dtype=current.dtype,
        )
        current = np.concatenate([current, padding], axis=0)

    return torch.from_numpy(current)


def describe_routerset_array(
    path: Union[str, Path],
    *,
    source_dataset: str,
    target_channels: int = 8,
) -> dict[str, Any]:
    array = np.load(path)
    normalized = normalize_routerset_array(array, source_dataset=source_dataset, target_channels=target_channels)
    return {
        "path": str(path),
        "original_shape": list(array.shape),
        "original_dtype": str(array.dtype),
        "normalized_shape": list(normalized.shape),
        "source_dataset": source_dataset,
    }


def resolve_release_name(release_name: Optional[str] = None) -> str:
    suffix = release_name or utc_timestamp()
    return f"phidranet_{suffix}"


def _student_checkpoint_catalog_path() -> Path:
    return Path(__file__).with_name("model_weights.csv")


def _select_student_checkpoint_catalog_entry(
    *,
    task: str,
    training: str,
    n_shots: int,
) -> Optional[dict[str, str]]:
    catalog_path = _student_checkpoint_catalog_path()
    if not catalog_path.exists():
        return None
    with catalog_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in DictReader(handle)
            if row.get("model") == "student"
            and row.get("task") == task
            and row.get("training") == training
            and row.get("n_shots") == str(float(n_shots))
        ]
    if not rows:
        return None
    rows.sort(key=lambda row: row.get("datetime", ""), reverse=True)
    return rows[0]


def resolve_student_checkpoint_report(
    expert_names: Sequence[str],
    *,
    training: str,
    n_shots: int,
    weights_dir: Optional[str] = None,
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for task in expert_names:
        catalog_entry = _select_student_checkpoint_catalog_entry(
            task=task,
            training=training,
            n_shots=n_shots,
        )
        source_path = catalog_entry["file_path"] if catalog_entry is not None else ""
        deterministic_path = (
            str(Path(weights_dir) / source_path)
            if weights_dir is not None and source_path
            else ""
        )
        item: dict[str, Any] = {
            "status": "ok",
            "training": training,
            "n_shots": int(n_shots),
            "catalog_path": str(_student_checkpoint_catalog_path()),
            "source_path": source_path,
            "deterministic_path": deterministic_path,
            "checkpoint_path": "",
        }
        try:
            checkpoint_path = _resolve_student_checkpoint_path(
                task=task,
                training=training,
                n_shots=n_shots,
                weights_dir=weights_dir,
            )
            if not checkpoint_path:
                raise FileNotFoundError(f"Checkpoint path was empty for expert {task!r}.")
            item["checkpoint_path"] = checkpoint_path
            if not item["deterministic_path"]:
                item["deterministic_path"] = checkpoint_path
        except BaseException as error:
            item["status"] = "failed"
            item["error_type"] = type(error).__name__
            item["error"] = str(error)
        report[task] = item
    return report


@dataclass
class RoutersetRecord:
    row_token: str
    image_path: Path
    source_dataset: str
    base_source_sample_id: str
    source_sample_id: str
    source_split: str
    target: List[float]
    record_status: str
    label_names: List[str]
    original_shape: List[int]
    normalized_shape: List[int]
    training_shape: List[int]
    tile_origin: List[int]
    tile_size: List[int]
    is_tiled: bool


def build_routerset_training_tensor(
    array: np.ndarray,
    *,
    source_dataset: str,
    target_size: int,
    target_channels: int = 8,
) -> torch.Tensor:
    image = normalize_routerset_array(
        array,
        source_dataset=source_dataset,
        target_channels=target_channels,
    ).float()
    return crop_or_pad_routerset_tensor(image, target_size=target_size)


def crop_or_pad_routerset_tensor(image: torch.Tensor, *, target_size: int) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected channel-first tensor, got shape {tuple(image.shape)}")

    _, height, width = image.shape
    if height > target_size:
        crop_top = (height - target_size) // 2
        image = image[:, crop_top : crop_top + target_size, :]
    if width > target_size:
        crop_left = (width - target_size) // 2
        image = image[:, :, crop_left : crop_left + target_size]

    _, height, width = image.shape
    if height < target_size or width < target_size:
        padded = torch.zeros(
            (image.shape[0], target_size, target_size),
            dtype=image.dtype,
            device=image.device,
        )
        offset_y = (target_size - height) // 2
        offset_x = (target_size - width) // 2
        padded[:, offset_y : offset_y + height, offset_x : offset_x + width] = image
        image = padded

    return image


def anomaly_tile_origins(height: int, width: int, target_size: int) -> List[tuple[int, int]]:
    if height <= target_size or width <= target_size:
        return [(0, 0)]

    step = target_size
    y_starts = list(range(0, max(height - target_size, 0) + 1, step))
    x_starts = list(range(0, max(width - target_size, 0) + 1, step))

    if y_starts[-1] != height - target_size:
        y_starts.append(height - target_size)
    if x_starts[-1] != width - target_size:
        x_starts.append(width - target_size)

    return [(y, x) for y in y_starts for x in x_starts]


class RoutersetMoEDataset(Dataset):
    """Dataset adapter that turns routerset manifest rows into MoE routing targets."""

    def __init__(
        self,
        routerset_dir: Union[str, Path],
        *,
        manifest_path: Optional[Union[str, Path]] = None,
        expert_names: Sequence[str],
        split: str = "train",
        target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
        target_channels: int = 8,
        include_statuses: Sequence[str] = ("positive", "below_threshold", "explicit_negative"),
    ) -> None:
        self.routerset_dir = Path(routerset_dir)
        self.manifest_path = resolve_routerset_manifest_path(routerset_dir, manifest_path)
        self.expert_names = list(expert_names)
        self.split = split
        self.target_size = int(target_size)
        self.target_channels = int(target_channels)
        self.include_statuses = set(include_statuses)
        self.records = self._load_records()

    def _load_records(self) -> List[RoutersetRecord]:
        records: List[RoutersetRecord] = []
        expert_set = set(self.expert_names)
        for row in load_routerset_manifest_rows(self.routerset_dir, self.manifest_path):
            source_dataset = row["source_dataset"]
            if source_dataset not in expert_set:
                continue
            if row.get("moe_split", row["source_split"]) != self.split:
                continue
            if row["record_status"] not in self.include_statuses:
                continue

            image_path = routerset_image_path(self.routerset_dir, row)
            if not image_path.exists():
                continue

            description = describe_routerset_array(
                image_path,
                source_dataset=source_dataset,
                target_channels=self.target_channels,
            )
            target = [1.0 if name == source_dataset else 0.0 for name in self.expert_names]
            normalized_shape = list(description["normalized_shape"])
            tile_origins = [(0, 0)]
            if source_dataset == "anomaly_detection":
                _, height, width = normalized_shape
                tile_origins = anomaly_tile_origins(height, width, self.target_size)

            for tile_y, tile_x in tile_origins:
                tile_suffix = ""
                if source_dataset == "anomaly_detection":
                    tile_suffix = f"_tile_{tile_y}_{tile_x}"
                records.append(
                    RoutersetRecord(
                        row_token=routerset_patch_token(row),
                        image_path=image_path,
                        source_dataset=source_dataset,
                        base_source_sample_id=row["source_sample_id"],
                        source_sample_id=f"{row['source_sample_id']}{tile_suffix}",
                        source_split=row.get("moe_split", row["source_split"]),
                        target=target,
                        record_status=row["record_status"],
                        label_names=list(row["label_names"]),
                        original_shape=list(description["original_shape"]),
                        normalized_shape=normalized_shape,
                        training_shape=[self.target_channels, self.target_size, self.target_size],
                        tile_origin=[tile_y, tile_x],
                        tile_size=[self.target_size, self.target_size],
                        is_tiled=len(tile_origins) > 1,
                    )
                )

        if not records:
            raise ValueError(
                f"No routerset records matched split={self.split!r} and experts={self.expert_names!r}."
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, record: RoutersetRecord) -> torch.Tensor:
        array = np.load(record.image_path)
        normalized = normalize_routerset_array(
            array,
            source_dataset=record.source_dataset,
            target_channels=self.target_channels,
        ).float()
        tile_y, tile_x = record.tile_origin
        tile_height, tile_width = record.tile_size
        if record.source_dataset == "anomaly_detection":
            normalized = normalized[:, tile_y : tile_y + tile_height, tile_x : tile_x + tile_width]
        return build_routerset_training_tensor(
            normalized.numpy(),
            source_dataset=record.source_dataset,
            target_size=self.target_size,
            target_channels=self.target_channels,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        return {
            "image": self._load_image(record),
            "target": torch.tensor(record.target, dtype=torch.float32),
            "expert_name": record.source_dataset,
            "source_sample_id": record.source_sample_id,
            "image_path": str(record.image_path),
            "record_status": record.record_status,
            "tile_origin": torch.tensor(record.tile_origin, dtype=torch.int64),
        }

    def summary(self) -> dict[str, Any]:
        counts: Dict[str, int] = {name: 0 for name in self.expert_names}
        raw_counts: Dict[str, int] = {name: 0 for name in self.expert_names}
        statuses: Dict[str, int] = {}
        original_shapes: Dict[str, Dict[str, int]] = {}
        normalized_shapes: Dict[str, Dict[str, int]] = {}
        training_shapes: Dict[str, Dict[str, int]] = {}
        positive_counts: Dict[str, int] = {name: 0 for name in self.expert_names}
        raw_positive_counts: Dict[str, int] = {name: 0 for name in self.expert_names}
        tiled_counts: Dict[str, int] = {name: 0 for name in self.expert_names}
        seen_rows: set[tuple[str, str]] = set()
        seen_positive_rows: set[tuple[str, str]] = set()
        for record in self.records:
            counts[record.source_dataset] += 1
            statuses[record.record_status] = statuses.get(record.record_status, 0) + 1
            original_shapes.setdefault(record.source_dataset, {})
            normalized_shapes.setdefault(record.source_dataset, {})
            training_shapes.setdefault(record.source_dataset, {})
            original_key = "x".join(str(x) for x in record.original_shape)
            normalized_key = "x".join(str(x) for x in record.normalized_shape)
            training_key = "x".join(str(x) for x in record.training_shape)
            original_shapes[record.source_dataset][original_key] = (
                original_shapes[record.source_dataset].get(original_key, 0) + 1
            )
            normalized_shapes[record.source_dataset][normalized_key] = (
                normalized_shapes[record.source_dataset].get(normalized_key, 0) + 1
            )
            training_shapes[record.source_dataset][training_key] = (
                training_shapes[record.source_dataset].get(training_key, 0) + 1
            )
            if record.record_status == "positive":
                positive_counts[record.source_dataset] += 1
            if record.is_tiled:
                tiled_counts[record.source_dataset] += 1
            raw_key = (record.source_dataset, record.row_token)
            if raw_key not in seen_rows:
                seen_rows.add(raw_key)
                raw_counts[record.source_dataset] += 1
            if record.record_status == "positive" and raw_key not in seen_positive_rows:
                seen_positive_rows.add(raw_key)
                raw_positive_counts[record.source_dataset] += 1
        return {
            "split": self.split,
            "manifest_path": str(self.manifest_path),
            "num_records": len(self.records),
            "experts": list(self.expert_names),
            "counts_by_expert": counts,
            "expanded_counts_by_expert": counts,
            "raw_counts_by_expert": raw_counts,
            "positive_counts_by_expert": positive_counts,
            "expanded_positive_counts_by_expert": positive_counts,
            "raw_positive_counts_by_expert": raw_positive_counts,
            "generated_tiles_by_expert": tiled_counts,
            "record_status_counts": statuses,
            "target_size": self.target_size,
            "target_channels": self.target_channels,
            "original_shapes_by_expert": original_shapes,
            "normalized_shapes_by_expert": normalized_shapes,
            "training_shapes_by_expert": training_shapes,
        }


class RoutersetMoEDataModule:
    """Plain data helper that reads local routerset routing targets."""

    def __init__(
        self,
        routerset_dir: Union[str, Path],
        *,
        manifest_path: Optional[Union[str, Path]] = None,
        expert_names: Sequence[str],
        batch_size: int = 4,
        num_workers: int = 0,
        target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
        target_channels: int = 8,
        balanced_sampling: bool = True,
    ) -> None:
        self.routerset_dir = Path(routerset_dir)
        self.manifest_path = resolve_routerset_manifest_path(routerset_dir, manifest_path)
        self.expert_names = list(expert_names)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.target_size = int(target_size)
        self.target_channels = int(target_channels)
        self.balanced_sampling = bool(balanced_sampling)
        self.train_dataset: Optional[RoutersetMoEDataset] = None
        self.val_dataset: Optional[RoutersetMoEDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = RoutersetMoEDataset(
                self.routerset_dir,
                manifest_path=self.manifest_path,
                expert_names=self.expert_names,
                split="train",
                target_size=self.target_size,
                target_channels=self.target_channels,
            )
            self.val_dataset = RoutersetMoEDataset(
                self.routerset_dir,
                manifest_path=self.manifest_path,
                expert_names=self.expert_names,
                split="validation",
                target_size=self.target_size,
                target_channels=self.target_channels,
            )
        elif stage in (None, "predict", "validate"):
            self.val_dataset = RoutersetMoEDataset(
                self.routerset_dir,
                manifest_path=self.manifest_path,
                expert_names=self.expert_names,
                split="validation",
                target_size=self.target_size,
                target_channels=self.target_channels,
            )

    def _train_sample_weights(self) -> List[float]:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        counts: Dict[str, int] = defaultdict(int)
        for record in self.train_dataset.records:
            counts[record.source_dataset] += 1
        return [1.0 / counts[record.source_dataset] for record in self.train_dataset.records]

    def train_sampling_report(self) -> Dict[str, Any]:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        counts: Dict[str, int] = defaultdict(int)
        for record in self.train_dataset.records:
            counts[record.source_dataset] += 1
        return {
            "balanced_sampling": self.balanced_sampling,
            "expanded_counts_by_expert": dict(counts),
            "class_weights": {expert: (1.0 / counts[expert]) for expert in counts},
            "num_samples": len(self.train_dataset.records),
        }

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        sampler = None
        shuffle = True
        if self.balanced_sampling:
            sample_weights = self._train_sample_weights()
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            sampler=sampler,
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

    def summary(self) -> dict[str, Any]:
        if self.train_dataset is None or self.val_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        assert self.val_dataset is not None
        return {
            "train": self.train_dataset.summary(),
            "validation": self.val_dataset.summary(),
            "sampling": self.train_sampling_report(),
        }

def _lightning_import_probe_code() -> str:
    return (
        "import os; "
        "os.environ.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '1'); "
        "os.environ.setdefault('CUDA_VISIBLE_DEVICES', ''); "
        "import lightning_fabric; "
        "import pytorch_lightning"
    )


def probe_lightning_import(*, timeout_seconds: int = DEFAULT_STARTUP_TIMEOUT_SECONDS) -> None:
    env = dict(os.environ)
    env.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        result = subprocess.run(
            [sys.executable, "-c", _lightning_import_probe_code()],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            "Lightning import probe timed out before trainer startup completed."
        ) from error

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"Lightning import probe failed: {detail}")


def run_training_startup_gate(
    *,
    routerset_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    expert_names: Sequence[str],
    target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
    target_channels: int = 8,
    balanced_sampling: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    startup_timeout_seconds: int = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    run_lightning_probe: bool = True,
) -> dict[str, Any]:
    active_manifest_path = resolve_routerset_manifest_path(routerset_dir, manifest_path)
    manifest_rows = load_routerset_manifest_rows(routerset_dir, active_manifest_path)
    dataset = RoutersetMoEDataset(
        routerset_dir,
        manifest_path=active_manifest_path,
        expert_names=expert_names,
        split="train",
        target_size=target_size,
        target_channels=target_channels,
    )
    sample = dataset[0]
    gate_report: dict[str, Any] = {
        "status": "pending",
        "routerset_dir": str(routerset_dir),
        "dataset_root": str(resolve_routerset_dataset_root(routerset_dir)),
        "manifest_path": str(active_manifest_path),
        "manifest_row_count": len(manifest_rows),
        "target_size": int(target_size),
        "target_channels": int(target_channels),
        "balanced_sampling": bool(balanced_sampling),
        "timeout_seconds": int(startup_timeout_seconds),
        "torch_version": torch.__version__,
        "train_record_count": len(dataset.records),
        "sample_source_sample_id": sample["source_sample_id"],
        "sample_expert_name": sample["expert_name"],
        "sample_image_shape": list(sample["image"].shape),
        "sample_target_shape": list(sample["target"].shape),
        "sample_tile_origin": sample["tile_origin"].tolist(),
    }

    if run_lightning_probe:
        try:
            probe_lightning_import(timeout_seconds=startup_timeout_seconds)
            gate_report["status"] = "ok"
            gate_report["lightning_probe"] = {
                "status": "ok",
                "timeout_seconds": int(startup_timeout_seconds),
            }
        except BaseException as error:
            gate_report["status"] = "failed"
            gate_report["error_type"] = type(error).__name__
            gate_report["error"] = str(error)
            gate_report["traceback"] = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            gate_report["lightning_probe"] = {
                "status": "failed",
                "timeout_seconds": int(startup_timeout_seconds),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": gate_report["traceback"],
            }
            if output_dir is not None:
                save_json(Path(output_dir) / "startup_gate.json", gate_report)
            raise
    else:
        gate_report["status"] = "skipped"
        gate_report["lightning_probe"] = {
            "status": "skipped",
            "timeout_seconds": int(startup_timeout_seconds),
        }

    if output_dir is not None:
        save_json(Path(output_dir) / "startup_gate.json", gate_report)
    return gate_report


def load_lightning_training_components() -> tuple[Any, Any, Any]:
    from .moe_lightning import (
        MoESwitcherLightningModule as LightningModuleClass,
        build_trainer,
        seed_everything,
    )

    return LightningModuleClass, build_trainer, seed_everything


class MoESwitcherLightningModule:
    """Lazy proxy that defers Lightning imports until explicitly constructed."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        lightning_module_cls, _, _ = load_lightning_training_components()
        return lightning_module_cls(*args, **kwargs)


def create_moe_run_config(
    *,
    routerset_dir: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    expert_names: Optional[Sequence[str]] = None,
    training: str = "finetuning",
    n_shots: int = 5000,
    target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
    target_channels: int = 8,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    release_name: Optional[str] = None,
    balanced_sampling: bool = True,
    runtime_root: Optional[Union[str, Path]] = None,
    weights_dir: Optional[Union[str, Path]] = None,
    accelerator: str = "cpu",
    devices: Union[str, int, Sequence[int]] = 1,
    precision: Optional[str] = None,
) -> dict[str, Any]:
    experts = list(expert_names or DEFAULT_ROUTERSET_EXPERTS)
    return {
        "routerset_dir": str(routerset_dir),
        "output_dir": str(output_dir),
        "manifest_path": str(resolve_routerset_manifest_path(routerset_dir, manifest_path)),
        "expert_names": experts,
        "training": training,
        "n_shots": int(n_shots),
        "target_size": int(target_size),
        "target_channels": int(target_channels),
        "threshold": float(threshold),
        "top_k": top_k if top_k is not None else len(experts),
        "balanced_sampling": bool(balanced_sampling),
        "runtime_root": str(runtime_root) if runtime_root is not None else "",
        "weights_dir": str(weights_dir) if weights_dir is not None else "",
        "trainer": {
            "accelerator": accelerator,
            "devices": normalize_trainer_devices(devices),
            "precision": precision,
        },
        "timestamp": utc_timestamp(),
        "release_name": resolve_release_name(release_name),
    }


def capture_baseline_summary(model: MoEStudent, output_dir: Union[str, Path]) -> Path:
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
    routerset_dir: Union[str, Path],
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


def collect_routerset_dataset_report(
    routerset_dir: Union[str, Path],
    *,
    manifest_path: Optional[Union[str, Path]] = None,
    expert_names: Sequence[str],
    target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
    target_channels: int = 8,
    balanced_sampling: bool = True,
) -> dict[str, Any]:
    datamodule = RoutersetMoEDataModule(
        routerset_dir,
        manifest_path=manifest_path,
        expert_names=expert_names,
        batch_size=1,
        num_workers=0,
        target_size=target_size,
        target_channels=target_channels,
        balanced_sampling=balanced_sampling,
    )
    datamodule.setup("fit")
    return datamodule.summary()


def validate_routerset_dataset_report(
    dataset_report: Mapping[str, Any],
    *,
    expert_names: Sequence[str],
    target_size: int,
    target_channels: int,
) -> None:
    expected_shape = f"{target_channels}x{target_size}x{target_size}"
    empty_splits: List[str] = []
    missing_records: Dict[str, List[str]] = {}
    missing_positive: Dict[str, List[str]] = {}
    invalid_shapes: Dict[str, Dict[str, Mapping[str, int]]] = {}

    for split_name in ("train", "validation"):
        split_report = dataset_report[split_name]
        if int(split_report.get("num_records", 0)) <= 0:
            empty_splits.append(split_name)
        counts = split_report["raw_counts_by_expert"]
        positive_counts = split_report["raw_positive_counts_by_expert"]
        for expert_name in expert_names:
            if counts.get(expert_name, 0) <= 0:
                missing_records.setdefault(expert_name, []).append(split_name)
            if positive_counts.get(expert_name, 0) <= 0:
                missing_positive.setdefault(expert_name, []).append(split_name)

        for expert_name, shapes in split_report["training_shapes_by_expert"].items():
            if list(shapes.keys()) != [expected_shape]:
                invalid_shapes.setdefault(split_name, {})[expert_name] = shapes

    if empty_splits:
        joined = ", ".join(empty_splits)
        raise ValueError(f"Routerset manifest produced empty required split(s): {joined}.")

    if missing_records:
        details = ", ".join(
            f"{expert} ({'/'.join(splits)})" for expert, splits in sorted(missing_records.items())
        )
        raise ValueError(
            "Routerset manifest is missing required experts in the selected splits: "
            f"{details}."
        )

    if missing_positive:
        details = ", ".join(
            f"{expert} ({'/'.join(splits)})" for expert, splits in sorted(missing_positive.items())
        )
        raise ValueError(
            "Routerset split rebuild required before canonical training; missing positive samples for "
            f"{details}."
        )

    if invalid_shapes:
        raise ValueError(
            "Routerset adaptation produced non-canonical training shapes: "
            f"{json.dumps(invalid_shapes, sort_keys=True)}"
        )


def validate_student_checkpoint_report(
    checkpoint_report: Mapping[str, Mapping[str, Any]],
    *,
    expert_names: Sequence[str],
) -> None:
    missing_entries = [expert for expert in expert_names if expert not in checkpoint_report]
    if missing_entries:
        joined = ", ".join(sorted(missing_entries))
        raise ValueError(f"Checkpoint report is missing required experts: {joined}.")

    failures: List[str] = []
    for expert_name in expert_names:
        payload = checkpoint_report[expert_name]
        if payload.get("status") == "ok" and payload.get("checkpoint_path"):
            continue
        source_hint = (
            str(payload.get("source_path") or "")
            or str(payload.get("deterministic_path") or "")
            or str(payload.get("catalog_path") or "")
        )
        error = str(payload.get("error") or "checkpoint resolution failed")
        failures.append(f"{expert_name} from {source_hint}: {error}")

    if failures:
        details = "; ".join(failures)
        raise ValueError(f"Missing required expert checkpoint set: {details}")


def preflight_routerset_training(
    *,
    routerset_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    expert_names: Optional[Sequence[str]] = None,
    weights_dir: Optional[str] = None,
    training: str = "finetuning",
    n_shots: int = 5000,
    target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
    target_channels: int = 8,
    output_dir: Optional[Union[str, Path]] = None,
    release_name: Optional[str] = None,
    rebuild_splits: bool = False,
    rebuilt_manifest_out: Optional[Union[str, Path]] = None,
    balanced_sampling: bool = True,
) -> dict[str, Any]:
    experts = list(expert_names or DEFAULT_ROUTERSET_EXPERTS)
    active_manifest_path = resolve_routerset_manifest_path(routerset_dir, manifest_path)
    rebuilt_manifest_path = None
    dataset_report_path = Path(output_dir) / "dataset_report.json" if output_dir is not None else None
    checkpoint_report_path = Path(output_dir) / "checkpoint_report.json" if output_dir is not None else None
    if rebuild_splits:
        rebuilt_manifest_path = rebuild_routerset_split_manifest(
            routerset_dir,
            expert_names=experts,
            manifest_path=active_manifest_path,
            output_path=rebuilt_manifest_out,
        )
        active_manifest_path = rebuilt_manifest_path
    dataset_report = collect_routerset_dataset_report(
        routerset_dir,
        manifest_path=active_manifest_path,
        expert_names=experts,
        target_size=target_size,
        target_channels=target_channels,
        balanced_sampling=balanced_sampling,
    )
    validate_routerset_dataset_report(
        dataset_report,
        expert_names=experts,
        target_size=target_size,
        target_channels=target_channels,
    )
    checkpoint_report = resolve_student_checkpoint_report(
        experts,
        training=training,
        n_shots=n_shots,
        weights_dir=weights_dir,
    )
    if dataset_report_path is not None:
        save_json(dataset_report_path, dataset_report)
    if checkpoint_report_path is not None:
        save_json(checkpoint_report_path, checkpoint_report)
    validate_student_checkpoint_report(checkpoint_report, expert_names=experts)
    payload = {
        "release_name": resolve_release_name(release_name),
        "routerset_dir": str(routerset_dir),
        "manifest_path": str(active_manifest_path),
        "rebuilt_manifest_path": str(rebuilt_manifest_path) if rebuilt_manifest_path is not None else None,
        "experts": experts,
        "training": training,
        "n_shots": int(n_shots),
        "target_size": int(target_size),
        "target_channels": int(target_channels),
        "balanced_sampling": bool(balanced_sampling),
        "dataset_report": dataset_report,
        "checkpoint_report": checkpoint_report,
    }
    if output_dir is not None:
        save_json(Path(output_dir) / "preflight_report.json", payload)
    return payload


def prepare_routerset_training(
    *,
    routerset_dir: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    expert_names: Optional[Sequence[str]] = None,
    weights_dir: Optional[Union[str, Path]] = None,
    runtime_root: Optional[Union[str, Path]] = None,
    training: str = "finetuning",
    n_shots: int = 5000,
    target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
    target_channels: int = 8,
    release_name: Optional[str] = None,
    rebuild_splits: bool = False,
    rebuilt_manifest_out: Optional[Union[str, Path]] = None,
    balanced_sampling: bool = True,
    run_startup_gate: bool = True,
    startup_timeout_seconds: int = DEFAULT_STARTUP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    runtime_report = configure_local_runtime_environment(
        output_dir=output_dir,
        runtime_root=runtime_root,
        weights_dir=weights_dir,
    )
    runtime_environment_path = save_json(Path(output_dir) / "runtime_environment.json", runtime_report)

    preflight = preflight_routerset_training(
        routerset_dir=routerset_dir,
        manifest_path=manifest_path,
        expert_names=expert_names,
        weights_dir=runtime_report["weights_dir"],
        training=training,
        n_shots=n_shots,
        target_size=target_size,
        target_channels=target_channels,
        output_dir=output_dir,
        release_name=release_name,
        rebuild_splits=rebuild_splits,
        rebuilt_manifest_out=rebuilt_manifest_out,
        balanced_sampling=balanced_sampling,
    )
    startup_gate = run_training_startup_gate(
        routerset_dir=routerset_dir,
        manifest_path=preflight["manifest_path"],
        expert_names=list(expert_names or DEFAULT_ROUTERSET_EXPERTS),
        target_size=target_size,
        target_channels=target_channels,
        balanced_sampling=balanced_sampling,
        output_dir=output_dir,
        startup_timeout_seconds=startup_timeout_seconds,
        run_lightning_probe=run_startup_gate,
    )
    payload = {
        "status": "prepared",
        "routerset_dir": str(routerset_dir),
        "manifest_path": preflight["manifest_path"],
        "runtime_environment_path": str(runtime_environment_path),
        "preflight_report_path": str(Path(output_dir) / "preflight_report.json"),
        "startup_gate_path": str(Path(output_dir) / "startup_gate.json"),
        "weights_dir": runtime_report["weights_dir"],
        "runtime_root": runtime_report["runtime_root"],
        "release_name": preflight["release_name"],
        "checkpoint_report": preflight["checkpoint_report"],
        "dataset_report": preflight["dataset_report"],
        "startup_gate": startup_gate,
    }
    save_json(Path(output_dir) / "prepare_report.json", payload)
    return payload


def write_routing_predictions(
    model: MoEStudent,
    dataloader: DataLoader,
    output_path: Union[str, Path],
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


def _copy_if_exists(source: Union[str, Path], destination: Union[str, Path]) -> Optional[Path]:
    source_path = Path(source)
    if not source_path.exists():
        return None
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


def write_release_manifest(
    release_dir: Union[str, Path],
    *,
    model: MoEStudent,
    config: Mapping[str, Any],
    dataset_report: Mapping[str, Any],
    checkpoint_report: Mapping[str, Any],
    run_summary: Mapping[str, Any],
) -> Path:
    manifest = {
        "artifact_name": config["release_name"],
        "bundle_type": "phidranet_student_moe",
        "created_at_utc": utc_timestamp(),
        "experts": list(model.expert_names),
        "encoder_source_task": model.encoder_source_task,
        "threshold": model.threshold,
        "top_k": model.top_k,
        "target_channels": config["target_channels"],
        "target_size": config["target_size"],
        "checkpoint_report": checkpoint_report,
        "dataset_report": dataset_report,
        "run_summary": dict(run_summary),
    }
    return save_json(Path(release_dir) / "release_manifest.json", manifest)


def write_deployment_readme(release_dir: Union[str, Path], *, release_name: str) -> Path:
    text = f"""# {release_name}

This directory contains the final `phidranet` student MoE release package.

## Contents

- `student_moe_bundle.pt`: final MoE bundle loadable with `hydranet.load_student_moe_bundle(...)`
- `config.json`: canonical training configuration
- `metrics.json`: final logged metrics from training
- `routing_predictions.jsonl`: validation routing report
- `dataset_report.json`: train/validation dataset summary after routerset adaptation
- `release_manifest.json`: self-contained release metadata

## Load

```python
from hydranet import load_student_moe_bundle

model = load_student_moe_bundle("{release_name}/student_moe_bundle.pt")
```

## CLI Inference

```bash
PYTHONPATH=src python3 scripts/infer_moe_switcher.py \\
  --bundle-path {release_name}/student_moe_bundle.pt \\
  --routerset-dir routerset \\
  --output-dir outputs/moe/infer_{release_name}
```
"""
    return save_text(Path(release_dir) / "DEPLOY.md", text)


def create_phidranet_release(
    *,
    release_root: Union[str, Path],
    release_name: str,
    model: MoEStudent,
    config: Mapping[str, Any],
    dataset_report: Mapping[str, Any],
    checkpoint_report: Mapping[str, Any],
    run_output_dir: Union[str, Path],
    bundle_path: Union[str, Path],
    prediction_path: Union[str, Path],
) -> dict[str, str]:
    release_dir = ensure_dir(Path(release_root) / release_name)
    final_bundle_path = Path(release_dir) / "student_moe_bundle.pt"
    if Path(bundle_path) != final_bundle_path:
        _copy_if_exists(bundle_path, final_bundle_path)

    config_path = _copy_if_exists(Path(run_output_dir) / "config.json", Path(release_dir) / "config.json")
    metrics_path = _copy_if_exists(Path(run_output_dir) / "metrics.json", Path(release_dir) / "metrics.json")
    baseline_path = _copy_if_exists(Path(run_output_dir) / "baseline_summary.json", Path(release_dir) / "baseline_summary.json")
    routing_path = _copy_if_exists(prediction_path, Path(release_dir) / "routing_predictions.jsonl")
    dataset_report_path = save_json(Path(release_dir) / "dataset_report.json", dataset_report)
    checkpoint_report_path = save_json(Path(release_dir) / "checkpoint_report.json", checkpoint_report)
    manifest_path = write_release_manifest(
        release_dir,
        model=model,
        config=config,
        dataset_report=dataset_report,
        checkpoint_report=checkpoint_report,
        run_summary={
            "bundle_path": str(final_bundle_path),
            "routing_predictions": str(routing_path) if routing_path else None,
            "metrics_path": str(metrics_path) if metrics_path else None,
        },
    )
    deployment_path = write_deployment_readme(release_dir, release_name=release_name)

    return {
        "release_dir": str(release_dir),
        "bundle_path": str(final_bundle_path),
        "config_path": str(config_path) if config_path else "",
        "metrics_path": str(metrics_path) if metrics_path else "",
        "baseline_summary_path": str(baseline_path) if baseline_path else "",
        "routing_predictions_path": str(routing_path) if routing_path else "",
        "dataset_report_path": str(dataset_report_path),
        "checkpoint_report_path": str(checkpoint_report_path),
        "release_manifest_path": str(manifest_path),
        "deployment_path": str(deployment_path),
    }


def write_run_summary(output_dir: Union[str, Path], payload: Mapping[str, Any]) -> Path:
    return save_json(Path(output_dir) / "summary.json", payload)


def run_exported_moe_inference(
    *,
    routerset_dir: Union[str, Path],
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    bundle_path: Union[str, Path],
    expert_names: Optional[Sequence[str]] = None,
    batch_size: int = 4,
    num_workers: int = 0,
    target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
    target_channels: int = 8,
    balanced_sampling: bool = True,
) -> dict[str, str]:
    inference_dir = ensure_dir(output_dir)
    model = load_student_moe_bundle(bundle_path)
    datamodule = RoutersetMoEDataModule(
        routerset_dir,
        manifest_path=manifest_path,
        expert_names=list(expert_names or DEFAULT_ROUTERSET_EXPERTS),
        batch_size=batch_size,
        num_workers=num_workers,
        target_size=target_size,
        target_channels=target_channels,
        balanced_sampling=balanced_sampling,
    )
    datamodule.setup("validate")
    prediction_path = write_routing_predictions(
        model,
        datamodule.val_dataloader(),
        inference_dir / "routing_predictions.jsonl",
    )
    summary_path = save_json(
        inference_dir / "summary.json",
        {
            "status": "completed",
            "bundle_path": str(bundle_path),
            "prediction_path": str(prediction_path),
            "manifest_path": str(manifest_path),
        },
    )
    return {
        "prediction_path": str(prediction_path),
        "summary_path": str(summary_path),
    }


def train_switcher(
    *,
    routerset_dir: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    expert_names: Optional[Sequence[str]] = None,
    weights_dir: Optional[str] = None,
    training: str = "finetuning",
    n_shots: int = 5000,
    batch_size: int = 4,
    num_workers: int = 0,
    max_epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
    target_channels: int = 8,
    seed: int = 42,
    auto_load_weights: bool = True,
    release_name: Optional[str] = None,
    release_root: Union[str, Path] = DEFAULT_RELEASE_ROOT,
    rebuild_splits: bool = False,
    rebuilt_manifest_out: Optional[Union[str, Path]] = None,
    balanced_sampling: bool = True,
    runtime_root: Optional[Union[str, Path]] = None,
    accelerator: str = "cpu",
    devices: Union[str, int, Sequence[int]] = 1,
    precision: Optional[str] = None,
    run_startup_gate: bool = True,
    startup_timeout_seconds: int = DEFAULT_STARTUP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    recorder = StartupRecorder(output_dir)
    recorder.mark("output_dir_ready", output_dir=str(output_dir))
    runtime_report = configure_local_runtime_environment(
        output_dir=output_dir,
        runtime_root=runtime_root,
        weights_dir=weights_dir,
    )
    runtime_environment_path = save_json(Path(output_dir) / "runtime_environment.json", runtime_report)
    recorder.mark(
        "runtime_environment_configured",
        runtime_root=runtime_report["runtime_root"],
        weights_dir=runtime_report["weights_dir"],
    )
    expert_names = list(expert_names or DEFAULT_ROUTERSET_EXPERTS)
    release_name_value = resolve_release_name(release_name)
    active_manifest_path = resolve_routerset_manifest_path(routerset_dir, manifest_path)
    manifest_path_value = str(active_manifest_path)
    dataset_root_value = str(resolve_routerset_dataset_root(routerset_dir))
    normalized_devices = normalize_trainer_devices(devices)
    config_path = Path(output_dir) / "config.json"
    dataset_report_path = Path(output_dir) / "dataset_report.json"
    checkpoint_report_path = Path(output_dir) / "checkpoint_report.json"
    preflight_report_path = Path(output_dir) / "preflight_report.json"
    metrics_path = Path(output_dir) / "metrics.json"
    summary_path = Path(output_dir) / "summary.json"
    startup_gate_path = Path(output_dir) / "startup_gate.json"

    try:
        recorder.mark(
            "manifest_resolved",
            routerset_dir=str(routerset_dir),
            dataset_root=dataset_root_value,
            manifest_path=manifest_path_value,
        )
        if rebuild_splits:
            active_manifest_path = rebuild_routerset_split_manifest(
                routerset_dir,
                expert_names=expert_names,
                manifest_path=active_manifest_path,
                output_path=rebuilt_manifest_out,
            )
            manifest_path_value = str(active_manifest_path)
            recorder.mark("split_rebuilt", manifest_path=manifest_path_value)

        config = create_moe_run_config(
            routerset_dir=routerset_dir,
            output_dir=output_dir,
            manifest_path=active_manifest_path,
            expert_names=expert_names,
            training=training,
            n_shots=n_shots,
            target_size=target_size,
            target_channels=target_channels,
            threshold=threshold,
            top_k=top_k,
            release_name=release_name_value.replace("phidranet_", "", 1),
            balanced_sampling=balanced_sampling,
            runtime_root=runtime_report["runtime_root"],
            weights_dir=runtime_report["weights_dir"],
            accelerator=accelerator,
            devices=normalized_devices,
            precision=precision,
        )
        save_json(config_path, config)
        recorder.mark("config_written", config_path=str(config_path))

        dataset_report = collect_routerset_dataset_report(
            routerset_dir,
            manifest_path=active_manifest_path,
            expert_names=expert_names,
            target_size=target_size,
            target_channels=target_channels,
            balanced_sampling=balanced_sampling,
        )
        validate_routerset_dataset_report(
            dataset_report,
            expert_names=expert_names,
            target_size=target_size,
            target_channels=target_channels,
        )
        save_json(dataset_report_path, dataset_report)
        recorder.mark("dataset_report_written", dataset_report_path=str(dataset_report_path))

        if run_startup_gate:
            recorder.mark("startup_gate_started", timeout_seconds=int(startup_timeout_seconds))
            try:
                run_training_startup_gate(
                    routerset_dir=routerset_dir,
                    manifest_path=active_manifest_path,
                    expert_names=expert_names,
                    target_size=target_size,
                    target_channels=target_channels,
                    balanced_sampling=balanced_sampling,
                    output_dir=output_dir,
                    startup_timeout_seconds=startup_timeout_seconds,
                    run_lightning_probe=True,
                )
            except BaseException:
                recorder.mark(
                    "startup_gate_failed",
                    startup_gate_path=str(startup_gate_path) if startup_gate_path.exists() else "",
                )
                recorder.current_stage = "startup_failed"
                raise
            recorder.mark("startup_gate_completed")
        else:
            save_json(
                startup_gate_path,
                {
                    "status": "skipped",
                    "manifest_path": manifest_path_value,
                    "timeout_seconds": int(startup_timeout_seconds),
                    "lightning_probe": {
                        "status": "skipped",
                        "timeout_seconds": int(startup_timeout_seconds),
                        "reason": "startup gate skipped by caller",
                    },
                },
            )
            recorder.mark("startup_gate_skipped")

        checkpoint_report = resolve_student_checkpoint_report(
            expert_names,
            training=training,
            n_shots=n_shots,
            weights_dir=runtime_report["weights_dir"],
        )
        save_json(checkpoint_report_path, checkpoint_report)
        recorder.mark("checkpoint_report_written", checkpoint_report_path=str(checkpoint_report_path))
        validate_student_checkpoint_report(checkpoint_report, expert_names=expert_names)
        save_json(
            preflight_report_path,
            {
                "release_name": release_name_value,
                "routerset_dir": str(routerset_dir),
                "manifest_path": manifest_path_value,
                "rebuilt_manifest_path": str(active_manifest_path) if rebuild_splits else None,
                "experts": expert_names,
                "training": training,
                "n_shots": int(n_shots),
                "target_size": int(target_size),
                "target_channels": int(target_channels),
                "balanced_sampling": bool(balanced_sampling),
                "dataset_report": dataset_report,
                "checkpoint_report": checkpoint_report,
            },
        )
        recorder.mark("preflight_report_written", preflight_report_path=str(preflight_report_path))

        model = build_routerset_moe(
            routerset_dir=routerset_dir,
            expert_names=expert_names,
            training=training,
            n_shots=n_shots,
            weights_dir=runtime_report["weights_dir"],
            threshold=threshold,
            top_k=top_k,
            auto_load_weights=auto_load_weights,
        )
        capture_baseline_summary(model, output_dir)
        recorder.mark("model_assembled", encoder_source_task=model.encoder_source_task)

        if run_startup_gate:
            recorder.mark("lightning_probe_completed", source="startup_gate")
        else:
            recorder.mark("lightning_probe_started")
            probe_lightning_import(timeout_seconds=startup_timeout_seconds)
            recorder.mark("lightning_probe_completed")

        recorder.mark("lightning_import_started")
        lightning_module_cls, build_trainer_fn, seed_everything_fn = load_lightning_training_components()
        recorder.mark("lightning_import_completed")

        seed_everything_fn(seed, workers=True)
        datamodule = RoutersetMoEDataModule(
            routerset_dir,
            manifest_path=active_manifest_path,
            expert_names=expert_names,
            batch_size=batch_size,
            num_workers=num_workers,
            target_size=target_size,
            target_channels=target_channels,
            balanced_sampling=balanced_sampling,
        )
        lightning_module = lightning_module_cls(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        recorder.mark("trainer_creation_started")
        trainer = build_trainer_fn(
            output_dir,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=normalized_devices,
            precision=precision,
        )
        recorder.mark("trainer_created")

        datamodule.setup("fit")
        recorder.mark("fit_started")
        trainer.fit(
            lightning_module,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
        )
        recorder.mark("fit_completed")
        metrics = {key: float(value) for key, value in trainer.callback_metrics.items()}
        save_json(metrics_path, metrics)

        bundle_path = save_student_moe_bundle(model, Path(output_dir) / "student_moe_bundle.pt", metadata=config)
        prediction_path = write_routing_predictions(
            model,
            datamodule.val_dataloader(),
            Path(output_dir) / "routing_predictions.jsonl",
        )

        release_summary = create_phidranet_release(
            release_root=release_root,
            release_name=release_name_value,
            model=model,
            config=config,
            dataset_report=dataset_report,
            checkpoint_report=checkpoint_report,
            run_output_dir=output_dir,
            bundle_path=bundle_path,
            prediction_path=prediction_path,
        )
        contract = build_run_contract(
            output_dir=output_dir,
            completed_stages=("preflight", "startup_gate", "training", "export"),
            release_dir=release_summary["release_dir"],
        )
        validate_run_contract_artifacts(contract, required_sections=("run_artifacts", "release_artifacts"))

        summary = {
            "status": "completed",
            "startup_stage": recorder.current_stage,
            "failure_stage": recorder.failure_stage,
            "routerset_dir": str(routerset_dir),
            "dataset_root": dataset_root_value,
            "manifest_path": manifest_path_value,
            "runtime_root": runtime_report["runtime_root"],
            "runtime_environment_path": str(runtime_environment_path),
            "preflight_report_path": str(preflight_report_path),
            "bundle_path": str(bundle_path),
            "metrics_path": str(metrics_path),
            "prediction_path": str(prediction_path),
            "config_path": str(config_path),
            "dataset_report_path": str(dataset_report_path),
            "checkpoint_report_path": str(checkpoint_report_path),
            "startup_log_path": str(recorder.log_path),
            "startup_stage_path": str(recorder.stage_path),
            "startup_gate_path": str(startup_gate_path) if startup_gate_path.exists() else "",
            "release_dir": release_summary["release_dir"],
            "release_manifest_path": release_summary["release_manifest_path"],
            "contract": contract,
        }
        write_run_summary(output_dir, summary)
        return summary
    except BaseException as error:
        recorder.fail(recorder.current_stage or "startup_failed", error)
        failure_summary = {
            "status": "failed",
            "startup_stage": recorder.current_stage,
            "failure_stage": recorder.failure_stage or recorder.current_stage,
            "routerset_dir": str(routerset_dir),
            "dataset_root": dataset_root_value,
            "manifest_path": manifest_path_value,
            "runtime_root": runtime_report["runtime_root"],
            "runtime_environment_path": str(runtime_environment_path),
            "preflight_report_path": str(preflight_report_path) if preflight_report_path.exists() else "",
            "config_path": str(config_path) if config_path.exists() else "",
            "dataset_report_path": str(dataset_report_path) if dataset_report_path.exists() else "",
            "checkpoint_report_path": str(checkpoint_report_path) if checkpoint_report_path.exists() else "",
            "metrics_path": str(metrics_path) if metrics_path.exists() else "",
            "startup_log_path": str(recorder.log_path),
            "startup_stage_path": str(recorder.stage_path),
            "startup_gate_path": str(startup_gate_path) if startup_gate_path.exists() else "",
            "summary_path": str(summary_path),
            "error_type": type(error).__name__,
            "error": str(error),
            "contract": build_run_contract(
                output_dir=output_dir,
                completed_stages=completed_stages_for_failed_run(output_dir),
            ),
        }
        write_run_summary(output_dir, failure_summary)
        raise


def run_full_training(
    *,
    routerset_dir: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    expert_names: Optional[Sequence[str]] = None,
    weights_dir: Optional[str] = None,
    training: str = "finetuning",
    n_shots: int = 5000,
    batch_size: int = 4,
    num_workers: int = 0,
    max_epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
    target_channels: int = 8,
    seed: int = 42,
    release_name: Optional[str] = None,
    release_root: Union[str, Path] = DEFAULT_RELEASE_ROOT,
    rebuild_splits: bool = False,
    rebuilt_manifest_out: Optional[Union[str, Path]] = None,
    balanced_sampling: bool = True,
    runtime_root: Optional[Union[str, Path]] = None,
    accelerator: str = "cpu",
    devices: Union[str, int, Sequence[int]] = 1,
    precision: Optional[str] = None,
    run_startup_gate: bool = True,
    startup_timeout_seconds: int = DEFAULT_STARTUP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    inference_dir = output_dir / "inference"
    smoke_summary_path = output_dir / "smoke_test_summary.json"
    training_summary: Optional[dict[str, Any]] = None
    active_manifest_path = manifest_path
    expert_list = list(expert_names or DEFAULT_ROUTERSET_EXPERTS)

    try:
        training_summary = train_switcher(
            routerset_dir=routerset_dir,
            output_dir=output_dir,
            manifest_path=manifest_path,
            expert_names=expert_list,
            weights_dir=weights_dir,
            training=training,
            n_shots=n_shots,
            batch_size=batch_size,
            num_workers=num_workers,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            threshold=threshold,
            top_k=top_k,
            target_size=target_size,
            target_channels=target_channels,
            seed=seed,
            release_name=release_name,
            release_root=release_root,
            rebuild_splits=rebuild_splits,
            rebuilt_manifest_out=rebuilt_manifest_out,
            balanced_sampling=balanced_sampling,
            runtime_root=runtime_root,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            run_startup_gate=run_startup_gate,
            startup_timeout_seconds=startup_timeout_seconds,
        )
        active_manifest_path = training_summary["manifest_path"]

        inference_report = run_exported_moe_inference(
            routerset_dir=routerset_dir,
            manifest_path=active_manifest_path,
            output_dir=inference_dir,
            bundle_path=training_summary["bundle_path"],
            expert_names=expert_list,
            batch_size=batch_size,
            num_workers=num_workers,
            target_size=target_size,
            target_channels=target_channels,
            balanced_sampling=balanced_sampling,
        )

        smoke_summary = {
            "status": "completed",
            "failure_stage": None,
            "preflight_report_path": training_summary["preflight_report_path"],
            "runtime_environment_path": training_summary["runtime_environment_path"],
            "manifest_path": active_manifest_path,
            "training_summary_path": str(output_dir / "summary.json"),
            "bundle_path": training_summary["bundle_path"],
            "inference_summary_path": inference_report["summary_path"],
            "inference_prediction_path": inference_report["prediction_path"],
            "release_dir": training_summary["release_dir"],
        }
        save_json(smoke_summary_path, smoke_summary)

        full_summary = dict(training_summary)
        full_summary.update(
            {
                "status": "completed",
                "failure_stage": None,
                "smoke_test_summary_path": str(smoke_summary_path),
                "inference_summary_path": inference_report["summary_path"],
                "inference_prediction_path": inference_report["prediction_path"],
                "contract": build_run_contract(
                    output_dir=output_dir,
                    completed_stages=FULL_TRAINING_STAGES,
                    release_dir=training_summary["release_dir"],
                ),
            }
        )
        write_run_summary(output_dir, full_summary)
        return full_summary
    except BaseException as error:
        existing_summary: dict[str, Any] = {}
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))

        failure_stage = (
            "smoke"
            if training_summary is not None
            else existing_summary.get("failure_stage") or existing_summary.get("startup_stage") or "startup_failed"
        )
        failure_payload = {
            "status": "failed",
            "failure_stage": failure_stage,
            "training_summary_path": str(summary_path) if summary_path.exists() else "",
            "startup_stage_path": existing_summary.get("startup_stage_path", ""),
            "startup_log_path": existing_summary.get("startup_log_path", ""),
            "manifest_path": active_manifest_path or existing_summary.get("manifest_path", ""),
            "bundle_path": training_summary["bundle_path"] if training_summary else existing_summary.get("bundle_path", ""),
            "runtime_environment_path": existing_summary.get("runtime_environment_path", ""),
            "preflight_report_path": existing_summary.get("preflight_report_path", ""),
            "inference_summary_path": str(inference_dir / "summary.json") if (inference_dir / "summary.json").exists() else "",
            "inference_prediction_path": str(inference_dir / "routing_predictions.jsonl")
            if (inference_dir / "routing_predictions.jsonl").exists()
            else "",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        save_json(smoke_summary_path, failure_payload)

        if training_summary is not None:
            failed_summary = dict(training_summary)
            failed_summary.update(
                {
                    "status": "failed",
                    "failure_stage": "smoke",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "smoke_test_summary_path": str(smoke_summary_path),
                    "inference_summary_path": failure_payload["inference_summary_path"],
                    "inference_prediction_path": failure_payload["inference_prediction_path"],
                    "contract": build_run_contract(
                        output_dir=output_dir,
                        completed_stages=("preflight", "startup_gate", "training", "export"),
                        release_dir=training_summary["release_dir"],
                    ),
                }
            )
            write_run_summary(output_dir, failed_summary)
        raise


def run_moe_smoke_test(
    *,
    routerset_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path],
    weights_dir: Optional[str] = None,
    training: str = "finetuning",
    n_shots: int = 5000,
    batch_size: int = 4,
    num_workers: int = 0,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
    target_channels: int = 8,
    seed: int = 42,
    release_name: str = "smoke_v1",
    release_root: Union[str, Path] = DEFAULT_RELEASE_ROOT,
    rebuilt_manifest_out: Optional[Union[str, Path]] = None,
    balanced_sampling: bool = True,
    runtime_root: Optional[Union[str, Path]] = None,
    accelerator: str = "cpu",
    devices: Union[str, int, Sequence[int]] = 1,
    precision: Optional[str] = None,
    startup_timeout_seconds: int = DEFAULT_STARTUP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    runtime_report = configure_local_runtime_environment(
        output_dir=output_dir,
        runtime_root=runtime_root,
        weights_dir=weights_dir,
    )
    save_json(Path(output_dir) / "runtime_environment.json", runtime_report)
    rebuilt_manifest_path = Path(rebuilt_manifest_out) if rebuilt_manifest_out else default_rebuilt_manifest_path(routerset_dir)
    inference_dir = Path(output_dir) / "inference"
    smoke_summary_path = Path(output_dir) / "smoke_test_summary.json"

    preflight: Optional[dict[str, Any]] = None
    training_summary: Optional[dict[str, Any]] = None
    try:
        preflight = preflight_routerset_training(
            routerset_dir=routerset_dir,
            manifest_path=manifest_path,
            expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
            weights_dir=runtime_report["weights_dir"],
            training=training,
            n_shots=n_shots,
            target_size=target_size,
            target_channels=target_channels,
            output_dir=output_dir,
            release_name=release_name,
            rebuild_splits=True,
            rebuilt_manifest_out=rebuilt_manifest_path,
            balanced_sampling=balanced_sampling,
        )
        training_summary = train_switcher(
            routerset_dir=routerset_dir,
            output_dir=output_dir,
            manifest_path=preflight["manifest_path"],
            expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
            weights_dir=runtime_report["weights_dir"],
            training=training,
            n_shots=n_shots,
            batch_size=batch_size,
            num_workers=num_workers,
            max_epochs=1,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            threshold=threshold,
            top_k=top_k,
            target_size=target_size,
            target_channels=target_channels,
            seed=seed,
            release_name=release_name,
            release_root=release_root,
            balanced_sampling=balanced_sampling,
            runtime_root=runtime_report["runtime_root"],
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            run_startup_gate=True,
            startup_timeout_seconds=startup_timeout_seconds,
        )

        inference_report = run_exported_moe_inference(
            routerset_dir=routerset_dir,
            manifest_path=preflight["manifest_path"],
            output_dir=inference_dir,
            bundle_path=training_summary["bundle_path"],
            expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
            batch_size=batch_size,
            num_workers=num_workers,
            target_size=target_size,
            target_channels=target_channels,
            balanced_sampling=balanced_sampling,
        )

        smoke_summary = {
            "status": "completed",
            "preflight_report_path": str(Path(output_dir) / "preflight_report.json"),
            "runtime_environment_path": str(Path(output_dir) / "runtime_environment.json"),
            "rebuilt_manifest_path": preflight["manifest_path"],
            "training_summary_path": str(Path(output_dir) / "summary.json"),
            "bundle_path": training_summary["bundle_path"],
            "inference_summary_path": inference_report["summary_path"],
            "inference_prediction_path": inference_report["prediction_path"],
            "release_dir": training_summary["release_dir"],
        }
        save_json(smoke_summary_path, smoke_summary)
        return smoke_summary
    except BaseException as error:
        failure_summary = {
            "status": "failed",
            "preflight_report_path": str(Path(output_dir) / "preflight_report.json"),
            "runtime_environment_path": str(Path(output_dir) / "runtime_environment.json") if (Path(output_dir) / "runtime_environment.json").exists() else "",
            "training_summary_path": str(Path(output_dir) / "summary.json") if (Path(output_dir) / "summary.json").exists() else "",
            "startup_stage_path": str(Path(output_dir) / "startup_stage.json") if (Path(output_dir) / "startup_stage.json").exists() else "",
            "startup_log_path": str(Path(output_dir) / "startup_log.txt") if (Path(output_dir) / "startup_log.txt").exists() else "",
            "rebuilt_manifest_path": preflight["manifest_path"] if preflight else "",
            "bundle_path": training_summary["bundle_path"] if training_summary else "",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        save_json(smoke_summary_path, failure_summary)
        raise
