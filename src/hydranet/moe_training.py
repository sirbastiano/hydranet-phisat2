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
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if TYPE_CHECKING:
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
DEFAULT_ROUTERSET_DATASET_SUBDIR = "multilabel_dataset"
DEFAULT_ROUTERSET_MANIFEST = "multilabel_dataset/manifest.jsonl"
ROUTERSET_SWAPPED_TILE_DATASETS = frozenset({"burned_area", "worldfloods"})
ROUTERSET_PHI2FM_RAW_S2_DATASETS = frozenset({"lc", "roads"})
ROUTERSET_PHI2FM_FLOAT_MINMAX_DATASETS = frozenset(
    {"anomaly_detection", "burned_area", "fire", "worldfloods"}
)
ROUTERSET_PHI2FM_STUDENT_SCALE = 10000.0
DEFAULT_ROUTERSET_REPORT_RAW_SAMPLES_PER_EXPERT = 2
DEFAULT_ROUTERSET_FAULT_EXAMPLES_PER_CODE = 5
DEFAULT_STARTUP_TIMEOUT_SECONDS = 120
DEFAULT_RUNTIME_SUBDIR = "runtime"
DEFAULT_BUNDLE_SUBDIR = "bundle"
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
    "routing_target_report": "routing_target_report.json",
    "baseline_summary": "baseline_summary.json",
    "metrics": "metrics.json",
    "bundle": "student_moe_bundle.pt",
    "routing_predictions": "routing_predictions.jsonl",
    "summary": "summary.json",
}
ROUTING_TARGET_SOURCE = "expert_inference_v1"
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
ARTIFACT_LAYOUT_DIRS = {
    "runtime_root": DEFAULT_RUNTIME_SUBDIR,
    "checkpoints": "weights",
    "configs": ".",
    "reports": ".",
    "bundle_root": DEFAULT_BUNDLE_SUBDIR,
    "inference": "inference",
}


def _loading_module():
    return import_module("hydranet.loading")


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


def _routing_target_row_token(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("source_dataset", "")),
        str(row.get("source_sample_id", "")),
        str(row.get("moe_split", row.get("source_split", ""))),
        int(row.get("patch_x") or 0),
        int(row.get("patch_y") or 0),
        str(row.get("patch_width")),
        str(row.get("patch_height")),
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def ensure_within_root(path: Union[str, Path], *, root: Union[str, Path], label: str) -> Path:
    candidate = Path(path)
    root_path = Path(root)
    if not _is_relative_to(candidate, root_path):
        raise ValueError(f"{label} must stay under {root_path}, got {candidate}.")
    return candidate


def resolve_release_root(
    *,
    output_dir: Union[str, Path],
    release_root: Optional[Union[str, Path]] = None,
) -> Path:
    output_dir = Path(output_dir)
    if release_root is None:
        candidate = output_dir / DEFAULT_BUNDLE_SUBDIR
    else:
        release_root_path = Path(release_root)
        if release_root_path.is_absolute() or _is_relative_to(release_root_path, output_dir):
            candidate = release_root_path
        else:
            candidate = output_dir / release_root_path
    ensure_within_root(candidate, root=output_dir, label="release_root")
    return ensure_dir(candidate)


def build_artifact_layout(
    *,
    output_dir: Union[str, Path],
    runtime_root: Union[str, Path],
    release_name: Optional[str] = None,
    release_root: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    runtime_root = Path(runtime_root)
    ensure_within_root(runtime_root, root=runtime_root, label="runtime_root")
    checkpoints_dir = ensure_within_root(runtime_root / ARTIFACT_LAYOUT_DIRS["checkpoints"], root=runtime_root, label="checkpoints_dir")
    bundle_root = resolve_release_root(output_dir=output_dir, release_root=release_root)
    layout: dict[str, Any] = {
        "output_root": str(output_dir),
        "runtime_root": str(runtime_root),
        "directories": {
            "configs_dir": str(output_dir),
            "reports_dir": str(output_dir),
            "runtime_root": str(runtime_root),
            "checkpoints_dir": str(checkpoints_dir),
            "bundle_root": str(bundle_root),
            "inference_dir": str(output_dir / ARTIFACT_LAYOUT_DIRS["inference"]),
        },
        "expected_subdirs": dict(ARTIFACT_LAYOUT_DIRS),
    }
    if release_name is not None:
        release_dir = ensure_within_root(bundle_root / release_name, root=output_dir, label="release_dir")
        layout["directories"]["release_dir"] = str(release_dir)
        layout["release_artifacts"] = {
            name: str(release_dir / filename) for name, filename in RELEASE_ARTIFACT_FILENAMES.items()
        }
    layout["run_artifacts"] = {
        name: str(output_dir / filename) for name, filename in RUN_ARTIFACT_FILENAMES.items()
    }
    return layout


def build_run_contract(
    *,
    output_dir: Union[str, Path],
    completed_stages: Sequence[str],
    runtime_root: Optional[Union[str, Path]] = None,
    release_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    unique_completed_stages = [stage for stage in FULL_TRAINING_STAGES if stage in set(completed_stages)]
    next_stage = next((stage for stage in FULL_TRAINING_STAGES if stage not in unique_completed_stages), None)
    resolved_runtime_root = Path(runtime_root) if runtime_root is not None else resolve_runtime_root(output_dir=output_dir)
    artifact_layout = build_artifact_layout(
        output_dir=output_dir,
        runtime_root=resolved_runtime_root,
        release_name=Path(release_dir).name if release_dir is not None else None,
        release_root=Path(release_dir).parent if release_dir is not None else None,
    )
    contract: dict[str, Any] = {
        "version": FULL_TRAINING_CONTRACT_VERSION,
        "canonical_stages": list(FULL_TRAINING_STAGES),
        "completed_stages": unique_completed_stages,
        "next_stage": next_stage,
        "run_dir": str(output_dir),
        "layout": artifact_layout,
        "run_artifacts": dict(artifact_layout["run_artifacts"]),
        "expected_artifact_names": {
            "run_dir": dict(RUN_ARTIFACT_FILENAMES),
            "release_dir": dict(RELEASE_ARTIFACT_FILENAMES),
        },
    }
    if release_dir is not None:
        release_dir = Path(release_dir)
        contract["release_dir"] = str(release_dir)
        contract["release_artifacts"] = dict(artifact_layout["release_artifacts"])
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
        resolved_runtime_root = ensure_dir(runtime_root)
    else:
        resolved_runtime_root = ensure_dir(Path(output_dir) / DEFAULT_RUNTIME_SUBDIR)
    ensure_within_root(resolved_runtime_root, root=resolved_runtime_root, label="runtime_root")
    return resolved_runtime_root


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
    resolved_weights_dir = ensure_dir(weights_dir or (resolved_runtime_root / ARTIFACT_LAYOUT_DIRS["checkpoints"]))
    ensure_within_root(resolved_weights_dir, root=resolved_runtime_root, label="weights_dir")

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

    def mark(self, stage: str, *, update_current_stage: bool = True, **payload: Any) -> None:
        if update_current_stage:
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


def extract_non_finite_loss_details(error: BaseException) -> Optional[dict[str, Any]]:
    diagnostics = getattr(error, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        return None

    batch_index = diagnostics.get("batch_index")
    try:
        normalized_batch_index = int(batch_index) if batch_index is not None else -1
    except (TypeError, ValueError):
        normalized_batch_index = -1

    sample_ids = diagnostics.get("sample_ids")
    if isinstance(sample_ids, Sequence) and not isinstance(sample_ids, (str, bytes)):
        normalized_sample_ids = [str(value) for value in sample_ids]
    else:
        sample_id = diagnostics.get("sample_id")
        normalized_sample_ids = [str(sample_id)] if sample_id is not None else []

    expert_context = diagnostics.get("expert_context")
    if not isinstance(expert_context, Mapping):
        expert_context = {}

    return {
        "stage": str(diagnostics.get("stage") or "fit"),
        "batch_index": normalized_batch_index,
        "batch_size": int(diagnostics.get("batch_size") or len(normalized_sample_ids)),
        "loss_value": str(diagnostics.get("loss_value") or str(error)),
        "sample_id": normalized_sample_ids[0] if normalized_sample_ids else "",
        "sample_ids": normalized_sample_ids,
        "expert_context": dict(expert_context),
    }


def resolve_routerset_dataset_root(routerset_dir: Union[str, Path]) -> Path:
    return Path(routerset_dir) / DEFAULT_ROUTERSET_DATASET_SUBDIR


def default_rebuilt_manifest_path(routerset_dir: Union[str, Path]) -> Path:
    return resolve_routerset_dataset_root(routerset_dir) / "manifest_moe_train.jsonl"


def legacy_root_rebuilt_manifest_path(routerset_dir: Union[str, Path]) -> Path:
    return Path(routerset_dir) / "manifest_moe_train.jsonl"


def routerset_patch_token(row: Mapping[str, Any]) -> str:
    width = row["patch_width"] if row["patch_width"] is not None else "full"
    height = row["patch_height"] if row["patch_height"] is not None else "full"
    return f"{row['source_sample_id']}_{row['patch_x']}_{row['patch_y']}_{height}_{width}.npy"


def routerset_compatibility_patch_token(row: Mapping[str, Any]) -> Optional[str]:
    width = row["patch_width"]
    height = row["patch_height"]
    if width is None or height is None:
        return None
    if row["source_dataset"] not in ROUTERSET_SWAPPED_TILE_DATASETS:
        return None
    return f"{row['source_sample_id']}_{row['patch_y']}_{row['patch_x']}_{height}_{width}.npy"


def routerset_image_path(routerset_dir: Union[str, Path], row: Mapping[str, Any]) -> Path:
    materialized_image_path = row.get("materialized_image_path")
    if materialized_image_path:
        return Path(str(materialized_image_path))

    routerset_root = resolve_routerset_dataset_root(routerset_dir)
    filename = routerset_patch_token(row)
    image_dir = routerset_root / "images" / row["source_dataset"] / row["source_split"]
    primary = image_dir / filename
    if primary.exists():
        return primary

    compatibility_filename = routerset_compatibility_patch_token(row)
    if compatibility_filename is not None:
        compatibility_path = image_dir / compatibility_filename
        if compatibility_path.exists():
            return compatibility_path

    return primary


def routerset_uses_compatibility_path(row: Mapping[str, Any], image_path: Union[str, Path]) -> bool:
    if row.get("materialized_image_path"):
        return False
    primary_filename = routerset_patch_token(row)
    compatibility_filename = routerset_compatibility_patch_token(row)
    if compatibility_filename is None:
        return False
    if compatibility_filename == primary_filename:
        return False
    return Path(image_path).name == compatibility_filename


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
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in final_rows) + "\n"
    destination.write_text(payload, encoding="utf-8")

    legacy_destination = legacy_root_rebuilt_manifest_path(routerset_dir)
    if legacy_destination != destination and legacy_destination.exists():
        legacy_destination.write_text(payload, encoding="utf-8")

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


def sanitize_routerset_array(array: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    current = np.asarray(array)
    finite_mask = np.isfinite(current)
    non_finite_count = int((~finite_mask).sum())
    diagnostics = {
        "had_non_finite": non_finite_count > 0,
        "non_finite_count": non_finite_count,
        "nan_count": int(np.isnan(current).sum()),
        "pos_inf_count": int(np.isposinf(current).sum()),
        "neg_inf_count": int(np.isneginf(current).sum()),
    }
    if non_finite_count <= 0:
        return current, diagnostics

    finite_values = current[finite_mask]
    finite_min = float(finite_values.min()) if finite_values.size > 0 else 0.0
    finite_max = float(finite_values.max()) if finite_values.size > 0 else 0.0
    diagnostics["finite_min"] = finite_min
    diagnostics["finite_max"] = finite_max
    sanitized = np.nan_to_num(
        current,
        nan=0.0,
        posinf=finite_max,
        neginf=finite_min,
        copy=True,
    )
    return sanitized, diagnostics


def _normalize_routerset_channel_count(current: np.ndarray, *, target_channels: int) -> np.ndarray:
    if current.shape[0] > target_channels:
        return current[:target_channels]
    if current.shape[0] < target_channels:
        padding = np.zeros(
            (target_channels - current.shape[0], current.shape[1], current.shape[2]),
            dtype=current.dtype,
        )
        return np.concatenate([current, padding], axis=0)
    return current


def _minmax_normalize_routerset_channels(current: np.ndarray) -> np.ndarray:
    normalized = np.array(current, dtype=np.float32, copy=True)
    for channel_index in range(normalized.shape[0]):
        channel = np.nan_to_num(
            normalized[channel_index],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            copy=False,
        )
        min_value = float(np.min(channel)) if channel.size > 0 else 0.0
        max_value = float(np.max(channel)) if channel.size > 0 else 0.0
        denom = max_value - min_value
        if denom > 1e-12:
            normalized[channel_index] = (channel - min_value) / denom
        else:
            normalized[channel_index] = np.zeros_like(channel, dtype=np.float32)
    return normalized


def _map_phi2fm_s2_to_student_channels(current: np.ndarray) -> np.ndarray:
    """
    Map PhilEO-Bench Sentinel-2 channels into the 8-channel student layout used by
    the original Phi2FM UNet/Myriad downstream runs.

    Input S2 order in Phi2FM:
    B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12

    Student-compatible 8-channel order:
    B02, B03, B04, PAN, B08, B05, B06, B07

    The raw S2 exports do not contain the PhiSat PAN channel, so that slot stays 0.
    """
    if current.shape[0] < 7:
        raise ValueError(f"Expected at least 7 S2 channels, got shape {tuple(current.shape)}")
    _, height, width = current.shape
    mapped = np.zeros((8, height, width), dtype=current.dtype)
    mapped[0] = current[0]  # B02
    mapped[1] = current[1]  # B03
    mapped[2] = current[2]  # B04
    mapped[4] = current[3]  # B08
    mapped[5] = current[4]  # B05
    mapped[6] = current[5]  # B06
    mapped[7] = current[6]  # B07
    return mapped


def _normalize_routerset_array_impl(
    array: np.ndarray,
    *,
    source_dataset: str,
    target_channels: int,
) -> tuple[np.ndarray, dict[str, Any], str, str]:
    sanitized, diagnostics = sanitize_routerset_array(np.asarray(array))
    original_dtype = str(sanitized.dtype)
    current = _normalize_to_channel_first(sanitized)
    current = current.astype(np.float32, copy=False)
    normalization_mode = "channel_adapter"

    if source_dataset in ROUTERSET_PHI2FM_RAW_S2_DATASETS:
        if target_channels == 8 and current.shape[0] >= 10:
            current = _map_phi2fm_s2_to_student_channels(current)
            normalization_mode = "phi2fm_student_s2_layout"
        else:
            current = _normalize_routerset_channel_count(current, target_channels=target_channels)
            normalization_mode = "phi2fm_student_channel_adapter"
        current_max = float(np.max(current)) if current.size > 0 else 0.0
        if np.issubdtype(sanitized.dtype, np.integer) or current_max > 1.5:
            current = current / ROUTERSET_PHI2FM_STUDENT_SCALE
            normalization_mode = f"{normalization_mode}_scaled"
        current = current.astype(np.float32, copy=False)
    else:
        current = _normalize_routerset_channel_count(current, target_channels=target_channels)
        if source_dataset in ROUTERSET_PHI2FM_FLOAT_MINMAX_DATASETS:
            current = _minmax_normalize_routerset_channels(current)
            normalization_mode = "phi2fm_student_float_minmax"

    return current, diagnostics, normalization_mode, original_dtype


def infer_routerset_normalization_mode(*, source_dataset: str, target_channels: int) -> str:
    if source_dataset in ROUTERSET_PHI2FM_RAW_S2_DATASETS:
        if target_channels == 8:
            return "phi2fm_student_s2_layout_scaled"
        return "phi2fm_student_channel_adapter_scaled"
    if source_dataset in ROUTERSET_PHI2FM_FLOAT_MINMAX_DATASETS:
        return "phi2fm_student_float_minmax"
    return "channel_adapter"


def normalize_routerset_array(
    array: np.ndarray,
    *,
    source_dataset: str,
    target_channels: int = 8,
) -> torch.Tensor:
    """Normalize any routerset source tensor into channel-first 8-channel float32."""
    current, _, _, _ = _normalize_routerset_array_impl(
        array,
        source_dataset=source_dataset,
        target_channels=target_channels,
    )
    if not current.flags.writeable:
        current = np.array(current, copy=True)
    return torch.from_numpy(current)


def describe_routerset_array(
    path: Union[str, Path],
    *,
    source_dataset: str,
    target_channels: int = 8,
) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r")
    normalized, sanitization, normalization_mode, original_dtype = _normalize_routerset_array_impl(
        array,
        source_dataset=source_dataset,
        target_channels=target_channels,
    )
    return {
        "path": str(path),
        "original_shape": list(array.shape),
        "original_dtype": original_dtype,
        "normalized_shape": list(normalized.shape),
        "source_dataset": source_dataset,
        "normalization_mode": normalization_mode,
        "had_non_finite": bool(sanitization["had_non_finite"]),
        "non_finite_count": int(sanitization["non_finite_count"]),
        "nan_count": int(sanitization["nan_count"]),
        "pos_inf_count": int(sanitization["pos_inf_count"]),
        "neg_inf_count": int(sanitization["neg_inf_count"]),
    }


def sample_routerset_raw_array(path: Union[str, Path]) -> dict[str, Any]:
    raw_array = np.load(path, mmap_mode="r")
    raw_view = np.asarray(raw_array)
    sample: dict[str, Any] = {
        "raw_shape": list(raw_view.shape),
        "raw_dtype": str(raw_view.dtype),
    }

    finite_mask = np.isfinite(raw_view)
    non_finite_count = int((~finite_mask).sum())
    sample["raw_non_finite_count"] = non_finite_count
    sample["raw_zero_fraction"] = float(np.count_nonzero(raw_view == 0) / raw_view.size) if raw_view.size > 0 else 0.0

    if non_finite_count >= raw_view.size:
        sample["raw_min"] = 0.0
        sample["raw_max"] = 0.0
        sample["raw_mean"] = 0.0
        return sample

    finite_values = raw_view[finite_mask] if non_finite_count > 0 else raw_view.reshape(-1)
    sample["raw_min"] = float(finite_values.min())
    sample["raw_max"] = float(finite_values.max())
    sample["raw_mean"] = float(finite_values.mean())
    return sample


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
            checkpoint_path = _loading_module()._resolve_student_checkpoint_path(
                task=task,
                training=training,
                n_shots=n_shots,
                weights_dir=weights_dir,
            )
            if not checkpoint_path:
                raise FileNotFoundError(f"Checkpoint path was empty for expert {task!r}.")
            item["checkpoint_path"] = checkpoint_path
            if weights_dir is not None:
                try:
                    relative_source = Path(checkpoint_path).resolve().relative_to(Path(weights_dir).resolve())
                    item["source_path"] = str(relative_source)
                    item["deterministic_path"] = str(Path(weights_dir) / relative_source)
                except ValueError:
                    pass
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
    source_had_non_finite: bool
    source_non_finite_count: int
    normalization_mode: str
    used_compatibility_path: bool
    used_manifest_shape: bool


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


def _load_student_expert_models(
    *,
    expert_names: Sequence[str],
    training: str,
    n_shots: int,
    weights_dir: Optional[str],
    device: Optional[str] = None,
) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    model_device = torch.device(device) if device is not None else None
    for expert_name in expert_names:
        model = _loading_module().load_student(
            task=expert_name,
            training=training,
            n_shots=n_shots,
            weights_dir=weights_dir,
            auto_load_weights=True,
        )
        if model_device is not None:
            model = model.to(model_device)
        model.eval()
        models[expert_name] = model
    return models


def _reduce_expert_output_to_routing_score(logits: torch.Tensor) -> float:
    if logits.ndim != 4:
        raise ValueError(f"Expected expert logits with shape (batch, channels, height, width), got {tuple(logits.shape)}")
    if logits.shape[1] <= 0:
        raise ValueError("Expert logits must contain at least one output channel.")

    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
        flat = probs.flatten(start_dim=1)
        k = max(1, flat.shape[1] // 256)
        score = flat.topk(k, dim=1).values.mean()
        return float(score.detach().cpu().item())

    probs = torch.softmax(logits, dim=1)
    foreground = probs[:, 1:, :, :]
    if foreground.numel() <= 0:
        foreground = probs
    score = foreground.mean()
    return float(score.detach().cpu().item())


def _f1_at_threshold(scores: Sequence[float], labels: Sequence[int], threshold: float) -> tuple[float, int, int, int]:
    tp = fp = fn = 0
    for score, label in zip(scores, labels):
        pred = 1 if float(score) >= float(threshold) else 0
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif (not pred) and label:
            fn += 1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1), int(tp), int(fp), int(fn)


def _calibrate_routing_thresholds(
    rows: Sequence[Mapping[str, Any]],
    *,
    expert_names: Sequence[str],
    score_rows: Sequence[Optional[Mapping[str, float]]],
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    expert_set = set(expert_names)
    for expert_name in expert_names:
        labels: list[int] = []
        scores: list[float] = []
        candidates = {0.5}
        for row, score_row in zip(rows, score_rows):
            if row.get("source_dataset") not in expert_set:
                continue
            if row.get("moe_split", row.get("source_split")) != "validation":
                continue
            if score_row is None:
                continue
            score = float(score_row[expert_name])
            scores.append(score)
            labels.append(1 if row.get("source_dataset") == expert_name else 0)
            candidates.add(score)

        if not scores:
            thresholds[expert_name] = 0.5
            continue

        best_threshold = 0.5
        best_f1, _, _, _ = _f1_at_threshold(scores, labels, best_threshold)
        for threshold in sorted(candidates):
            f1, _, _, _ = _f1_at_threshold(scores, labels, threshold)
            if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
                best_f1 = f1
                best_threshold = float(threshold)
        thresholds[expert_name] = float(best_threshold)
    return thresholds


def manifest_has_generated_routing_targets(
    routerset_dir: Union[str, Path],
    *,
    manifest_path: Optional[Union[str, Path]] = None,
    expert_names: Sequence[str],
) -> bool:
    expert_set = set(expert_names)
    for row in load_routerset_manifest_rows(routerset_dir, manifest_path):
        if row.get("source_dataset") not in expert_set:
            continue
        target = row.get("routing_target")
        source = row.get("routing_target_source")
        experts = row.get("routing_target_experts")
        scores = row.get("routing_scores")
        if not isinstance(target, list) or len(target) != len(expert_names):
            return False
        if source != ROUTING_TARGET_SOURCE:
            return False
        if not isinstance(experts, list):
            return False
        if not isinstance(scores, dict) or set(scores.keys()) != expert_set:
            return False
    return True


def generate_routerset_routing_targets(
    routerset_dir: Union[str, Path],
    *,
    manifest_path: Optional[Union[str, Path]] = None,
    expert_names: Sequence[str],
    training: str = "finetuning",
    n_shots: int = 5000,
    weights_dir: Optional[str] = None,
    target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
    target_channels: int = 8,
    output_path: Optional[Union[str, Path]] = None,
    report_path: Optional[Union[str, Path]] = None,
    force: bool = False,
    inference_batch_size: int = 64,
    inference_device: Optional[str] = None,
    inference_use_amp: bool = True,
) -> Path:
    source_manifest_path = resolve_routerset_manifest_path(routerset_dir, manifest_path)
    destination = Path(output_path) if output_path is not None else source_manifest_path
    rows = load_routerset_manifest_rows(routerset_dir, source_manifest_path)
    if not force and manifest_has_generated_routing_targets(routerset_dir, manifest_path=source_manifest_path, expert_names=expert_names):
        if report_path is not None:
            save_json(
                report_path,
                {
                    "status": "reused",
                    "manifest_path": str(source_manifest_path),
                    "expert_names": list(expert_names),
                    "target_source": ROUTING_TARGET_SOURCE,
                },
            )
        return destination

    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be a positive integer.")

    target_device = inference_device if inference_device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    expert_models = _load_student_expert_models(
        expert_names=expert_names,
        training=training,
        n_shots=n_shots,
        weights_dir=weights_dir,
        device=target_device,
    )
    expert_set = set(expert_names)
    score_rows: list[Optional[dict[str, float]]] = [None] * len(rows)
    indexed_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("source_dataset") in expert_set
    ]
    if not indexed_rows:
        raise ValueError(f"No manifest rows matched expert list {expert_set}.")

    for batch_start in range(0, len(indexed_rows), inference_batch_size):
        batch_slice = indexed_rows[batch_start : batch_start + inference_batch_size]
        batch_row_indices = [index for index, _ in batch_slice]
        batch_tensors = []
        for _, row in batch_slice:
            image_path = routerset_image_path(routerset_dir, row)
            if not image_path.exists():
                raise FileNotFoundError(f"Routerset image for routing target generation not found: {image_path}")
            array = np.load(image_path)
            tensor = build_routerset_training_tensor(
                array,
                source_dataset=str(row["source_dataset"]),
                target_size=target_size,
                target_channels=target_channels,
            )
            batch_tensors.append(tensor)

        batch_inputs = torch.stack(batch_tensors, dim=0)
        if target_device:
            batch_inputs = batch_inputs.to(target_device)

        with torch.no_grad():
            amp_ctx = torch.amp.autocast(
                device_type="cuda" if target_device.startswith("cuda") else "cpu",
                enabled=bool(inference_use_amp and target_device.startswith("cuda")),
            )
            with amp_ctx:
                for expert_name, model in expert_models.items():
                    logits = model(batch_inputs)
                    batch_scores = [
                        _reduce_expert_output_to_routing_score(logits[sample_index : sample_index + 1])
                        for sample_index in range(logits.shape[0])
                    ]
                    for row_index, score in zip(batch_row_indices, batch_scores):
                        row_scores = score_rows[row_index]
                        if row_scores is None:
                            row_scores = {}
                            score_rows[row_index] = row_scores
                        row_scores[expert_name] = score

    thresholds = _calibrate_routing_thresholds(
        rows,
        expert_names=expert_names,
        score_rows=score_rows,
    )

    updated_rows: list[dict[str, Any]] = []
    fallback_count = 0
    active_count_sum = 0
    activation_counts: dict[str, int] = {expert: 0 for expert in expert_names}
    for index, row in enumerate(rows):
        updated = dict(row)
        row_scores = score_rows[index]
        if row_scores is not None:
            active = [
                expert_name
                for expert_name in expert_names
                if float(row_scores[expert_name]) >= float(thresholds[expert_name])
            ]
            if not active:
                fallback_count += 1
                best_expert = max(expert_names, key=lambda name: float(row_scores[name]))
                active = [best_expert]
            target = [1.0 if expert_name in active else 0.0 for expert_name in expert_names]
            active_count_sum += len(active)
            for expert_name in active:
                activation_counts[expert_name] += 1
            updated["routing_target"] = target
            updated["routing_target_experts"] = list(active)
            updated["routing_scores"] = {expert_name: float(row_scores[expert_name]) for expert_name in expert_names}
            updated["routing_target_source"] = ROUTING_TARGET_SOURCE
        updated_rows.append(updated)

    payload = "\n".join(json.dumps(row, sort_keys=True) for row in updated_rows) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")

    if report_path is not None:
        generated_rows = sum(1 for row in updated_rows if row.get("routing_target_source") == ROUTING_TARGET_SOURCE)
        average_active = (active_count_sum / generated_rows) if generated_rows > 0 else 0.0
        save_json(
            report_path,
            {
                "status": "generated",
                "manifest_path": str(destination),
                "target_source": ROUTING_TARGET_SOURCE,
                "expert_names": list(expert_names),
                "thresholds_by_expert": thresholds,
                "activation_counts_by_expert": activation_counts,
                "generated_row_count": int(generated_rows),
                "fallback_row_count": int(fallback_count),
                "average_active_experts": float(average_active),
            },
        )

    return destination


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
        padded[:, :height, :width] = image
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


def _materialized_routerset_dataset_root(path: Union[str, Path]) -> Path:
    return ensure_dir(path)


def _materialized_routerset_root(base_dir: Union[str, Path]) -> Path:
    return _materialized_routerset_dataset_root(Path(base_dir) / "routerset_materialized")


def _materialized_routerset_manifest_path(dataset_root: Union[str, Path], *, target_size: int) -> Path:
    return _materialized_routerset_dataset_root(dataset_root) / f"manifest_{int(target_size)}.jsonl"


def _materialized_routerset_fault_rows_manifest_path(dataset_root: Union[str, Path], *, target_size: int) -> Path:
    return _materialized_routerset_dataset_root(dataset_root) / f"fault_rows_{int(target_size)}.jsonl"


def _materialized_routerset_image_dir(
    dataset_root: Union[str, Path],
    *,
    source_dataset: str,
    source_split: str,
) -> Path:
    root = _materialized_routerset_dataset_root(dataset_root)
    return ensure_dir(root / "images" / source_dataset / source_split)


def _tile_routerset_array(
    normalized: np.ndarray,
    *,
    target_size: int,
) -> List[tuple[np.ndarray, tuple[int, int]]]:
    _, height, width = normalized.shape
    tile_origins = anomaly_tile_origins(height, width, target_size)
    tiles: List[tuple[np.ndarray, tuple[int, int]]] = []
    for tile_y, tile_x in tile_origins:
        tile = normalized[:, tile_y : tile_y + target_size, tile_x : tile_x + target_size]
        tensor = crop_or_pad_routerset_tensor(torch.from_numpy(tile), target_size=target_size)
        tiles.append((tensor.numpy(), (tile_y, tile_x)))
    return tiles


def _materialize_routerset_anomaly_tiles(
    array: np.ndarray,
    *,
    source_dataset: str,
    target_size: int,
    target_channels: int,
) -> List[tuple[np.ndarray, tuple[int, int]]]:
    channel_first = _normalize_to_channel_first(np.asarray(array))
    _, height, width = channel_first.shape
    tile_payloads: List[tuple[np.ndarray, tuple[int, int]]] = []
    for tile_y, tile_x in anomaly_tile_origins(height, width, target_size):
        raw_tile = channel_first[:, tile_y : tile_y + target_size, tile_x : tile_x + target_size]
        tile = build_routerset_training_tensor(
            raw_tile,
            source_dataset=source_dataset,
            target_size=target_size,
            target_channels=target_channels,
        ).numpy()
        tile_payloads.append((tile, (tile_y, tile_x)))
    return tile_payloads


def _write_materialized_routerset_sample(
    *,
    dataset_root: Union[str, Path],
    row: Mapping[str, Any],
    array: np.ndarray,
    source_image_path: Union[str, Path],
    patch_x: int,
    patch_y: int,
    target_size: int,
    image_ref_mode: str,
) -> tuple[dict[str, Any], bool]:
    tile_row = dict(row)
    tile_row["patch_x"] = int(patch_x)
    tile_row["patch_y"] = int(patch_y)
    tile_row["patch_width"] = int(target_size)
    tile_row["patch_height"] = int(target_size)
    tile_row["image_ref"] = f"{row.get('image_ref', 'materialized')}::{image_ref_mode}:{int(patch_x)}:{int(patch_y)}:{int(target_size)}:{int(target_size)}"
    tile_row["materialized_from"] = str(source_image_path)
    tile_filename = routerset_patch_token(tile_row)
    tile_path = _materialized_routerset_image_dir(
        dataset_root,
        source_dataset=str(row["source_dataset"]),
        source_split=str(row["source_split"]),
    ) / tile_filename
    tile_row["materialized_image_path"] = str(tile_path)
    created = not tile_path.exists()
    if created:
        np.save(tile_path, array.astype(np.float32, copy=False))
    return tile_row, created


def _classify_materialized_routerset_tile_faults(array: np.ndarray) -> List[str]:
    if array.size <= 0:
        return ["empty_materialized_tile"]
    if not np.isfinite(array).all():
        return ["non_finite_materialized_tile"]
    if float(np.max(array)) == 0.0 and float(np.min(array)) == 0.0:
        return ["all_zero_materialized_tile"]
    return []


def _append_materialized_fault_example(
    summary: dict[str, Any],
    *,
    fault_code: str,
    row: Mapping[str, Any],
) -> None:
    examples = summary["fault_examples_by_code"].setdefault(fault_code, [])
    if len(examples) >= DEFAULT_ROUTERSET_FAULT_EXAMPLES_PER_CODE:
        return
    examples.append(
        {
            "source_dataset": str(row["source_dataset"]),
            "source_split": str(row["source_split"]),
            "source_sample_id": str(row["source_sample_id"]),
            "patch_x": int(row.get("patch_x") or 0),
            "patch_y": int(row.get("patch_y") or 0),
            "patch_width": row.get("patch_width"),
            "patch_height": row.get("patch_height"),
            "record_status": str(row.get("record_status", "")),
            "selection_bucket": str(row.get("selection_bucket", "")),
            "candidate_materialized_image_path": str(row.get("candidate_materialized_image_path") or row.get("materialized_image_path") or ""),
        }
    )


def _collect_routerset_training_blockers(
    dataset_report: Mapping[str, Any],
    *,
    expert_names: Sequence[str],
) -> List[dict[str, Any]]:
    blockers: List[dict[str, Any]] = []
    for split_name in ("train", "validation"):
        split_report = dataset_report[split_name]
        raw_counts = split_report["raw_counts_by_expert"]
        raw_positive_counts = split_report["raw_positive_counts_by_expert"]
        for expert_name in expert_names:
            if int(raw_counts.get(expert_name, 0)) <= 0:
                blockers.append(
                    {
                        "code": "missing_records",
                        "expert": expert_name,
                        "split": split_name,
                        "count": int(raw_counts.get(expert_name, 0)),
                    }
                )
            if int(raw_positive_counts.get(expert_name, 0)) <= 0:
                blockers.append(
                    {
                        "code": "missing_positive_rows",
                        "expert": expert_name,
                        "split": split_name,
                        "count": int(raw_positive_counts.get(expert_name, 0)),
                    }
                )
    return blockers


def _materialize_routerset_manifest_to_dataset_root(
    *,
    routerset_dir: Union[str, Path],
    manifest_path: Union[str, Path],
    dataset_root: Union[str, Path],
    expert_names: Sequence[str],
    target_size: int,
    target_channels: int,
    materialize_all: bool,
    strict_missing: bool,
    clean_export: bool = False,
    selected_only: bool = False,
) -> tuple[Path, dict[str, Any]]:
    active_manifest_path = Path(manifest_path)
    rows = load_routerset_manifest_rows(routerset_dir, active_manifest_path)
    expert_set = set(expert_names)
    materialized_manifest_path = _materialized_routerset_manifest_path(dataset_root, target_size=target_size)
    fault_rows_manifest_path = _materialized_routerset_fault_rows_manifest_path(dataset_root, target_size=target_size)
    rewritten_rows: List[dict[str, Any]] = []
    fault_rows: List[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "source_row_count": 0,
        "materialized_row_count": 0,
        "source_rows_by_expert": {},
        "materialized_rows_by_expert": {},
        "created_file_count": 0,
        "reused_file_count": 0,
        "missing_source_rows": 0,
        "passthrough_rows": 0,
        "skipped_unselected_rows": 0,
        "skipped_unselected_rows_by_expert": {},
        "metadata_only_rows": 0,
        "single_file_source_rows": 0,
        "tiled_source_rows": 0,
        "materialize_all": bool(materialize_all),
        "clean_export": bool(clean_export),
        "selected_only": bool(selected_only),
        "fault_row_count": 0,
        "fault_rows_by_code": {},
        "fault_rows_by_expert": {},
        "fault_examples_by_code": {},
        "excluded_row_count": 0,
        "excluded_rows_by_expert": {},
        "fault_rows_manifest_path": str(fault_rows_manifest_path) if materialize_all else "",
    }

    for row in rows:
        source_dataset = str(row["source_dataset"])
        summary["source_row_count"] += 1
        summary["source_rows_by_expert"][source_dataset] = int(summary["source_rows_by_expert"].get(source_dataset, 0)) + 1
        if source_dataset not in expert_set:
            if selected_only:
                summary["skipped_unselected_rows"] += 1
                summary["skipped_unselected_rows_by_expert"][source_dataset] = int(
                    summary["skipped_unselected_rows_by_expert"].get(source_dataset, 0)
                ) + 1
                continue
            rewritten_rows.append(dict(row))
            summary["passthrough_rows"] += 1
            continue

        patch_width = row.get("patch_width")
        patch_height = row.get("patch_height")
        if not materialize_all:
            if patch_width == target_size and patch_height == target_size:
                rewritten_rows.append(dict(row))
                summary["passthrough_rows"] += 1
                continue

            # Smaller routerset patches are padded during training; only materialize
            # samples whose size is unknown or larger than the canonical tile size.
            if (
                patch_width is not None
                and patch_height is not None
                and patch_width <= target_size
                and patch_height <= target_size
            ):
                rewritten_rows.append(dict(row))
                summary["passthrough_rows"] += 1
                continue

            # Fire patches currently use full/full metadata even though the stored
            # arrays are already 256x256, so avoid reopening every file on startup.
            if source_dataset == "fire" and patch_width is None and patch_height is None:
                source_image_path = routerset_image_path(routerset_dir, row)
                normalized_row = dict(row)
                normalized_row["patch_width"] = int(target_size)
                normalized_row["patch_height"] = int(target_size)
                normalized_row["materialized_image_path"] = str(source_image_path)
                rewritten_rows.append(normalized_row)
                summary["metadata_only_rows"] += 1
                continue

        source_image_path = routerset_image_path(routerset_dir, row)
        if not source_image_path.exists():
            summary["missing_source_rows"] += 1
            if strict_missing:
                raise FileNotFoundError(f"Missing routerset source image for row {routerset_patch_token(row)}: {source_image_path}")
            rewritten_rows.append(dict(row))
            continue

        array = np.load(source_image_path, mmap_mode="r")
        base_patch_x = int(row.get("patch_x") or 0)
        base_patch_y = int(row.get("patch_y") or 0)

        if source_dataset == "anomaly_detection":
            channel_first = _normalize_to_channel_first(np.asarray(array))
            _, height, width = channel_first.shape

            if not materialize_all:
                needs_materialization = (
                    height != target_size
                    or width != target_size
                    or patch_width != target_size
                    or patch_height != target_size
                )
                if not needs_materialization:
                    rewritten_rows.append(dict(row))
                    summary["passthrough_rows"] += 1
                    continue

            tile_payloads = [
                (tile_array, base_patch_x + int(tile_x), base_patch_y + int(tile_y), "tile")
                for tile_array, (tile_y, tile_x) in _materialize_routerset_anomaly_tiles(
                    array,
                    source_dataset=source_dataset,
                    target_size=target_size,
                    target_channels=target_channels,
                )
            ]
            summary["tiled_source_rows"] += 1
        else:
            if materialize_all:
                prepared = build_routerset_training_tensor(
                    array,
                    source_dataset=source_dataset,
                    target_size=target_size,
                    target_channels=target_channels,
                ).numpy()
                _, height, width = prepared.shape
            else:
                prepared = normalize_routerset_array(
                    array,
                    source_dataset=source_dataset,
                    target_channels=target_channels,
                ).numpy()
                _, height, width = prepared.shape

            if not materialize_all:
                needs_materialization = (
                    height != target_size
                    or width != target_size
                    or patch_width != target_size
                    or patch_height != target_size
                )
                if not needs_materialization:
                    rewritten_rows.append(dict(row))
                    summary["passthrough_rows"] += 1
                    continue

            cropped = prepared if materialize_all else crop_or_pad_routerset_tensor(torch.from_numpy(prepared), target_size=target_size).numpy()
            tile_payloads = [(cropped, base_patch_x, base_patch_y, "materialized")]
            summary["single_file_source_rows"] += 1

        for tile_array, tile_patch_x, tile_patch_y, image_ref_mode in tile_payloads:
            tile_row_preview = dict(row)
            tile_row_preview["patch_x"] = int(tile_patch_x)
            tile_row_preview["patch_y"] = int(tile_patch_y)
            tile_row_preview["patch_width"] = int(target_size)
            tile_row_preview["patch_height"] = int(target_size)
            tile_row_preview["image_ref"] = (
                f"{row.get('image_ref', 'materialized')}::{image_ref_mode}:{int(tile_patch_x)}:{int(tile_patch_y)}:{int(target_size)}:{int(target_size)}"
            )
            tile_row_preview["materialized_from"] = str(source_image_path)
            tile_row_preview["candidate_materialized_image_path"] = str(
                _materialized_routerset_image_dir(
                    dataset_root,
                    source_dataset=str(row["source_dataset"]),
                    source_split=str(row["source_split"]),
                )
                / routerset_patch_token(tile_row_preview)
            )
            fault_codes = _classify_materialized_routerset_tile_faults(tile_array)
            if fault_codes:
                summary["fault_row_count"] += 1
                summary["fault_rows_by_expert"][source_dataset] = int(summary["fault_rows_by_expert"].get(source_dataset, 0)) + 1
                fault_row = dict(tile_row_preview)
                fault_row["fault_codes"] = list(fault_codes)
                fault_row["fault_action"] = "excluded_from_clean_export" if clean_export else "retained_in_export"
                for fault_code in fault_codes:
                    summary["fault_rows_by_code"][fault_code] = int(summary["fault_rows_by_code"].get(fault_code, 0)) + 1
                    _append_materialized_fault_example(summary, fault_code=fault_code, row=fault_row)
                if clean_export:
                    summary["excluded_row_count"] += 1
                    summary["excluded_rows_by_expert"][source_dataset] = int(summary["excluded_rows_by_expert"].get(source_dataset, 0)) + 1
                    fault_rows.append(fault_row)
                    continue
                fault_rows.append(fault_row)

            tile_row, created = _write_materialized_routerset_sample(
                dataset_root=dataset_root,
                row=row,
                array=tile_array,
                source_image_path=source_image_path,
                patch_x=tile_patch_x,
                patch_y=tile_patch_y,
                target_size=target_size,
                image_ref_mode=image_ref_mode,
            )
            rewritten_rows.append(tile_row)
            summary["materialized_row_count"] += 1
            summary["materialized_rows_by_expert"][source_dataset] = int(summary["materialized_rows_by_expert"].get(source_dataset, 0)) + 1
            if created:
                summary["created_file_count"] += 1
            else:
                summary["reused_file_count"] += 1

    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rewritten_rows) + "\n"
    materialized_manifest_path.write_text(payload, encoding="utf-8")
    if materialize_all:
        fault_payload = "\n".join(json.dumps(row, sort_keys=True) for row in fault_rows)
        fault_rows_manifest_path.write_text((fault_payload + "\n") if fault_payload else "", encoding="utf-8")
    return materialized_manifest_path, summary


def materialize_routerset_training_manifest(
    *,
    routerset_dir: Union[str, Path],
    manifest_path: Union[str, Path],
    materialization_root: Union[str, Path],
    expert_names: Sequence[str],
    target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
    target_channels: int = 8,
) -> Path:
    materialized_manifest_path, _ = _materialize_routerset_manifest_to_dataset_root(
        routerset_dir=routerset_dir,
        manifest_path=manifest_path,
        dataset_root=_materialized_routerset_root(materialization_root),
        expert_names=expert_names,
        target_size=target_size,
        target_channels=target_channels,
        materialize_all=False,
        strict_missing=False,
        selected_only=False,
    )
    return materialized_manifest_path


def materialize_routerset_dataset(
    *,
    routerset_dir: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    expert_names: Optional[Sequence[str]] = None,
    target_size: int = DEFAULT_ROUTERSET_TARGET_SIZE,
    target_channels: int = 8,
    clean_export: bool = False,
    selected_only: bool = False,
) -> dict[str, Any]:
    experts = list(expert_names or DEFAULT_ROUTERSET_EXPERTS)
    dataset_root = _materialized_routerset_dataset_root(output_dir)
    active_manifest_path = resolve_routerset_manifest_path(routerset_dir, manifest_path)
    materialized_manifest_path, core_summary = _materialize_routerset_manifest_to_dataset_root(
        routerset_dir=routerset_dir,
        manifest_path=active_manifest_path,
        dataset_root=dataset_root,
        expert_names=experts,
        target_size=target_size,
        target_channels=target_channels,
        materialize_all=True,
        strict_missing=True,
        clean_export=clean_export,
        selected_only=selected_only,
    )
    dataset_report = collect_routerset_dataset_report(
        routerset_dir,
        manifest_path=materialized_manifest_path,
        expert_names=experts,
        target_size=target_size,
        target_channels=target_channels,
        balanced_sampling=True,
        num_workers=0,
    )
    validate_materialized_routerset_export_report(
        dataset_report,
        expert_names=experts,
        target_size=target_size,
        target_channels=target_channels,
    )
    dataset_report_path = save_json(dataset_root / "dataset_report.json", dataset_report)
    training_blockers = _collect_routerset_training_blockers(dataset_report, expert_names=experts)
    fault_report = {
        "export_mode": "clean" if clean_export else "canonical",
        "selected_only": bool(selected_only),
        "phi2fm_reference": {
            "repository": "https://github.com/carlos-collado/phi2FM",
            "training_alignment": "roads_and_lc_follow_generic_downstream_student_reflectance_scaling",
        },
        "row_fault_count": int(core_summary["fault_row_count"]),
        "row_faults_by_code": dict(core_summary["fault_rows_by_code"]),
        "row_faults_by_expert": dict(core_summary["fault_rows_by_expert"]),
        "fault_examples_by_code": dict(core_summary["fault_examples_by_code"]),
        "excluded_row_count": int(core_summary["excluded_row_count"]),
        "excluded_rows_by_expert": dict(core_summary["excluded_rows_by_expert"]),
        "fault_rows_manifest_path": str(core_summary.get("fault_rows_manifest_path", "")),
        "training_ready": not training_blockers,
        "training_blockers": training_blockers,
    }
    fault_report_path = save_json(dataset_root / "fault_report.json", fault_report)

    label_vocab_path = resolve_routerset_dataset_root(routerset_dir) / "label_vocab.json"
    copied_label_vocab_path = ""
    if label_vocab_path.exists():
        copied = dataset_root / "label_vocab.json"
        shutil.copy2(label_vocab_path, copied)
        copied_label_vocab_path = str(copied)

    materialized_paths = {
        str(row["materialized_image_path"])
        for row in load_routerset_manifest_rows(routerset_dir, materialized_manifest_path)
        if row.get("materialized_image_path")
    }
    materialized_total_bytes = sum(Path(path).stat().st_size for path in materialized_paths)
    summary = {
        "status": "materialized",
        "routerset_dir": str(routerset_dir),
        "source_manifest_path": str(active_manifest_path),
        "materialized_dataset_root": str(dataset_root),
        "manifest_path": str(materialized_manifest_path),
        "dataset_report_path": str(dataset_report_path),
        "fault_report_path": str(fault_report_path),
        "label_vocab_path": copied_label_vocab_path,
        "experts": experts,
        "target_size": int(target_size),
        "target_channels": int(target_channels),
        "export_mode": "clean" if clean_export else "canonical",
        "selected_only": bool(selected_only),
        "materialized_unique_file_count": int(len(materialized_paths)),
        "materialized_total_bytes": int(materialized_total_bytes),
        "corrections_applied": [
            "all_selected_rows_are_saved_as_concrete_npy_files",
            f"all_exported_tensors_are_channel_first_8x{int(target_size)}x{int(target_size)}",
            "roads_and_lc_follow_phi2fm_student_band_mapping_and_scaling",
            "small_patches_are_zero_padded_after_normalization",
            "oversized_anomaly_detection_arrays_are_tiled_deterministically",
            "swapped_coordinate_source_paths_are_resolved_and_rewritten_to_canonical_filenames",
        ],
        "dataset_report": dataset_report,
        "fault_report": fault_report,
        "materialization_summary": core_summary,
    }
    save_json(dataset_root / "materialization_summary.json", summary)
    return summary


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
            used_compatibility_path = routerset_uses_compatibility_path(row, image_path)

            patch_width = row.get("patch_width")
            patch_height = row.get("patch_height")
            use_manifest_shape = (
                isinstance(patch_width, (int, float))
                and isinstance(patch_height, (int, float))
                and int(patch_width) > 0
                and int(patch_height) > 0
            )
            description: Optional[dict[str, Any]] = None
            if use_manifest_shape:
                normalized_shape = [self.target_channels, int(patch_height), int(patch_width)]
                original_shape = list(normalized_shape)
                source_had_non_finite = False
                source_non_finite_count = 0
                normalization_mode = infer_routerset_normalization_mode(
                    source_dataset=source_dataset,
                    target_channels=self.target_channels,
                )
            else:
                description = describe_routerset_array(
                    image_path,
                    source_dataset=source_dataset,
                    target_channels=self.target_channels,
                )
                normalized_shape = list(description["normalized_shape"])
                original_shape = list(description["original_shape"])
                source_had_non_finite = bool(description.get("had_non_finite", False))
                source_non_finite_count = int(description.get("non_finite_count", 0))
                normalization_mode = str(description.get("normalization_mode", "channel_adapter"))
            raw_target = row.get("routing_target")
            if isinstance(raw_target, list) and len(raw_target) == len(self.expert_names):
                target = [float(value) for value in raw_target]
            else:
                target = [1.0 if name == source_dataset else 0.0 for name in self.expert_names]
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
                        original_shape=original_shape,
                        normalized_shape=normalized_shape,
                        training_shape=[self.target_channels, self.target_size, self.target_size],
                        tile_origin=[tile_y, tile_x],
                        tile_size=[self.target_size, self.target_size],
                        is_tiled=len(tile_origins) > 1,
                        source_had_non_finite=source_had_non_finite,
                        source_non_finite_count=source_non_finite_count,
                        normalization_mode=normalization_mode,
                        used_compatibility_path=used_compatibility_path,
                        used_manifest_shape=use_manifest_shape,
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
        non_finite_source_counts: Dict[str, int] = {name: 0 for name in self.expert_names}
        sanitized_tensor_counts: Dict[str, int] = {name: 0 for name in self.expert_names}
        non_finite_value_counts: Dict[str, int] = {name: 0 for name in self.expert_names}
        compatibility_path_counts: Dict[str, int] = {name: 0 for name in self.expert_names}
        manifest_shape_counts: Dict[str, int] = {name: 0 for name in self.expert_names}
        normalization_modes: Dict[str, Dict[str, int]] = {name: {} for name in self.expert_names}
        manifest_shape_samples: Dict[str, List[dict[str, Any]]] = {name: [] for name in self.expert_names}
        seen_rows: set[tuple[str, str]] = set()
        seen_positive_rows: set[tuple[str, str]] = set()
        seen_non_finite_rows: set[tuple[str, str]] = set()
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
                mode_counts = normalization_modes[record.source_dataset]
                mode_counts[record.normalization_mode] = mode_counts.get(record.normalization_mode, 0) + 1
                if record.used_compatibility_path:
                    compatibility_path_counts[record.source_dataset] += 1
                if record.used_manifest_shape:
                    manifest_shape_counts[record.source_dataset] += 1
                    samples = manifest_shape_samples[record.source_dataset]
                    if len(samples) < DEFAULT_ROUTERSET_REPORT_RAW_SAMPLES_PER_EXPERT:
                        sample: dict[str, Any] = {
                            "row_token": record.row_token,
                            "source_sample_id": record.base_source_sample_id,
                            "image_path": str(record.image_path),
                            "normalization_mode": record.normalization_mode,
                            "used_compatibility_path": bool(record.used_compatibility_path),
                        }
                        try:
                            sample.update(sample_routerset_raw_array(record.image_path))
                        except BaseException as error:
                            sample["error_type"] = type(error).__name__
                            sample["error"] = str(error)
                        samples.append(sample)
            if record.record_status == "positive" and raw_key not in seen_positive_rows:
                seen_positive_rows.add(raw_key)
                raw_positive_counts[record.source_dataset] += 1
            if record.source_had_non_finite:
                sanitized_tensor_counts[record.source_dataset] += 1
                non_finite_value_counts[record.source_dataset] += record.source_non_finite_count
                if raw_key not in seen_non_finite_rows:
                    seen_non_finite_rows.add(raw_key)
                    non_finite_source_counts[record.source_dataset] += 1
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
            "non_finite_source_records_by_expert": non_finite_source_counts,
            "sanitized_training_tensors_by_expert": sanitized_tensor_counts,
            "non_finite_values_by_expert": non_finite_value_counts,
            "compatibility_path_source_records_by_expert": compatibility_path_counts,
            "manifest_shape_source_records_by_expert": manifest_shape_counts,
            "manifest_shape_raw_samples_by_expert": manifest_shape_samples,
            "normalization_modes_by_expert": normalization_modes,
            "num_non_finite_source_records": int(sum(non_finite_source_counts.values())),
            "num_sanitized_training_tensors": int(sum(sanitized_tensor_counts.values())),
            "num_non_finite_values": int(sum(non_finite_value_counts.values())),
            "num_compatibility_path_source_records": int(sum(compatibility_path_counts.values())),
            "num_manifest_shape_source_records": int(sum(manifest_shape_counts.values())),
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
    return _loading_module().load_student_moe(
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
    num_workers: int = 0,
) -> dict[str, Any]:
    datamodule = RoutersetMoEDataModule(
        routerset_dir,
        manifest_path=manifest_path,
        expert_names=expert_names,
        batch_size=1,
        num_workers=num_workers,
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


def validate_materialized_routerset_export_report(
    dataset_report: Mapping[str, Any],
    *,
    expert_names: Sequence[str],
    target_size: int,
    target_channels: int,
) -> None:
    expected_shape = f"{target_channels}x{target_size}x{target_size}"
    invalid_shapes: Dict[str, Dict[str, Mapping[str, int]]] = {}
    total_counts: Dict[str, int] = {expert_name: 0 for expert_name in expert_names}

    for split_name in ("train", "validation"):
        split_report = dataset_report[split_name]
        counts = split_report["raw_counts_by_expert"]
        for expert_name in expert_names:
            total_counts[expert_name] += int(counts.get(expert_name, 0))
        for expert_name, shapes in split_report["training_shapes_by_expert"].items():
            if list(shapes.keys()) != [expected_shape]:
                invalid_shapes.setdefault(split_name, {})[expert_name] = shapes

    missing_experts = [expert_name for expert_name, count in total_counts.items() if count <= 0]
    if missing_experts:
        raise ValueError(
            "Materialized routerset export is missing selected experts: "
            f"{', '.join(sorted(missing_experts))}."
        )

    if invalid_shapes:
        raise ValueError(
            "Materialized routerset export produced non-canonical tensor shapes: "
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
    runtime_root: Optional[Union[str, Path]] = None,
    release_name: Optional[str] = None,
    rebuild_splits: bool = False,
    rebuilt_manifest_out: Optional[Union[str, Path]] = None,
    balanced_sampling: bool = True,
    dataset_num_workers: int = 0,
    inference_batch_size: int = 64,
    inference_device: Optional[str] = None,
    inference_use_amp: bool = True,
    skip_existing_routing_targets: bool = False,
) -> dict[str, Any]:
    experts = list(expert_names or DEFAULT_ROUTERSET_EXPERTS)
    active_manifest_path = resolve_routerset_manifest_path(routerset_dir, manifest_path)
    rebuilt_manifest_path = None
    materialization_base = Path(runtime_root) if runtime_root is not None else (Path(output_dir) if output_dir is not None else None)
    dataset_report_path = Path(output_dir) / "dataset_report.json" if output_dir is not None else None
    checkpoint_report_path = Path(output_dir) / "checkpoint_report.json" if output_dir is not None else None
    routing_target_report_path = Path(output_dir) / "routing_target_report.json" if output_dir is not None else None
    if rebuild_splits:
        rebuilt_manifest_path = rebuild_routerset_split_manifest(
            routerset_dir,
            expert_names=experts,
            manifest_path=active_manifest_path,
            output_path=rebuilt_manifest_out,
        )
        active_manifest_path = rebuilt_manifest_path
    if materialization_base is not None:
        active_manifest_path = materialize_routerset_training_manifest(
            routerset_dir=routerset_dir,
            manifest_path=active_manifest_path,
            materialization_root=materialization_base,
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
    if checkpoint_report_path is not None:
        save_json(checkpoint_report_path, checkpoint_report)
    validate_student_checkpoint_report(checkpoint_report, expert_names=experts)
    active_manifest_path = generate_routerset_routing_targets(
        routerset_dir=routerset_dir,
        manifest_path=active_manifest_path,
        output_path=active_manifest_path,
        expert_names=experts,
        training=training,
        n_shots=n_shots,
        weights_dir=weights_dir,
        target_size=target_size,
        target_channels=target_channels,
        report_path=routing_target_report_path,
        force=not skip_existing_routing_targets,
        inference_batch_size=inference_batch_size,
        inference_device=inference_device,
        inference_use_amp=inference_use_amp,
    )
    dataset_report = collect_routerset_dataset_report(
        routerset_dir,
        manifest_path=active_manifest_path,
        expert_names=experts,
        target_size=target_size,
        target_channels=target_channels,
        balanced_sampling=balanced_sampling,
        num_workers=dataset_num_workers,
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
    payload = {
        "release_name": resolve_release_name(release_name),
        "routerset_dir": str(routerset_dir),
        "manifest_path": str(active_manifest_path),
        "rebuilt_manifest_path": str(rebuilt_manifest_path) if rebuilt_manifest_path is not None else None,
        "materialized_manifest_path": str(active_manifest_path) if materialization_base is not None else None,
        "experts": experts,
        "training": training,
        "n_shots": int(n_shots),
        "target_size": int(target_size),
        "target_channels": int(target_channels),
        "balanced_sampling": bool(balanced_sampling),
        "dataset_report": dataset_report,
        "checkpoint_report": checkpoint_report,
        "routing_target_report_path": str(routing_target_report_path) if routing_target_report_path is not None else "",
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
    dataset_num_workers: int = 0,
    inference_batch_size: int = 64,
    inference_device: Optional[str] = None,
    inference_use_amp: bool = True,
    skip_existing_routing_targets: bool = False,
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
        runtime_root=runtime_report["runtime_root"],
        release_name=release_name,
        rebuild_splits=rebuild_splits,
        rebuilt_manifest_out=rebuilt_manifest_out,
        balanced_sampling=balanced_sampling,
        dataset_num_workers=dataset_num_workers,
        inference_batch_size=inference_batch_size,
        inference_device=inference_device,
        inference_use_amp=inference_use_amp,
        skip_existing_routing_targets=skip_existing_routing_targets,
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
- `dataset_report.json`: train/validation dataset summary after routerset adaptation, including normalization-mode, compatibility-path, and sampled raw-shape/value-range diagnostics
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
    output_dir: Union[str, Path],
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
    release_dir = ensure_within_root(Path(release_root) / release_name, root=output_dir, label="release_dir")
    release_dir = ensure_dir(release_dir)
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
    model = _loading_module().load_student_moe_bundle(bundle_path)
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
    release_root: Optional[Union[str, Path]] = None,
    rebuild_splits: bool = False,
    rebuilt_manifest_out: Optional[Union[str, Path]] = None,
    balanced_sampling: bool = True,
    runtime_root: Optional[Union[str, Path]] = None,
    accelerator: str = "cpu",
    devices: Union[str, int, Sequence[int]] = 1,
    precision: Optional[str] = None,
    dataset_num_workers: int = 0,
    inference_batch_size: int = 64,
    inference_device: Optional[str] = None,
    inference_use_amp: bool = True,
    skip_existing_routing_targets: bool = False,
    skip_dataset_report: bool = False,
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
    routing_target_report_path = Path(output_dir) / "routing_target_report.json"
    preflight_report_path = Path(output_dir) / "preflight_report.json"
    metrics_path = Path(output_dir) / "metrics.json"
    summary_path = Path(output_dir) / "summary.json"
    artifact_layout = build_artifact_layout(
        output_dir=output_dir,
        runtime_root=runtime_report["runtime_root"],
        release_name=release_name_value,
        release_root=release_root,
    )
    startup_gate_path = Path(output_dir) / "startup_gate.json"
    config: Optional[dict[str, Any]] = None

    try:
        if str(accelerator).lower() in {"gpu", "cuda", "auto"} and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            recorder.mark("matmul_precision_set", precision="high")
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

        active_manifest_path = materialize_routerset_training_manifest(
            routerset_dir=routerset_dir,
            manifest_path=active_manifest_path,
            materialization_root=runtime_report["runtime_root"],
            expert_names=expert_names,
            target_size=target_size,
            target_channels=target_channels,
        )
        manifest_path_value = str(active_manifest_path)
        recorder.mark("materialized_manifest_written", manifest_path=manifest_path_value)

        checkpoint_report = resolve_student_checkpoint_report(
            expert_names,
            training=training,
            n_shots=n_shots,
            weights_dir=runtime_report["weights_dir"],
        )
        save_json(checkpoint_report_path, checkpoint_report)
        recorder.mark("checkpoint_report_written", checkpoint_report_path=str(checkpoint_report_path))
        validate_student_checkpoint_report(checkpoint_report, expert_names=expert_names)

        active_manifest_path = generate_routerset_routing_targets(
            routerset_dir=routerset_dir,
            manifest_path=active_manifest_path,
            output_path=active_manifest_path,
            expert_names=expert_names,
            training=training,
            n_shots=n_shots,
            weights_dir=runtime_report["weights_dir"],
            target_size=target_size,
            target_channels=target_channels,
            report_path=routing_target_report_path,
            force=not skip_existing_routing_targets,
            inference_batch_size=inference_batch_size,
            inference_device=inference_device,
            inference_use_amp=inference_use_amp,
        )
        manifest_path_value = str(active_manifest_path)
        recorder.mark("routing_targets_generated", manifest_path=manifest_path_value)

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

        if skip_dataset_report:
            dataset_report = {
                "status": "skipped",
                "reason": "dataset report skipped by caller",
                "manifest_path": manifest_path_value,
                "expert_names": list(expert_names),
                "target_size": int(target_size),
                "target_channels": int(target_channels),
                "balanced_sampling": bool(balanced_sampling),
            }
            save_json(dataset_report_path, dataset_report)
            recorder.mark("dataset_report_skipped", dataset_report_path=str(dataset_report_path))
        else:
            dataset_report = collect_routerset_dataset_report(
                routerset_dir,
                manifest_path=active_manifest_path,
                expert_names=expert_names,
                target_size=target_size,
                target_channels=target_channels,
                balanced_sampling=balanced_sampling,
                num_workers=dataset_num_workers,
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
                "routing_target_report_path": str(routing_target_report_path),
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

        bundle_path = _loading_module().save_student_moe_bundle(
            model,
            Path(output_dir) / "student_moe_bundle.pt",
            metadata=config,
        )
        prediction_path = write_routing_predictions(
            model,
            datamodule.val_dataloader(),
            Path(output_dir) / "routing_predictions.jsonl",
        )

        release_summary = create_phidranet_release(
            output_dir=output_dir,
            release_root=artifact_layout["directories"]["bundle_root"],
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
            runtime_root=runtime_report["runtime_root"],
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
        non_finite_loss = extract_non_finite_loss_details(error)
        if non_finite_loss is not None:
            non_finite_loss["config_snapshot"] = dict(config or {})
            recorder.mark(
                "non_finite_loss_detected",
                update_current_stage=False,
                error_type=type(error).__name__,
                failure_stage=recorder.failure_stage or recorder.current_stage,
                non_finite_loss=non_finite_loss,
            )
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
                runtime_root=runtime_report["runtime_root"],
            ),
        }
        if non_finite_loss is not None:
            failure_summary["non_finite_loss"] = non_finite_loss
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
    release_root: Optional[Union[str, Path]] = None,
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
                    runtime_root=training_summary["runtime_root"],
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
                        runtime_root=training_summary["runtime_root"],
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
    release_root: Optional[Union[str, Path]] = None,
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
