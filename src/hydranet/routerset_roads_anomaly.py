"""Helpers for rebuilding routerset roads mosaics and anomaly tiles."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ANOMALY_LABEL_MAPPING = {
    1: "water",
    2: "turbid_water",
    3: "land",
    4: "marine_plastic",
    5: "marine_oil",
    6: "marine_algae",
    7: "marine_sediments",
    8: "cloud",
}

RARE_LABELS = frozenset(
    {
        "active_fire",
        "burned_area",
        "marine_plastic",
        "marine_oil",
        "marine_algae",
        "marine_sediments",
        "road_present",
    }
)


def load_label_vocab(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"Invalid label vocab payload at {path}")
    return [str(label) for label in labels]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(f"{json.dumps(dict(row), sort_keys=True)}\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def threshold_for_label(label: str) -> float:
    if label == "road_present":
        return 0.005
    return 0.001 if label in RARE_LABELS else 0.05


def sanitize_coverages(coverages: Mapping[str, float]) -> dict[str, float]:
    return {
        str(label): round(float(value), 6)
        for label, value in sorted(coverages.items())
        if float(value) > 0.0
    }


def multi_hot(labels: Iterable[str], label_vocab: Iterable[str]) -> list[int]:
    label_to_index = {label: index for index, label in enumerate(label_vocab)}
    vector = [0] * len(label_to_index)
    for label in labels:
        vector[label_to_index[label]] = 1
    return vector


def combo_name(labels: Iterable[str]) -> str:
    ordered = sorted(set(str(label) for label in labels))
    return ",".join(ordered) if ordered else "<empty>"


def summarize_records(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, int]]:
    per_dataset_split: dict[str, Counter[str]] = {}
    per_label: Counter[str] = Counter()
    per_status: Counter[str] = Counter()
    for row in rows:
        dataset = str(row["source_dataset"])
        split = str(row["source_split"])
        per_dataset_split.setdefault(dataset, Counter())[split] += 1
        per_label.update(str(label) for label in row.get("label_names", []))
        per_status[str(row["record_status"])] += 1
    normalized_splits = {
        dataset: dict(sorted(counter.items()))
        for dataset, counter in sorted(per_dataset_split.items())
    }
    return normalized_splits, dict(sorted(per_label.items())), dict(sorted(per_status.items()))


def bucket_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[f"{row['source_dataset']}:{row['selection_bucket']}"] += 1
    return dict(sorted(counts.items()))


def dataset_split_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    per_dataset: dict[str, Counter[str]] = {}
    for row in rows:
        per_dataset.setdefault(str(row["source_dataset"]), Counter())[str(row["source_split"])] += 1
    return {dataset: dict(sorted(counter.items())) for dataset, counter in sorted(per_dataset.items())}


def replace_dataset_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    replacements: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    updated = [dict(row) for row in rows if str(row["source_dataset"]) != dataset]
    updated.extend(dict(row) for row in replacements)
    updated.sort(
        key=lambda row: (
            str(row["source_dataset"]),
            str(row["source_split"]),
            str(row["source_sample_id"]),
            int(row.get("patch_y") or 0),
            int(row.get("patch_x") or 0),
        )
    )
    return updated


def roads_group_rows(rows: Sequence[Mapping[str, Any]], *, group_size: int = 4) -> list[list[dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row["source_split"]), []).append(dict(row))

    grouped: list[list[dict[str, Any]]] = []
    for split, split_rows in sorted(by_split.items()):
        ordered = sorted(split_rows, key=lambda row: str(row["source_sample_id"]))
        if len(ordered) % group_size != 0:
            raise ValueError(f"roads split {split} has {len(ordered)} rows, not divisible by {group_size}")
        for start in range(0, len(ordered), group_size):
            grouped.append(ordered[start : start + group_size])
    return grouped


def mosaic_roads_images(images: Sequence[np.ndarray]) -> np.ndarray:
    if len(images) != 4:
        raise ValueError(f"Expected four road tiles, got {len(images)}")
    normalized = [np.asarray(image) for image in images]
    for image in normalized:
        if image.shape != (128, 128, 10):
            raise ValueError(f"Unexpected roads tile shape: {image.shape}")
    top = np.concatenate([normalized[0], normalized[1]], axis=1)
    bottom = np.concatenate([normalized[2], normalized[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def roads_native_coverage(rows: Sequence[Mapping[str, Any]]) -> float:
    coverages = [float((row.get("label_coverages") or {}).get("road_present") or 0.0) for row in rows]
    if not coverages:
        return 0.0
    return float(sum(coverages) / len(coverages))


def _roads_cloud_mask(reflectance: np.ndarray) -> np.ndarray:
    blue = reflectance[..., 0]
    green = reflectance[..., 1]
    red = reflectance[..., 2]
    visible_mean = (blue + green + red) / 3.0
    whiteness = np.maximum.reduce(
        [
            np.abs(blue - green),
            np.abs(green - red),
            np.abs(blue - red),
        ]
    )
    return (visible_mean >= 0.28) & (blue >= 0.22) & (whiteness <= 0.08)


def _roads_water_mask(reflectance: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    green = reflectance[..., 1]
    nir = reflectance[..., 3]
    ndwi = (green - nir) / (green + nir + 1e-6)
    return (ndwi >= 0.1) & (~cloud_mask)


def derive_roads_weak_payload(image: np.ndarray) -> tuple[list[str], dict[str, float]]:
    if image.shape != (256, 256, 10):
        raise ValueError(f"Unexpected roads mosaic shape: {image.shape}")
    reflectance = image.astype(np.float32, copy=False) / 10000.0
    cloud_mask = _roads_cloud_mask(reflectance)
    water_mask = _roads_water_mask(reflectance, cloud_mask)
    land_mask = ~(cloud_mask | water_mask)
    coverages = {
        "cloud": float(np.mean(cloud_mask)),
        "land": float(np.mean(land_mask)),
        "water": float(np.mean(water_mask)),
    }
    labels = sorted(
        label
        for label, coverage in coverages.items()
        if coverage >= threshold_for_label(label)
    )
    return labels, sanitize_coverages(coverages)


def roads_record_status(road_fraction: float) -> str:
    if road_fraction >= threshold_for_label("road_present"):
        return "positive"
    if road_fraction <= 0.0:
        return "explicit_negative"
    return "below_threshold"


def build_roads_mosaic_record(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_path: str,
    label_vocab: Iterable[str],
    mosaic_index: int,
    weak_label_names: Sequence[str],
    weak_coverages: Mapping[str, float],
    road_fraction: float | None = None,
    moe_split: str | None = None,
) -> dict[str, Any]:
    if len(rows) != 4:
        raise ValueError(f"Expected four road rows for mosaic record, got {len(rows)}")
    source_split = str(rows[0]["source_split"])
    sample_id = f"500shot_{'train' if source_split == 'train' else 'val'}_mosaic_{int(mosaic_index):06d}"
    road_fraction = roads_native_coverage(rows) if road_fraction is None else float(road_fraction)
    native_labels = ["road_present"] if road_fraction >= threshold_for_label("road_present") else []
    label_names = sorted(set(native_labels) | set(str(label) for label in weak_label_names))
    coverages = {"road_present": round(float(road_fraction), 6)}
    coverages.update(dict(weak_coverages))
    dataset_path_resolved = str(Path(dataset_path).resolve())
    record: dict[str, Any] = {
        "dataset_path": dataset_path_resolved,
        "image_ref": f"{dataset_path_resolved}::downstream_datasets_nshot/500_shot_roads/{sample_id}::mosaic:0:0:256:256",
        "label_coverages": sanitize_coverages(coverages),
        "label_names": label_names,
        "label_source": "native+heuristic_weak",
        "labels": multi_hot(label_names, label_vocab),
        "native_label_names": native_labels,
        "patch_height": 256,
        "patch_width": 256,
        "patch_x": 0,
        "patch_y": 0,
        "record_status": roads_record_status(road_fraction),
        "selection_bucket": combo_name(label_names),
        "source_dataset": "roads",
        "source_sample_id": sample_id,
        "source_split": source_split,
        "source_storage_group": "500_shot_roads_mosaic",
        "weak_label_names": sorted(set(str(label) for label in weak_label_names)),
    }
    if moe_split is not None:
        record["moe_split"] = str(moe_split)
    return record


def anomaly_tile_origins(height: int, width: int, target_size: int) -> list[tuple[int, int]]:
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    y_starts = list(range(0, max(height - target_size, 0) + 1, target_size))
    x_starts = list(range(0, max(width - target_size, 0) + 1, target_size))
    if not y_starts:
        y_starts = [0]
    if not x_starts:
        x_starts = [0]
    if y_starts[-1] != max(height - target_size, 0):
        y_starts.append(max(height - target_size, 0))
    if x_starts[-1] != max(width - target_size, 0):
        x_starts.append(max(width - target_size, 0))
    return [(y, x) for y in y_starts for x in x_starts]


def derive_anomaly_payload(label_patch: np.ndarray) -> tuple[list[str], dict[str, float], str]:
    patch = np.asarray(label_patch)
    if patch.ndim == 3 and patch.shape[0] == 1:
        patch = patch[0]
    if patch.ndim != 2:
        raise ValueError(f"Unexpected anomaly label patch shape: {patch.shape}")
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return [], {}, "below_threshold"
    rounded = np.rint(finite)
    if not np.allclose(finite, rounded, atol=1e-6):
        raise ValueError("Anomaly labels must be integral class ids")
    counts: Counter[int] = Counter(int(value) for value in rounded.astype(np.int32).reshape(-1))
    valid_total = sum(count for raw_class, count in counts.items() if raw_class != 0)
    if valid_total <= 0:
        return [], {}, "below_threshold"
    coverages = {
        label_name: counts.get(raw_class, 0) / valid_total
        for raw_class, label_name in ANOMALY_LABEL_MAPPING.items()
    }
    label_names = sorted(
        label
        for label, coverage in coverages.items()
        if coverage >= threshold_for_label(label)
    )
    record_status = "positive" if label_names else "below_threshold"
    return label_names, sanitize_coverages(coverages), record_status


def build_anomaly_tile_record(
    *,
    dataset_path: str,
    source_split: str,
    source_sample_id: str,
    source_storage_group: str,
    label_names: Sequence[str],
    label_coverages: Mapping[str, float],
    label_vocab: Iterable[str],
    patch_y: int,
    patch_x: int,
    record_status: str,
    moe_split: str | None = None,
) -> dict[str, Any]:
    dataset_path_resolved = str(Path(dataset_path).resolve())
    record: dict[str, Any] = {
        "dataset_path": dataset_path_resolved,
        "image_ref": (
            f"{dataset_path_resolved}::{source_storage_group}/{source_sample_id}/img::"
            f"patch:{int(patch_y)}:{int(patch_x)}:256:256"
        ),
        "label_coverages": dict(label_coverages),
        "label_names": sorted(str(label) for label in label_names),
        "label_source": "native",
        "labels": multi_hot(label_names, label_vocab),
        "native_label_names": sorted(str(label) for label in label_names),
        "patch_height": 256,
        "patch_width": 256,
        "patch_x": int(patch_x),
        "patch_y": int(patch_y),
        "record_status": str(record_status),
        "selection_bucket": combo_name(label_names),
        "source_dataset": "anomaly_detection",
        "source_sample_id": str(source_sample_id),
        "source_split": str(source_split),
        "source_storage_group": str(source_storage_group),
        "weak_label_names": [],
    }
    if moe_split is not None:
        record["moe_split"] = str(moe_split)
    return record
