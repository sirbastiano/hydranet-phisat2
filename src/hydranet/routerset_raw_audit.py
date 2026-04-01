"""Audit helpers for raw routerset dataset snapshots."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_ERROR = "error"
MAX_AUDIT_SAMPLE_ELEMENTS = 8_192
ROUTERSET_SWAPPED_TILE_DATASETS = frozenset({"burned_area", "worldfloods"})
RAW_DATASET_CONTRACTS: dict[str, dict[str, Any]] = {
    "fire": {"shape": (8, 256, 256), "dtype": "float32", "layout": "chw"},
    "burned_area": {"shape": (7, 256, 256), "dtype": "float32", "layout": "chw"},
    "anomaly_detection": {"shape": (8, 256, 256), "dtype": "float32", "layout": "chw"},
    "worldfloods": {"shape": (8, 256, 256), "dtype": "float32", "layout": "chw"},
    "lc": {"shape": (128, 128, 10), "dtype": "uint16", "layout": "hwc"},
    "roads": {"shape": (256, 256, 10), "dtype": "uint16", "layout": "hwc"},
}
RGB_CHANNELS = (2, 1, 0)
FALSE_RGB_CHANNELS = (4, 2, 1)


def _save_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _save_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows)
    path.write_text((payload + "\n") if payload else "", encoding="utf-8")
    return path


def resolve_raw_routerset_root(routerset_dir: str | Path) -> Path:
    path = Path(routerset_dir)
    if (path / "manifest.jsonl").exists() and (path / "images").exists():
        return path
    candidate = path / "multilabel_dataset"
    if (candidate / "manifest.jsonl").exists() and (candidate / "images").exists():
        return candidate
    raise FileNotFoundError(f"Routerset dataset root not found under {path}")


def load_raw_routerset_manifest_rows(routerset_dir: str | Path) -> list[dict[str, Any]]:
    root = resolve_raw_routerset_root(routerset_dir)
    manifest_path = root / "manifest.jsonl"
    return [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_raw_routerset_image_inventory(routerset_dir: str | Path) -> set[tuple[str, str, str]]:
    root = resolve_raw_routerset_root(routerset_dir)
    inventory: set[tuple[str, str, str]] = set()
    for path in (root / "images").glob("*/*/*.npy"):
        inventory.add((path.parent.parent.name, path.parent.name, path.name))
    return inventory


def raw_routerset_patch_token(row: Mapping[str, Any]) -> str:
    width = row["patch_width"] if row["patch_width"] is not None else "full"
    height = row["patch_height"] if row["patch_height"] is not None else "full"
    return f"{row['source_sample_id']}_{row['patch_x']}_{row['patch_y']}_{height}_{width}.npy"


def raw_routerset_compatibility_patch_token(row: Mapping[str, Any]) -> str | None:
    width = row.get("patch_width")
    height = row.get("patch_height")
    if width is None or height is None:
        return None
    if row["source_dataset"] not in ROUTERSET_SWAPPED_TILE_DATASETS:
        return None
    return f"{row['source_sample_id']}_{row['patch_y']}_{row['patch_x']}_{height}_{width}.npy"


def resolve_raw_routerset_image_path(
    routerset_dir: str | Path,
    row: Mapping[str, Any],
    *,
    inventory: set[tuple[str, str, str]] | None = None,
) -> Path:
    root = resolve_raw_routerset_root(routerset_dir)
    source_dataset = str(row["source_dataset"])
    source_split = str(row["source_split"])
    image_dir = root / "images" / source_dataset / source_split
    primary = image_dir / raw_routerset_patch_token(row)
    if inventory is None:
        if primary.exists():
            return primary
    elif (source_dataset, source_split, primary.name) in inventory:
        return primary
    compatibility_filename = raw_routerset_compatibility_patch_token(row)
    if compatibility_filename is not None:
        compatibility_path = image_dir / compatibility_filename
        if inventory is None:
            if compatibility_path.exists():
                return compatibility_path
        elif (source_dataset, source_split, compatibility_path.name) in inventory:
            return compatibility_path
    return primary


def _sample_view(view: np.ndarray, *, max_elements: int = MAX_AUDIT_SAMPLE_ELEMENTS) -> tuple[np.ndarray, bool]:
    if view.size <= max_elements:
        return view.reshape(-1), False
    step = max(1, math.ceil(view.size / max_elements))
    return view.reshape(-1)[::step], True


def _finite_values(view: np.ndarray) -> tuple[np.ndarray, int, bool]:
    sampled, used_sampling = _sample_view(view)
    finite_mask = np.isfinite(sampled)
    non_finite_count = int(sampled.size - finite_mask.sum())
    if finite_mask.all():
        return sampled.reshape(-1), non_finite_count, used_sampling
    if not finite_mask.any():
        return np.zeros((0,), dtype=np.float32), non_finite_count, used_sampling
    return sampled[finite_mask], non_finite_count, used_sampling


def infer_array_layout(view: np.ndarray) -> str:
    if view.ndim != 3:
        return "other"
    if view.shape[0] <= 16 and view.shape[1] > 16 and view.shape[2] > 16:
        return "chw"
    if view.shape[2] <= 16 and view.shape[0] > 16 and view.shape[1] > 16:
        return "hwc"
    return "other"


def to_chw(view: np.ndarray) -> np.ndarray:
    layout = infer_array_layout(view)
    if layout == "chw":
        return view
    if layout == "hwc":
        return np.moveaxis(view, -1, 0)
    raise ValueError(f"Unsupported array layout for shape {view.shape}")


def display_view(view: np.ndarray, *, max_size: int = 512) -> np.ndarray:
    chw = to_chw(np.asarray(view))
    height = chw.shape[1]
    width = chw.shape[2]
    step = max(1, math.ceil(max(height, width) / max_size))
    if step > 1:
        chw = chw[:, ::step, ::step]
    return chw


def normalize_display(chw: np.ndarray, *, channels: Sequence[int]) -> np.ndarray:
    safe_channels = tuple(min(index, chw.shape[0] - 1) for index in channels)
    rgb = np.stack([chw[index] for index in safe_channels], axis=-1).astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)
    for index in range(rgb.shape[-1]):
        channel = rgb[..., index]
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, [2, 98])
        if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
            continue
        clipped = np.clip(channel, low, high)
        out[..., index] = (clipped - low) / (high - low)
    return np.clip(out, 0.0, 1.0)


def array_stats(view: np.ndarray) -> dict[str, Any]:
    finite, non_finite_count, used_sampling = _finite_values(view)
    if finite.size == 0:
        min_value = max_value = mean_value = std_value = 0.0
    else:
        min_value = float(finite.min())
        max_value = float(finite.max())
        mean_value = float(finite.mean())
        std_value = float(finite.std())
    if used_sampling:
        sampled, _ = _sample_view(view)
        zero_fraction = float(np.count_nonzero(sampled == 0) / sampled.size) if sampled.size else 0.0
    else:
        zero_fraction = float(np.count_nonzero(view == 0) / view.size) if view.size else 0.0
    return {
        "shape": list(view.shape),
        "dtype": str(view.dtype),
        "layout": infer_array_layout(view),
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "std": std_value,
        "zero_fraction": zero_fraction,
        "non_finite_count": non_finite_count,
        "all_zero": bool(min_value == 0.0 and max_value == 0.0),
        "stats_sampled": used_sampling,
    }


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("source_dataset", "")),
        str(row.get("source_split", "")),
        str(row.get("source_sample_id", "")),
        int(row.get("patch_x") or 0),
        int(row.get("patch_y") or 0),
        str(row.get("patch_width")),
        str(row.get("patch_height")),
    )


def audit_row_record(
    routerset_dir: str | Path,
    row: Mapping[str, Any],
    *,
    inventory: set[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    path = resolve_raw_routerset_image_path(routerset_dir, row, inventory=inventory)
    record: dict[str, Any] = {
        "source_dataset": str(row["source_dataset"]),
        "source_split": str(row["source_split"]),
        "source_sample_id": str(row["source_sample_id"]),
        "record_status": str(row.get("record_status", "")),
        "selection_bucket": str(row.get("selection_bucket", "")),
        "label_names": list(row.get("label_names") or []),
        "native_label_names": list(row.get("native_label_names") or []),
        "weak_label_names": list(row.get("weak_label_names") or []),
        "label_source": str(row.get("label_source", "")),
        "label_coverages": dict(row.get("label_coverages") or {}),
        "source_storage_group": str(row.get("source_storage_group", "")),
        "patch_x": int(row.get("patch_x") or 0),
        "patch_y": int(row.get("patch_y") or 0),
        "patch_width": row.get("patch_width"),
        "patch_height": row.get("patch_height"),
        "image_path": str(path),
        "status": STATUS_OK,
        "issue_codes": [],
        "note_codes": [],
    }
    if not path.exists():
        record["status"] = STATUS_ERROR
        record["issue_codes"] = ["missing_file"]
    return record


def enrich_sample_row(row: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(row)
    path = Path(str(record["image_path"]))
    if not path.exists():
        return record

    view = np.load(path, mmap_mode="r")
    stats = array_stats(view)
    contract = RAW_DATASET_CONTRACTS.get(str(record["source_dataset"]))
    issue_codes = list(record.get("issue_codes", []))
    note_codes = list(record.get("note_codes", []))
    if contract is not None:
        if tuple(stats["shape"]) != tuple(contract["shape"]):
            issue_codes.append("shape_mismatch")
        if stats["dtype"] != str(contract["dtype"]):
            issue_codes.append("dtype_mismatch")
        if stats["layout"] != str(contract["layout"]):
            issue_codes.append("layout_mismatch")
    if int(stats["non_finite_count"]) > 0:
        issue_codes.append("non_finite_values")
    if bool(stats["all_zero"]):
        issue_codes.append("all_zero_array")
    if bool(stats["stats_sampled"]):
        note_codes.append("sampled_large_array_stats")
    if str(record["source_dataset"]) == "anomaly_detection" and max(int(stats["shape"][1]), int(stats["shape"][2])) > 256:
        note_codes.append("large_source_scene")
    if str(record["source_dataset"]) in {"roads", "lc"}:
        note_codes.append("raw_uint16_reflectance")

    record.update(stats)
    record["issue_codes"] = issue_codes
    record["note_codes"] = note_codes
    if issue_codes:
        record["status"] = STATUS_ERROR
    return record


def select_sample_rows(file_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in file_rows:
        grouped[(str(row["source_dataset"]), str(row["source_split"]))].append(row)

    def richness(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
        label_names = list(row.get("label_names") or [])
        weak_labels = list(row.get("weak_label_names") or [])
        coverages = row.get("label_coverages") or {}
        max_coverage = max((float(value) for value in coverages.values()), default=0.0)
        return (
            float(len(label_names)),
            float(len(weak_labels)),
            max_coverage,
            str(row.get("source_sample_id", "")),
        )

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=_row_sort_key)
        preferred = [row for row in rows if row["status"] != STATUS_ERROR]
        if not preferred:
            preferred = rows
        positive = [row for row in preferred if row.get("label_names")]
        candidate_pool = positive if positive else preferred
        chosen = max(candidate_pool, key=richness)
        selected.append(dict(chosen))
    return selected


def _plot_dataset_split_counts(file_rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    datasets = sorted({str(row["source_dataset"]) for row in file_rows})
    split_names = sorted({str(row["source_split"]) for row in file_rows})
    counts = {split: [0] * len(datasets) for split in split_names}
    for index, dataset in enumerate(datasets):
        subset = [row for row in file_rows if row["source_dataset"] == dataset]
        counter = Counter(str(row["source_split"]) for row in subset)
        for split in split_names:
            counts[split][index] = counter.get(split, 0)
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(datasets))
    for split in split_names:
        values = np.array(counts[split], dtype=float)
        ax.bar(datasets, values, bottom=bottom, label=split)
        bottom += values
    ax.set_ylabel("rows")
    ax.set_title("Raw Routerset Rows by Dataset and Split")
    ax.legend(loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_status_counts(file_rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    datasets = sorted({str(row["source_dataset"]) for row in file_rows})
    statuses = [STATUS_OK, STATUS_WARN, STATUS_ERROR]
    counts = {status: [0] * len(datasets) for status in statuses}
    colors = {STATUS_OK: "#2e8b57", STATUS_WARN: "#d49400", STATUS_ERROR: "#c0392b"}
    for index, dataset in enumerate(datasets):
        subset = [row for row in file_rows if row["source_dataset"] == dataset]
        counter = Counter(str(row["status"]) for row in subset)
        for status in statuses:
            counts[status][index] = counter.get(status, 0)
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(datasets))
    for status in statuses:
        values = np.array(counts[status], dtype=float)
        ax.bar(datasets, values, bottom=bottom, label=status, color=colors[status])
        bottom += values
    ax.set_ylabel("rows")
    ax.set_title("Raw Routerset Audit Status by Dataset")
    ax.legend(loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_zero_fraction_boxplot(file_rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    datasets = sorted({str(row["source_dataset"]) for row in file_rows if "zero_fraction" in row})
    series = [
        [float(row["zero_fraction"]) for row in file_rows if row["source_dataset"] == dataset and "zero_fraction" in row]
        for dataset in datasets
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(series, tick_labels=datasets, showfliers=False)
    ax.set_ylabel("zero fraction")
    ax.set_title("Representative Raw Routerset Zero Fraction by Dataset")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def audit_raw_routerset_dataset(routerset_dir: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    dataset_root = resolve_raw_routerset_root(routerset_dir)
    rows = load_raw_routerset_manifest_rows(dataset_root)
    inventory = build_raw_routerset_image_inventory(dataset_root)
    file_rows = [audit_row_record(dataset_root, row, inventory=inventory) for row in rows]
    sample_rows = [enrich_sample_row(row) for row in select_sample_rows(file_rows)]

    output_root = Path(output_dir) if output_dir is not None else dataset_root / "audit_raw"
    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "manifest_path": str(dataset_root / 'manifest.jsonl'),
        "row_count": len(file_rows),
        "status_counts": dict(Counter(str(row["status"]) for row in file_rows)),
        "issue_counts": dict(Counter(code for row in file_rows for code in row.get("issue_codes", []))),
        "note_counts": dict(Counter(code for row in sample_rows for code in row.get("note_codes", []))),
        "per_dataset": {},
        "sample_rows_path": str(output_root / "sample_rows.json"),
        "file_audit_path": str(output_root / "file_audit.jsonl"),
    }

    for dataset in sorted({str(row["source_dataset"]) for row in file_rows}):
        subset = [row for row in file_rows if row["source_dataset"] == dataset]
        sampled_subset = [row for row in sample_rows if row["source_dataset"] == dataset]
        zero_values = sorted(float(row.get("zero_fraction", 0.0)) for row in sampled_subset if "zero_fraction" in row)
        midpoint = len(zero_values) // 2
        summary["per_dataset"][dataset] = {
            "row_count": len(subset),
            "split_counts": dict(Counter(str(row["source_split"]) for row in subset)),
            "status_counts": dict(Counter(str(row["status"]) for row in subset)),
            "sample_shape_counts": dict(Counter("x".join(str(v) for v in row.get("shape", [])) for row in sample_rows if row["source_dataset"] == dataset and row.get("shape"))),
            "sample_dtype_counts": dict(Counter(str(row.get("dtype", "")) for row in sample_rows if row["source_dataset"] == dataset and row.get("dtype"))),
            "sample_layout_counts": dict(Counter(str(row.get("layout", "")) for row in sample_rows if row["source_dataset"] == dataset and row.get("layout"))),
            "sample_zero_fraction": {
                "min": zero_values[0] if zero_values else 0.0,
                "median": zero_values[midpoint] if zero_values else 0.0,
                "max": zero_values[-1] if zero_values else 0.0,
                "mean": float(sum(zero_values) / len(zero_values)) if zero_values else 0.0,
            },
        }

    _save_json(output_root / "summary.json", summary)
    _save_jsonl(output_root / "file_audit.jsonl", file_rows)
    _save_json(output_root / "sample_rows.json", {"rows": sample_rows})
    _plot_dataset_split_counts(file_rows, output_root / "dataset_split_counts.png")
    _plot_status_counts(file_rows, output_root / "status_counts.png")
    _plot_zero_fraction_boxplot(sample_rows, output_root / "zero_fraction_boxplot.png")
    return summary
