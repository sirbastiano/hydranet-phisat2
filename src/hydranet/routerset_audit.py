"""Audit helpers for materialized routerset dataset exports."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

EXPECTED_TILE_SHAPE = (8, 256, 256)
RGB_CHANNELS = (2, 1, 0)
FALSE_RGB_CHANNELS = (4, 2, 1)
STATUS_ERROR = "error"
STATUS_WARN = "warn"
STATUS_OK = "ok"
PADDING_HEAVY_DATASETS = frozenset({"burned_area", "lc", "roads"})


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _save_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows)
    path.write_text((payload + "\n") if payload else "", encoding="utf-8")
    return path


def _materialized_rows(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest_256.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing materialized manifest: {manifest_path}")
    return _load_jsonl(manifest_path)


def _fault_rows(dataset_root: Path) -> list[dict[str, Any]]:
    return _load_jsonl(dataset_root / "fault_rows_256.jsonl")


def _resolve_dataset_tile_path(dataset_root: Path, raw_path: str | Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()

    parts = path.parts
    if dataset_root.name in parts:
        dataset_index = parts.index(dataset_root.name)
        candidate = dataset_root / Path(*parts[dataset_index + 1 :])
        if candidate.exists():
            return candidate.resolve()

    if "images" in parts:
        images_index = parts.index("images")
        candidate = dataset_root / Path(*parts[images_index:])
        if candidate.exists():
            return candidate.resolve()

    return path.resolve()


def _finite_values(view: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(view)
    if finite_mask.all():
        return view.reshape(-1)
    if not finite_mask.any():
        return np.zeros((0,), dtype=np.float32)
    return view[finite_mask]


def _tile_stats(view: np.ndarray) -> dict[str, Any]:
    finite = _finite_values(view)
    non_finite_count = int(view.size - finite.size)
    if finite.size == 0:
        min_value = max_value = mean_value = std_value = 0.0
    else:
        min_value = float(finite.min())
        max_value = float(finite.max())
        mean_value = float(finite.mean())
        std_value = float(finite.std())
    zero_fraction = float(np.count_nonzero(view == 0) / view.size) if view.size > 0 else 0.0
    channel_means = [float(view[index].mean()) for index in range(min(view.shape[0], EXPECTED_TILE_SHAPE[0]))]
    return {
        "shape": list(view.shape),
        "dtype": str(view.dtype),
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "std": std_value,
        "zero_fraction": zero_fraction,
        "non_finite_count": non_finite_count,
        "channel_means": channel_means,
        "all_zero": bool(min_value == 0.0 and max_value == 0.0),
    }


def _row_split(row: Mapping[str, Any]) -> str:
    return str(row.get("moe_split") or row.get("source_split") or "")


def _row_tile_record(row: Mapping[str, Any], stats: Mapping[str, Any]) -> dict[str, Any]:
    issue_codes: list[str] = []
    note_codes: list[str] = []

    shape_tuple = tuple(stats["shape"])
    if shape_tuple != EXPECTED_TILE_SHAPE:
        issue_codes.append("shape_mismatch")
    if stats["dtype"] != "float32":
        issue_codes.append("dtype_mismatch")
    if int(stats["non_finite_count"]) > 0:
        issue_codes.append("non_finite_materialized_tile")
    if bool(stats["all_zero"]):
        issue_codes.append("all_zero_materialized_tile")

    if (
        str(row["source_dataset"]) in PADDING_HEAVY_DATASETS
        and float(stats["zero_fraction"]) >= 0.75
        and "all_zero_materialized_tile" not in issue_codes
    ):
        note_codes.append("padding_heavy_expected")

    if str(row.get("source_split", "")) != _row_split(row):
        note_codes.append("split_reassigned")

    if issue_codes:
        status = STATUS_ERROR
    elif note_codes:
        status = STATUS_WARN
    else:
        status = STATUS_OK

    return {
        "materialized_image_path": str(row["materialized_image_path"]),
        "source_dataset": str(row["source_dataset"]),
        "source_split": str(row.get("source_split", "")),
        "dataset_split": _row_split(row),
        "source_sample_id": str(row["source_sample_id"]),
        "record_status": str(row.get("record_status", "")),
        "selection_bucket": str(row.get("selection_bucket", "")),
        "patch_x": int(row.get("patch_x") or 0),
        "patch_y": int(row.get("patch_y") or 0),
        "patch_width": int(row.get("patch_width") or 0),
        "patch_height": int(row.get("patch_height") or 0),
        "status": status,
        "issue_codes": issue_codes,
        "note_codes": note_codes,
        "shape": list(stats["shape"]),
        "dtype": str(stats["dtype"]),
        "min": float(stats["min"]),
        "max": float(stats["max"]),
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "zero_fraction": float(stats["zero_fraction"]),
        "non_finite_count": int(stats["non_finite_count"]),
        "channel_means": list(stats["channel_means"]),
        "all_zero": bool(stats["all_zero"]),
    }


def _normalize_display(image: np.ndarray, *, channels: Sequence[int]) -> np.ndarray:
    stacked = np.stack([image[index] for index in channels], axis=-1).astype(np.float32, copy=False)
    out = np.zeros_like(stacked, dtype=np.float32)
    for channel_index in range(stacked.shape[-1]):
        channel = stacked[..., channel_index]
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, [2, 98])
        if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
            out[..., channel_index] = 0.0
            continue
        clipped = np.clip(channel, low, high)
        out[..., channel_index] = (clipped - low) / (high - low)
    return np.clip(out, 0.0, 1.0)


def _representative_rows(tile_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in tile_rows:
        by_dataset[str(row["source_dataset"])].append(row)

    selected: list[dict[str, Any]] = []
    for dataset in sorted(by_dataset):
        candidates = [row for row in by_dataset[dataset] if row["status"] != STATUS_ERROR]
        if not candidates:
            candidates = list(by_dataset[dataset])
        zero_values = sorted(float(row["zero_fraction"]) for row in candidates)
        median_zero = zero_values[len(zero_values) // 2]
        winner = min(candidates, key=lambda row: abs(float(row["zero_fraction"]) - median_zero))
        selected.append(dict(winner))
    return selected


def _reconstruct_fault_array(row: Mapping[str, Any], *, target_size: int = 256, target_channels: int = 8) -> np.ndarray:
    import torch

    from .moe_training import build_routerset_training_tensor, normalize_routerset_array

    source_path = Path(str(row["materialized_from"]))
    source_array = np.load(source_path, mmap_mode="r")
    source_dataset = str(row["source_dataset"])
    patch_x = int(row.get("patch_x") or 0)
    patch_y = int(row.get("patch_y") or 0)

    if source_dataset == "anomaly_detection":
        normalized = normalize_routerset_array(
            source_array,
            source_dataset=source_dataset,
            target_channels=target_channels,
        ).numpy()
        tile = normalized[:, patch_y : patch_y + target_size, patch_x : patch_x + target_size]
        if tile.shape != EXPECTED_TILE_SHAPE:
            padded = torch.zeros(EXPECTED_TILE_SHAPE, dtype=torch.float32)
            padded[:, : tile.shape[1], : tile.shape[2]] = torch.from_numpy(tile)
            return padded.numpy()
        return tile

    return build_routerset_training_tensor(
        source_array,
        source_dataset=source_dataset,
        target_size=target_size,
        target_channels=target_channels,
    ).numpy()


def _plot_status_counts(tile_rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    datasets = sorted({str(row["source_dataset"]) for row in tile_rows})
    statuses = [STATUS_OK, STATUS_WARN, STATUS_ERROR]
    counts = {status: [0] * len(datasets) for status in statuses}
    for dataset_index, dataset in enumerate(datasets):
        subset = [row for row in tile_rows if row["source_dataset"] == dataset]
        counter = Counter(str(row["status"]) for row in subset)
        for status in statuses:
            counts[status][dataset_index] = counter.get(status, 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(datasets))
    colors = {STATUS_OK: "#2e8b57", STATUS_WARN: "#d49400", STATUS_ERROR: "#c0392b"}
    for status in statuses:
        values = np.array(counts[status], dtype=float)
        ax.bar(datasets, values, bottom=bottom, label=status, color=colors[status])
        bottom += values
    ax.set_ylabel("Tiles")
    ax.set_title("Tile Audit Status by Expert")
    ax.legend()
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_zero_fraction(tile_rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    datasets = sorted({str(row["source_dataset"]) for row in tile_rows})
    data = [[float(row["zero_fraction"]) for row in tile_rows if row["source_dataset"] == dataset] for dataset in datasets]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(data, tick_labels=datasets, showfliers=False)
    ax.set_ylabel("Zero Fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Zero Fraction by Expert")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_split_reassignments(tile_rows: Sequence[Mapping[str, Any]], path: Path) -> Optional[Path]:
    counter = Counter(str(row["source_dataset"]) for row in tile_rows if "split_reassigned" in row["note_codes"])
    if not counter:
        return None
    datasets = sorted(counter)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(datasets, [counter[dataset] for dataset in datasets], color="#34495e")
    ax.set_ylabel("Rows")
    ax.set_title("Rows Reassigned Across Splits")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_tile_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    path: Path,
    load_array,
    title: str,
) -> Optional[Path]:
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, max(3, len(rows) * 2.8)))
    if len(rows) == 1:
        axes = np.array([axes])
    for index, row in enumerate(rows):
        image = load_array(row)
        rgb = _normalize_display(image, channels=RGB_CHANNELS)
        false_rgb = _normalize_display(image, channels=FALSE_RGB_CHANNELS)
        left, right = axes[index]
        left.imshow(rgb)
        left.set_title(f"{row['source_dataset']} RGB")
        right.imshow(false_rgb)
        right.set_title(f"{row['source_dataset']} False RGB")
        label = f"{row['source_sample_id']} [{row.get('dataset_split', row.get('source_split', ''))}]"
        left.set_ylabel(label)
        left.axis("off")
        right.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def audit_materialized_routerset_dataset(
    dataset_root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    audit_root = Path(output_dir).resolve() if output_dir is not None else (root / "audit").resolve()
    audit_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = _materialized_rows(root)
    fault_rows = _fault_rows(root)

    manifest_paths = [_resolve_dataset_tile_path(root, row["materialized_image_path"]) for row in manifest_rows]
    duplicate_refs = {str(path): count for path, count in Counter(manifest_paths).items() if count > 1}
    disk_paths = sorted(path.resolve() for path in (root / "images").rglob("*.npy"))
    disk_path_set = set(disk_paths)
    missing_paths = sorted(str(path) for path in manifest_paths if path not in disk_path_set)
    unreferenced_paths = sorted(str(path) for path in disk_path_set if path not in set(manifest_paths))

    tile_rows: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    note_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    by_dataset_split: dict[tuple[str, str], int] = defaultdict(int)

    for row in manifest_rows:
        path = _resolve_dataset_tile_path(root, row["materialized_image_path"])
        if not path.exists():
            record = {
                "materialized_image_path": str(path),
                "resolved_materialized_image_path": str(path),
                "source_dataset": str(row["source_dataset"]),
                "source_split": str(row.get("source_split", "")),
                "dataset_split": _row_split(row),
                "source_sample_id": str(row["source_sample_id"]),
                "record_status": str(row.get("record_status", "")),
                "selection_bucket": str(row.get("selection_bucket", "")),
                "patch_x": int(row.get("patch_x") or 0),
                "patch_y": int(row.get("patch_y") or 0),
                "patch_width": int(row.get("patch_width") or 0),
                "patch_height": int(row.get("patch_height") or 0),
                "status": STATUS_ERROR,
                "issue_codes": ["missing_materialized_file"],
                "note_codes": [],
                "shape": [],
                "dtype": "",
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "zero_fraction": 0.0,
                "non_finite_count": 0,
                "channel_means": [],
                "all_zero": False,
            }
        else:
            view = np.asarray(np.load(path, mmap_mode="r"))
            stats = _tile_stats(view)
            record = _row_tile_record(row, stats)
            record["resolved_materialized_image_path"] = str(path)
        tile_rows.append(record)
        status_counts[record["status"]] += 1
        by_dataset_split[(record["source_dataset"], record["dataset_split"])] += 1
        for code in record["issue_codes"]:
            issue_counts[code] += 1
        for code in record["note_codes"]:
            note_counts[code] += 1

    tile_audit_path = _save_jsonl(audit_root / "tile_audit.jsonl", tile_rows)

    representative_rows = _representative_rows(tile_rows)
    representative_plot = _plot_tile_grid(
        representative_rows,
        path=audit_root / "representative_rgb_false_rgb.png",
        load_array=lambda row: np.load(
            row.get("resolved_materialized_image_path") or row["materialized_image_path"],
            mmap_mode="r",
        ),
        title="Representative Tiles: RGB and False RGB",
    )

    fault_preview_rows: list[dict[str, Any]] = []
    if fault_rows:
        grouped_faults: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in fault_rows:
            grouped_faults[str(row["source_dataset"])].append(dict(row))
        for dataset in sorted(grouped_faults):
            fault_preview_rows.extend(grouped_faults[dataset][:3])
    fault_plot = _plot_tile_grid(
        fault_preview_rows,
        path=audit_root / "fault_rgb_false_rgb.png",
        load_array=lambda row: _reconstruct_fault_array(row),
        title="Quarantined Fault Tiles: RGB and False RGB",
    )

    status_plot = _plot_status_counts(tile_rows, audit_root / "status_counts.png")
    zero_fraction_plot = _plot_zero_fraction(tile_rows, audit_root / "zero_fraction_by_expert.png")
    split_reassignment_plot = _plot_split_reassignments(tile_rows, audit_root / "split_reassignments.png")

    per_dataset = {}
    for dataset in sorted({str(row["source_dataset"]) for row in tile_rows}):
        subset = [row for row in tile_rows if row["source_dataset"] == dataset]
        zeros = [float(row["zero_fraction"]) for row in subset]
        per_dataset[dataset] = {
            "count": len(subset),
            "status_counts": dict(Counter(str(row["status"]) for row in subset)),
            "issue_counts": dict(Counter(code for row in subset for code in row["issue_codes"])),
            "note_counts": dict(Counter(code for row in subset for code in row["note_codes"])),
            "zero_fraction_min": float(min(zeros)) if zeros else 0.0,
            "zero_fraction_median": float(np.median(zeros)) if zeros else 0.0,
            "zero_fraction_max": float(max(zeros)) if zeros else 0.0,
        }

    summary = {
        "dataset_root": str(root),
        "audit_root": str(audit_root),
        "manifest_path": str(root / "manifest_256.jsonl"),
        "fault_rows_path": str(root / "fault_rows_256.jsonl"),
        "tile_audit_path": str(tile_audit_path),
        "manifest_row_count": len(manifest_rows),
        "disk_file_count": len(disk_paths),
        "tile_audit_row_count": len(tile_rows),
        "fault_row_count": len(fault_rows),
        "duplicate_manifest_refs": duplicate_refs,
        "missing_paths": missing_paths,
        "unreferenced_paths": unreferenced_paths,
        "status_counts": dict(status_counts),
        "issue_counts": dict(issue_counts),
        "note_counts": dict(note_counts),
        "per_dataset": per_dataset,
        "rows_by_dataset_split": {f"{dataset}:{split}": count for (dataset, split), count in sorted(by_dataset_split.items())},
        "plot_paths": {
            "status_counts": str(status_plot),
            "zero_fraction_by_expert": str(zero_fraction_plot),
            "representative_rgb_false_rgb": str(representative_plot) if representative_plot else "",
            "fault_rgb_false_rgb": str(fault_plot) if fault_plot else "",
            "split_reassignments": str(split_reassignment_plot) if split_reassignment_plot else "",
        },
    }
    _save_json(audit_root / "audit_summary.json", summary)
    return summary
