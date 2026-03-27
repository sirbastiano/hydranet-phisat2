from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hydranet.routerset_audit import EXPECTED_TILE_SHAPE, audit_materialized_routerset_dataset


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text((payload + "\n") if payload else "", encoding="utf-8")


def _manifest_row(materialized_image_path: str, *, source_dataset: str = "worldfloods") -> dict[str, object]:
    return {
        "materialized_image_path": materialized_image_path,
        "materialized_from": "routerset/multilabel_dataset/images/worldfloods/train/sample.npy",
        "source_dataset": source_dataset,
        "source_split": "train",
        "moe_split": "train",
        "source_sample_id": "sample",
        "record_status": "selected",
        "selection_bucket": "positive",
        "patch_x": 0,
        "patch_y": 0,
        "patch_width": 256,
        "patch_height": 256,
    }


def test_audit_resolves_dataset_root_prefixed_relative_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "fix27March"
    image_path = dataset_root / "images/worldfloods/train/sample.npy"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(image_path, np.random.default_rng(0).random(EXPECTED_TILE_SHAPE, dtype=np.float32))

    _write_jsonl(
        dataset_root / "manifest_256.jsonl",
        [_manifest_row("outputs/routerset/fix27March/images/worldfloods/train/sample.npy")],
    )
    _write_jsonl(dataset_root / "fault_rows_256.jsonl", [])

    summary = audit_materialized_routerset_dataset(dataset_root)

    assert summary["manifest_row_count"] == 1
    assert summary["status_counts"] == {"ok": 1}
    assert summary["issue_counts"] == {}
    assert summary["missing_paths"] == []
    assert summary["unreferenced_paths"] == []


def test_audit_flags_all_zero_materialized_tile_as_error(tmp_path: Path) -> None:
    dataset_root = tmp_path / "fix27March"
    image_path = dataset_root / "images/worldfloods/train/zero.npy"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(image_path, np.zeros(EXPECTED_TILE_SHAPE, dtype=np.float32))

    _write_jsonl(
        dataset_root / "manifest_256.jsonl",
        [_manifest_row("outputs/routerset/fix27March/images/worldfloods/train/zero.npy")],
    )
    _write_jsonl(dataset_root / "fault_rows_256.jsonl", [])

    summary = audit_materialized_routerset_dataset(dataset_root)

    assert summary["status_counts"] == {"error": 1}
    assert summary["issue_counts"] == {"all_zero_materialized_tile": 1}
