#!/usr/bin/env python3
"""Rebuild routerset roads mosaics and anomaly 256x256 tiles."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zipfile import ZipFile

import numpy as np

from hydranet.routerset_roads_anomaly import (
    anomaly_tile_origins,
    bucket_counts,
    build_anomaly_tile_record,
    build_roads_mosaic_record,
    dataset_split_counts,
    derive_anomaly_payload,
    derive_roads_weak_payload,
    load_jsonl,
    load_label_vocab,
    replace_dataset_rows,
    roads_group_rows,
    mosaic_roads_images,
    summarize_records,
    write_jsonl,
)


DEFAULT_ANOMALY_SOURCE_ROOT = Path(
    "/shared/home/rdelprete/PythonProjects/phi2FM/downloads_data/anomaly_detection.zarr/marine_area_dataset.zarr"
)
DEFAULT_PLOT_SCRIPT = Path(
    "/shared/home/rdelprete/PythonProjects/phi2FM/phisat/phi2FM/downstream/plot_multilabel_dataset.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routerset-dir", default="routerset")
    parser.add_argument("--anomaly-source-root", default=str(DEFAULT_ANOMALY_SOURCE_ROOT))
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--repo-id", default="sirbastiano94/routerset")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def _timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _copy_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def backup_changed_paths(routerset_dir: Path, backup_root: Path) -> None:
    paths = [
        routerset_dir / "manifest.jsonl",
        routerset_dir / "manifest_moe_train.jsonl",
        routerset_dir / "summary.json",
        routerset_dir / "audit.json",
        routerset_dir / "materialization_summary.json",
        routerset_dir / "plots",
        routerset_dir / "multilabel_dataset" / "manifest.jsonl",
        routerset_dir / "multilabel_dataset" / "manifest_moe_train.jsonl",
        routerset_dir / "multilabel_dataset" / "summary.json",
        routerset_dir / "multilabel_dataset" / "audit.json",
        routerset_dir / "multilabel_dataset" / "materialization_summary.json",
        routerset_dir / "multilabel_dataset" / "plots",
    ]
    for path in paths:
        if path.exists():
            relative = path.relative_to(routerset_dir)
            _copy_path(path, backup_root / relative)


def _routerset_image_path(dataset_root: Path, row: dict[str, Any]) -> Path:
    patch_y = row.get("patch_y")
    patch_x = row.get("patch_x")
    patch_height = row.get("patch_height")
    patch_width = row.get("patch_width")
    patch_y_token = int(patch_y or 0)
    patch_x_token = int(patch_x or 0)
    patch_height_token = "full" if patch_height is None else int(patch_height)
    patch_width_token = "full" if patch_width is None else int(patch_width)
    filename = (
        f"{row['source_sample_id']}_{patch_y_token}_{patch_x_token}_"
        f"{patch_height_token}_{patch_width_token}.npy"
    )
    return dataset_root / "images" / str(row["source_dataset"]) / str(row["source_split"]) / filename


def _dataset_image_roots(base_dir: Path, dataset: str) -> list[Path]:
    return [
        base_dir / "multilabel_dataset" / "images" / dataset,
    ]


def _write_dataset_image(base_dir: Path, row: dict[str, Any], image: np.ndarray) -> None:
    filename = (
        f"{row['source_sample_id']}_{int(row.get('patch_y') or 0)}_{int(row.get('patch_x') or 0)}_"
        f"{int(row['patch_height'])}_{int(row['patch_width'])}.npy"
    )
    for root in _dataset_image_roots(base_dir, str(row["source_dataset"])):
        split_dir = root / str(row["source_split"])
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / filename, image)


def _swap_dataset_image_dirs(routerset_dir: Path, temp_root: Path, dataset: str) -> None:
    staged_nested = temp_root / "multilabel_dataset" / "images" / dataset
    final_root = routerset_dir / "images" / dataset
    if final_root.exists() or final_root.is_symlink():
        if final_root.is_symlink() or final_root.is_file():
            final_root.unlink()
        else:
            shutil.rmtree(final_root)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged_nested), str(final_root))


def _extract_zip_member(zip_path: Path, member_name: str, cache_dir: Path) -> Path:
    destination = cache_dir / Path(member_name).name
    if destination.exists():
        return destination
    cache_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zf, zf.open(member_name) as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return destination


def _roads_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row["source_dataset"] == "roads"]


def _anomaly_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row["source_dataset"] == "anomaly_detection"]


def _record_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row["source_dataset"]),
        str(row["source_split"]),
        str(row["source_sample_id"]),
        int(row.get("patch_y") or 0),
        int(row.get("patch_x") or 0),
    )


def build_roads_replacements(
    manifest_rows: list[dict[str, Any]],
    *,
    label_vocab: list[str],
    cache_dir: Path,
    image_output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    grouped = roads_group_rows(_roads_rows(manifest_rows))
    if not grouped:
        return [], [], {"old_rows": 0, "new_rows": 0, "split_counts": {}, "weak_label_counts": {}}, []
    zip_path = Path(str(grouped[0][0]["dataset_path"]))
    replacements_manifest: list[dict[str, Any]] = []
    replacements_moe: list[dict[str, Any]] = []
    stale_paths: list[str] = []
    split_counters: Counter[str] = Counter()
    weak_counter: Counter[str] = Counter()
    split_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for split in ("train", "validation"):
        split_token = "train" if split == "train" else "val"
        image_member = f"downstream_datasets_nshot/500_shot_roads/500shot_{split_token}_s2.npy"
        label_member = f"downstream_datasets_nshot/500_shot_roads/500shot_{split_token}_label_roads.npy"
        image_path = _extract_zip_member(zip_path, image_member, cache_dir)
        label_path = _extract_zip_member(zip_path, label_member, cache_dir)
        split_arrays[split] = (
            np.load(image_path, mmap_mode="r"),
            np.load(label_path, mmap_mode="r"),
        )

    for group in grouped:
        split = str(group[0]["source_split"])
        mosaic_index = split_counters[split]
        split_counters[split] += 1
        split_images, split_labels = split_arrays[split]
        start_index = mosaic_index * 4
        arrays = [split_images[start_index + offset] for offset in range(4)]
        label_arrays = [split_labels[start_index + offset] for offset in range(4)]
        for row in group:
            source_path = Path(
                f"{row['source_sample_id']}_{int(row.get('patch_y') or 0)}_{int(row.get('patch_x') or 0)}_"
                f"{int(row['patch_height'])}_{int(row['patch_width'])}.npy"
            )
            stale_paths.append(f"multilabel_dataset/images/roads/{split}/{source_path.name}")
            stale_paths.append(f"images/roads/{split}/{source_path.name}")
        mosaic = mosaic_roads_images(arrays)
        top = np.concatenate([label_arrays[0], label_arrays[1]], axis=1)
        bottom = np.concatenate([label_arrays[2], label_arrays[3]], axis=1)
        mosaic_label = np.concatenate([top, bottom], axis=0)
        road_fraction = float(np.mean(mosaic_label > 0))
        weak_labels, weak_coverages = derive_roads_weak_payload(mosaic)
        weak_counter.update(weak_labels)
        manifest_record = build_roads_mosaic_record(
            group,
            dataset_path=str(group[0]["dataset_path"]),
            label_vocab=label_vocab,
            mosaic_index=mosaic_index,
            weak_label_names=weak_labels,
            weak_coverages=weak_coverages,
            road_fraction=road_fraction,
        )
        moe_record = build_roads_mosaic_record(
            group,
            dataset_path=str(group[0]["dataset_path"]),
            label_vocab=label_vocab,
            mosaic_index=mosaic_index,
            weak_label_names=weak_labels,
            weak_coverages=weak_coverages,
            road_fraction=road_fraction,
            moe_split=split,
        )
        replacements_manifest.append(manifest_record)
        replacements_moe.append(moe_record)
        _write_dataset_image(image_output_root, manifest_record, mosaic.astype(np.uint16, copy=False))

    summary = {
        "old_rows": len(_roads_rows(manifest_rows)),
        "new_rows": len(replacements_manifest),
        "split_counts": dict(sorted(split_counters.items())),
        "weak_label_counts": dict(sorted(weak_counter.items())),
    }
    return replacements_manifest, replacements_moe, summary, stale_paths


def _build_anomaly_scene_tiles(
    row: dict[str, Any],
    *,
    source_root: Path,
    label_vocab: list[str],
    image_output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str], Counter[str], list[str]]:
    import zarr

    split = str(row["source_split"])
    sample_id = str(row["source_sample_id"])
    storage_group = str(row["source_storage_group"])
    sample_dir = source_root / storage_group / sample_id
    image_ds = zarr.open(str(sample_dir / "img"), mode="r")
    full_image = np.asarray(image_ds[:], dtype=np.float32)
    if full_image.ndim != 3 or int(full_image.shape[0]) != 8:
        raise ValueError(f"Unexpected anomaly image shape for {sample_id}: {tuple(full_image.shape)}")
    height = int(full_image.shape[1])
    width = int(full_image.shape[2])

    label_ds = zarr.open(str(sample_dir / "label"), mode="r")
    full_label = np.asarray(label_ds[:], dtype=np.float32)
    if full_label.ndim == 3 and int(full_label.shape[0]) == 1:
        full_label = full_label[0]
    if full_label.ndim != 2:
        raise ValueError(f"Unexpected anomaly label shape for {sample_id}: {tuple(full_label.shape)}")
    if tuple(full_label.shape) != (height, width):
        raise ValueError(
            f"Anomaly label/image shape mismatch for {sample_id}: image={(height, width)} label={tuple(full_label.shape)}"
        )

    scene_manifest: list[dict[str, Any]] = []
    scene_moe: list[dict[str, Any]] = []
    split_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    stale_name = f"{sample_id}_0_0_full_full.npy"
    stale_paths = [
        f"multilabel_dataset/images/anomaly_detection/{split}/{stale_name}",
        f"images/anomaly_detection/{split}/{stale_name}",
    ]

    for patch_y, patch_x in anomaly_tile_origins(height, width, 256):
        image_tile = np.asarray(full_image[:, patch_y : patch_y + 256, patch_x : patch_x + 256], dtype=np.float32)
        label_tile = full_label[patch_y : patch_y + 256, patch_x : patch_x + 256]
        label_names, label_coverages, record_status = derive_anomaly_payload(label_tile)
        label_counter.update(label_names)
        manifest_record = build_anomaly_tile_record(
            dataset_path=str(row["dataset_path"]),
            source_split=split,
            source_sample_id=sample_id,
            source_storage_group=storage_group,
            label_names=label_names,
            label_coverages=label_coverages,
            label_vocab=label_vocab,
            patch_y=patch_y,
            patch_x=patch_x,
            record_status=record_status,
        )
        moe_record = build_anomaly_tile_record(
            dataset_path=str(row["dataset_path"]),
            source_split=split,
            source_sample_id=sample_id,
            source_storage_group=storage_group,
            label_names=label_names,
            label_coverages=label_coverages,
            label_vocab=label_vocab,
            patch_y=patch_y,
            patch_x=patch_x,
            record_status=record_status,
            moe_split=split,
        )
        scene_manifest.append(manifest_record)
        scene_moe.append(moe_record)
        _write_dataset_image(image_output_root, manifest_record, image_tile)
        split_counter[split] += 1

    return scene_manifest, scene_moe, split_counter, label_counter, stale_paths


def build_anomaly_replacements(
    manifest_rows: list[dict[str, Any]],
    *,
    dataset_root: Path,
    source_root: Path,
    label_vocab: list[str],
    image_output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    replacements_manifest: list[dict[str, Any]] = []
    replacements_moe: list[dict[str, Any]] = []
    stale_paths: list[str] = []
    split_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    scene_rows = sorted(_anomaly_rows(manifest_rows), key=lambda item: str(item["source_sample_id"]))
    scene_counter = len(scene_rows)
    max_workers = min(max(os.cpu_count() or 1, 1), 8, max(scene_counter, 1))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _build_anomaly_scene_tiles,
                row,
                source_root=source_root,
                label_vocab=label_vocab,
                image_output_root=image_output_root,
            )
            for row in scene_rows
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            scene_manifest, scene_moe, scene_split_counter, scene_label_counter, scene_stale_paths = future.result()
            replacements_manifest.extend(scene_manifest)
            replacements_moe.extend(scene_moe)
            split_counter.update(scene_split_counter)
            label_counter.update(scene_label_counter)
            stale_paths.extend(scene_stale_paths)
            print(f"anomaly scenes completed: {index}/{scene_counter}", flush=True)

    replacements_manifest.sort(key=_record_sort_key)
    replacements_moe.sort(key=_record_sort_key)

    summary = {
        "old_rows": len(_anomaly_rows(manifest_rows)),
        "new_rows": len(replacements_manifest),
        "source_scene_count": scene_counter,
        "split_counts": dict(sorted(split_counter.items())),
        "label_counts": dict(sorted(label_counter.items())),
    }
    return replacements_manifest, replacements_moe, summary, stale_paths


def _update_summary(existing: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_dataset_split, per_label, per_status = summarize_records(rows)
    updated = dict(existing)
    updated["num_records"] = len(rows)
    updated["per_dataset_split"] = per_dataset_split
    updated["per_label"] = per_label
    updated["per_status"] = per_status
    sampling_note = dict(updated.get("sampling_note") or {})
    sampling_note["roads"] = {
        "selection": "all native records from the published n-shot subset, mosaiced sequentially into 256x256 2x2 collages",
        "source_archive": "/shared/home/rdelprete/PythonProjects/phi2FM/downloads_data/phileo_nshot/downstream_datasets_nshot.zip",
        "source_subset": "500_shot_roads",
        "weak_labels": "heuristic cloud/land/water tags derived from multispectral reflectance on each 256x256 mosaic",
    }
    sampling_note["anomaly_detection"] = "all checked-in train and validation ids, tiled deterministically into non-overlapping 256x256 patches"
    updated["sampling_note"] = sampling_note
    updated["build_mode"] = (
        "balanced multi-source raw extraction with source-driven 256x256 burned_area scenes, "
        "deterministically tiled 256x256 anomaly_detection patches, tiled worldfloods records, "
        "native lc n-shot archive records, and 256x256 mosaiced roads n-shot records with heuristic weak labels"
    )
    return updated


def _update_audit(existing: dict[str, Any], *, selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    _, _, per_status = summarize_records(selected_rows)
    updated = dict(existing)
    updated["selected_dataset_split"] = dataset_split_counts(selected_rows)
    updated["selected_status_counts"] = per_status
    updated["selected_bucket_counts"] = bucket_counts(selected_rows)
    updated["candidate_dataset_split"] = dataset_split_counts(selected_rows)
    updated["candidate_bucket_counts"] = bucket_counts(selected_rows)
    return updated


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def refresh_metadata(
    routerset_dir: Path,
    manifest_rows: list[dict[str, Any]],
    moe_rows: list[dict[str, Any]],
    rebuild_summary: dict[str, Any],
) -> None:
    dataset_root = routerset_dir / "multilabel_dataset"
    summary_path = dataset_root / "summary.json"
    audit_path = dataset_root / "audit.json"
    materialization_path = dataset_root / "materialization_summary.json"

    summary_payload = _update_summary(json.loads(summary_path.read_text(encoding="utf-8")), manifest_rows)
    audit_payload = _update_audit(
        json.loads(audit_path.read_text(encoding="utf-8")),
        selected_rows=manifest_rows,
    )
    materialization_payload = {
        "records": len(manifest_rows),
        "skipped": 0,
        "written": len(manifest_rows),
    }

    for destination in [routerset_dir / "summary.json", summary_path]:
        _write_json(destination, summary_payload)
    for destination in [routerset_dir / "audit.json", audit_path]:
        _write_json(destination, audit_payload)
    for destination in [routerset_dir / "materialization_summary.json", materialization_path]:
        _write_json(destination, materialization_payload)
    _write_json(routerset_dir / "roads_anomaly_rebuild_summary.json", rebuild_summary)

    for destination in [routerset_dir / "manifest.jsonl", dataset_root / "manifest.jsonl"]:
        write_jsonl(destination, manifest_rows)
    for destination in [routerset_dir / "manifest_moe_train.jsonl", dataset_root / "manifest_moe_train.jsonl"]:
        write_jsonl(destination, moe_rows)


def _plot_python_executable() -> Path:
    candidates = [
        Path(".venv/bin/python").resolve(),
        Path(sys.executable).resolve(),
        Path("/usr/bin/python3"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        probe = subprocess.run(
            [str(candidate), "-c", "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('matplotlib') else 1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return candidate
    raise RuntimeError("Could not find a Python executable with matplotlib for plot regeneration")


def regenerate_plots(routerset_dir: Path) -> None:
    if not DEFAULT_PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"Missing plot script: {DEFAULT_PLOT_SCRIPT}")
    dataset_root = routerset_dir / "multilabel_dataset"
    plot_python = _plot_python_executable()
    subprocess.run(
        [
            str(plot_python),
            str(DEFAULT_PLOT_SCRIPT),
            "--dataset-dir",
            str(dataset_root),
            "--output-dir",
            str(dataset_root / "plots"),
        ],
        check=True,
    )
    root_plots = routerset_dir / "plots"
    if root_plots.exists():
        shutil.rmtree(root_plots)
    shutil.copytree(dataset_root / "plots", root_plots)


def _repo_tree_file_paths(local_root: Path, *, path_in_repo: str) -> set[str]:
    if not local_root.exists():
        return set()
    return {
        f"{path_in_repo}/{path.relative_to(local_root).as_posix()}"
        for path in local_root.rglob("*")
        if path.is_file()
    }


def publish_routerset_snapshot(routerset_dir: Path, repo_id: str, stale_paths: list[str]) -> None:
    from huggingface_hub import CommitOperationDelete, HfApi

    api = HfApi()
    repo_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    stale_remote_paths = set(stale_paths) & repo_files
    for local_dir, repo_prefix in [
        (routerset_dir / "images" / "roads", "images/roads"),
        (routerset_dir / "images" / "anomaly_detection", "images/anomaly_detection"),
        (routerset_dir / "multilabel_dataset" / "images" / "roads", "multilabel_dataset/images/roads"),
        (
            routerset_dir / "multilabel_dataset" / "images" / "anomaly_detection",
            "multilabel_dataset/images/anomaly_detection",
        ),
    ]:
        expected_paths = _repo_tree_file_paths(local_dir, path_in_repo=repo_prefix)
        stale_remote_paths.update(
            path
            for path in repo_files
            if path.startswith(f"{repo_prefix}/") and path not in expected_paths
        )
    delete_ops = [CommitOperationDelete(path_in_repo=path) for path in sorted(stale_remote_paths)]
    if delete_ops:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Delete stale roads/anomaly routerset files",
            operations=delete_ops,
        )

    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(routerset_dir / "multilabel_dataset"),
        path_in_repo="multilabel_dataset",
        commit_message="Rebuild routerset roads mosaics and anomaly tiles",
    )
    for dataset_name in ["roads", "anomaly_detection"]:
        local_dir = routerset_dir / "images" / dataset_name
        if local_dir.exists():
            api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(local_dir),
                path_in_repo=f"images/{dataset_name}",
                commit_message=f"Refresh routerset {dataset_name} root images after source rebuild",
            )
    for filename in [
        "README.md",
        "manifest.jsonl",
        "manifest_moe_train.jsonl",
        "summary.json",
        "audit.json",
        "materialization_summary.json",
        "label_vocab.json",
        "missing_ids.json",
        "review_queue.jsonl",
        "roads_anomaly_rebuild_summary.json",
    ]:
        local_path = routerset_dir / filename
        if local_path.exists():
            api.upload_file(
                repo_id=repo_id,
                repo_type="dataset",
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                commit_message="Mirror routerset metadata after roads/anomaly rebuild",
            )
    if (routerset_dir / "plots").exists():
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(routerset_dir / "plots"),
            path_in_repo="plots",
            commit_message="Refresh routerset plots after roads/anomaly rebuild",
        )


def main() -> int:
    args = parse_args()
    routerset_dir = Path(args.routerset_dir).resolve()
    dataset_root = routerset_dir / "multilabel_dataset"
    anomaly_source_root = Path(args.anomaly_source_root).resolve()
    backup_dir = (
        Path(args.backup_dir).resolve()
        if args.backup_dir is not None
        else (Path("outputs/routerset/backups") / f"roads_anomaly_source_rebuild_{_timestamp_token()}").resolve()
    )

    manifest_rows = load_jsonl(dataset_root / "manifest.jsonl")
    moe_rows = load_jsonl(routerset_dir / "manifest_moe_train.jsonl")
    label_vocab = load_label_vocab(dataset_root / "label_vocab.json")
    temp_image_root = (routerset_dir / ".tmp_roads_anomaly_build").resolve()
    if temp_image_root.exists():
        shutil.rmtree(temp_image_root)

    backup_changed_paths(routerset_dir, backup_dir)

    road_manifest_replacements, road_moe_replacements, road_summary, stale_road_paths = build_roads_replacements(
        manifest_rows,
        label_vocab=label_vocab,
        cache_dir=Path("outputs/routerset/cache/roads_nshot_arrays"),
        image_output_root=temp_image_root,
    )
    anomaly_manifest_replacements, anomaly_moe_replacements, anomaly_summary, stale_anomaly_paths = build_anomaly_replacements(
        manifest_rows,
        dataset_root=dataset_root,
        source_root=anomaly_source_root,
        label_vocab=label_vocab,
        image_output_root=temp_image_root,
    )

    _swap_dataset_image_dirs(routerset_dir, temp_image_root, "roads")
    _swap_dataset_image_dirs(routerset_dir, temp_image_root, "anomaly_detection")

    updated_manifest_rows = replace_dataset_rows(manifest_rows, dataset="roads", replacements=road_manifest_replacements)
    updated_manifest_rows = replace_dataset_rows(
        updated_manifest_rows,
        dataset="anomaly_detection",
        replacements=anomaly_manifest_replacements,
    )
    updated_moe_rows = replace_dataset_rows(moe_rows, dataset="roads", replacements=road_moe_replacements)
    updated_moe_rows = replace_dataset_rows(
        updated_moe_rows,
        dataset="anomaly_detection",
        replacements=anomaly_moe_replacements,
    )

    rebuild_summary = {
        "roads": road_summary,
        "anomaly_detection": anomaly_summary,
        "old_manifest_rows": len(manifest_rows),
        "new_manifest_rows": len(updated_manifest_rows),
    }
    refresh_metadata(routerset_dir, updated_manifest_rows, updated_moe_rows, rebuild_summary)
    if not args.skip_plots:
        regenerate_plots(routerset_dir)

    stale_paths = stale_road_paths + stale_anomaly_paths
    if args.publish:
        publish_routerset_snapshot(routerset_dir, args.repo_id, stale_paths)

    result = {
        "routerset_dir": str(routerset_dir),
        "anomaly_source_root": str(anomaly_source_root),
        "backup_dir": str(backup_dir),
        "old_manifest_rows": len(manifest_rows),
        "new_manifest_rows": len(updated_manifest_rows),
        "roads_old_rows": road_summary["old_rows"],
        "roads_new_rows": road_summary["new_rows"],
        "anomaly_old_rows": anomaly_summary["old_rows"],
        "anomaly_new_rows": anomaly_summary["new_rows"],
        "published": bool(args.publish),
        "repo_id": args.repo_id if args.publish else "",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
