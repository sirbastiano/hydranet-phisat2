from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import numpy as np
import torch

from hydranet.moe_training import (
    DEFAULT_ROUTERSET_EXPERTS,
    FULL_TRAINING_STAGES,
    _reduce_expert_output_to_routing_score,
    build_artifact_layout,
    build_routerset_training_tensor,
    configure_local_runtime_environment,
    default_rebuilt_manifest_path,
    generate_routerset_routing_targets,
    MoESwitcherLightningModule,
    legacy_root_rebuilt_manifest_path,
    prepare_routerset_training,
    RoutersetMoEDataModule,
    RoutersetMoEDataset,
    create_phidranet_release,
    materialize_routerset_dataset,
    materialize_routerset_training_manifest,
    normalize_routerset_array,
    preflight_routerset_training,
    probe_lightning_import,
    rebuild_routerset_split_manifest,
    resolve_routerset_dataset_root,
    run_full_training,
    run_training_startup_gate,
    train_switcher,
    run_moe_smoke_test,
    routerset_image_path,
    routerset_patch_token,
    write_deployment_readme,
)
from hydranet.models.moe_student import build_moe_student_from_models
from hydranet.models.student import create_phisatnet


def _row(
    source_dataset: str,
    source_sample_id: str,
    source_split: str,
    *,
    patch_width: Optional[int] = None,
    patch_height: Optional[int] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    return {
        "dataset_path": "unused",
        "image_ref": "unused",
        "label_coverages": {},
        "label_names": labels or [],
        "label_source": "native",
        "labels": [0],
        "native_label_names": list(labels or []),
        "patch_height": patch_height,
        "patch_width": patch_width,
        "patch_x": 0,
        "patch_y": 0,
        "record_status": "positive",
        "selection_bucket": source_dataset,
        "source_dataset": source_dataset,
        "source_sample_id": source_sample_id,
        "source_split": source_split,
        "source_storage_group": "test",
        "weak_label_names": [],
    }


def _write_routerset_fixture(
    root: Path,
    *,
    broken_fire_validation: bool = False,
    all_zero_worldfloods_train: bool = False,
) -> None:
    dataset_root = resolve_routerset_dataset_root(root)
    rows = [
        _row("fire", "0000001", "train", labels=["active_fire"]),
        _row("fire", "0000013", "train", labels=["active_fire"]),
        _row("fire", "0000007", "validation", labels=[] if broken_fire_validation else ["active_fire"]),
        _row("anomaly_detection", "0000002", "train", labels=["water"]),
        _row("anomaly_detection", "0000008", "validation", labels=["water"]),
        _row("worldfloods", "0000003", "train", patch_width=16, patch_height=16, labels=["cloud"]),
        _row("worldfloods", "0000009", "validation", patch_width=16, patch_height=16, labels=["cloud"]),
        _row("burned_area", "0000004", "train", patch_width=16, patch_height=16, labels=["burned_area"]),
        _row("burned_area", "0000010", "validation", patch_width=16, patch_height=16, labels=["burned_area"]),
        _row("roads", "0000005", "train", patch_width=16, patch_height=16, labels=["road_present"]),
        _row("roads", "0000011", "validation", patch_width=16, patch_height=16, labels=["road_present"]),
        _row("lc", "0000006", "train", patch_width=16, patch_height=16, labels=["tree_cover"]),
        _row("lc", "0000012", "validation", patch_width=16, patch_height=16, labels=["tree_cover"]),
    ]

    for row in rows:
        image_path = routerset_image_path(root, row)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        src = str(row["source_dataset"])
        if src == "burned_area":
            arr = np.random.randn(7, 16, 16).astype(np.float32)
        elif src in {"roads", "lc"}:
            arr = np.random.randint(0, 1000, size=(16, 16, 10), dtype=np.uint16)
        elif src == "anomaly_detection":
            arr = np.random.randn(8, 32, 32).astype(np.float32)
        elif src == "worldfloods" and all_zero_worldfloods_train and row["source_split"] == "train":
            arr = np.zeros((8, 16, 16), dtype=np.float32)
        else:
            arr = np.random.randn(8, 16, 16).astype(np.float32)
        np.save(image_path, arr)

        if broken_fire_validation and src == "fire" and row["source_split"] == "validation":
            row["record_status"] = "explicit_negative"

    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_text = "\n".join(json.dumps(row) for row in rows) + "\n"
    (dataset_root / "manifest.jsonl").write_text(manifest_text, encoding="utf-8")
    (dataset_root / "label_vocab.json").write_text(json.dumps({"labels": sorted({label for row in rows for label in row["label_names"]})}), encoding="utf-8")
    default_rebuilt_manifest_path(root).write_text(manifest_text, encoding="utf-8")


def _inject_non_finite_routerset_sample(
    root: Path,
    *,
    source_dataset: str,
    source_split: str,
    source_sample_id: str,
) -> None:
    row = {
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_sample_id": source_sample_id,
        "patch_x": 0,
        "patch_y": 0,
        "patch_width": None,
        "patch_height": None,
    }
    image_path = routerset_image_path(root, row)
    array = np.load(image_path)
    array = array.copy()
    array[..., 0, 0] = np.nan
    array[..., 0, 1] = np.inf
    array[..., 0, 2] = -np.inf
    np.save(image_path, array)


def _checkpoint_report(
    *,
    failed_expert: Optional[str] = None,
    weights_dir: str = "weights",
) -> Dict[str, Dict[str, object]]:
    report: Dict[str, Dict[str, object]] = {}
    for expert in DEFAULT_ROUTERSET_EXPERTS:
        source_path = f"catalog/{expert}/student.pt"
        payload: Dict[str, object] = {
            "status": "ok",
            "training": "finetuning",
            "n_shots": 5000,
            "catalog_path": "src/hydranet/model_weights.csv",
            "source_path": source_path,
            "deterministic_path": f"{weights_dir}/{source_path}",
            "checkpoint_path": f"{weights_dir}/{source_path}",
        }
        if expert == failed_expert:
            payload["status"] = "failed"
            payload["checkpoint_path"] = ""
            payload["error_type"] = "ValueError"
            payload["error"] = f"No weights found for {expert}"
        report[expert] = payload
    return report


def _drop_manifest_rows(root: Path, *, expert: str, split: str) -> None:
    dataset_root = resolve_routerset_dataset_root(root)
    paths = [dataset_root / "manifest.jsonl", default_rebuilt_manifest_path(root)]
    for path in paths:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        kept = [row for row in rows if not (row["source_dataset"] == expert and row["source_split"] == split)]
        path.write_text("\n".join(json.dumps(row) for row in kept) + "\n", encoding="utf-8")


def _write_mock_training_artifacts(output_dir: Path, release_dir: Path) -> None:
    for path in [
        output_dir / "runtime_environment.json",
        output_dir / "preflight_report.json",
        output_dir / "config.json",
        output_dir / "dataset_report.json",
        output_dir / "checkpoint_report.json",
        output_dir / "metrics.json",
        output_dir / "startup_log.txt",
        output_dir / "startup_stage.json",
        output_dir / "startup_gate.json",
        output_dir / "routing_predictions.jsonl",
        output_dir / "student_moe_bundle.pt",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".pt":
            path.write_bytes(b"bundle")
        else:
            path.write_text("{}", encoding="utf-8")
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "release_manifest.json").write_text("{}", encoding="utf-8")


def _set_fixture_marker(root: Path, row: Dict[str, object], value: float) -> None:
    image_path = routerset_image_path(root, row)
    array = np.load(image_path)
    array = np.asarray(array).copy()
    array.reshape(-1)[0] = value
    np.save(image_path, array)


class _FakeExpertModel(torch.nn.Module):
    def __init__(self, *, score_by_marker: Dict[int, float], single_channel: bool = False) -> None:
        super().__init__()
        self.score_by_marker = {int(key): float(value) for key, value in score_by_marker.items()}
        self.single_channel = bool(single_channel)

    @staticmethod
    def _logit(probability: float) -> float:
        probability = min(max(float(probability), 1e-4), 1.0 - 1e-4)
        return float(np.log(probability / (1.0 - probability)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        marker = int(round(float(inputs[0, 0, 0, 0].detach().cpu().item())))
        probability = self.score_by_marker.get(marker, 0.05)
        height, width = int(inputs.shape[-2]), int(inputs.shape[-1])
        if self.single_channel:
            logit = self._logit(probability)
            return torch.full((inputs.shape[0], 1, height, width), logit, dtype=inputs.dtype, device=inputs.device)
        bg_logit = 0.0
        fg_logit = self._logit(probability)
        logits = torch.zeros((inputs.shape[0], 2, height, width), dtype=inputs.dtype, device=inputs.device)
        logits[:, 0, :, :] = bg_logit
        logits[:, 1, :, :] = fg_logit
        return logits


class _FakeLoadingModule:
    def __init__(self, *, save_bundle_side_effect=None) -> None:
        self._save_bundle_side_effect = save_bundle_side_effect

    def save_student_moe_bundle(self, model, output_path, metadata=None) -> Path:
        if self._save_bundle_side_effect is not None:
            return self._save_bundle_side_effect(model, output_path, metadata=metadata)
        path = Path(output_path)
        path.write_bytes(b"bundle")
        return path


class TestMoETraining(unittest.TestCase):
    def test_routerset_helpers(self) -> None:
        row = {
            "source_dataset": "fire",
            "source_split": "train",
            "source_sample_id": "0000010",
            "patch_x": 0,
            "patch_y": 0,
            "patch_width": None,
            "patch_height": None,
        }
        self.assertEqual(routerset_patch_token(row), "0000010_0_0_full_full.npy")
        self.assertTrue(
            str(routerset_image_path("routerset", row)).endswith(
                "routerset/multilabel_dataset/images/fire/train/0000010_0_0_full_full.npy"
            )
        )

    def test_routerset_image_path_uses_swapped_tile_compatibility_for_materialized_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = resolve_routerset_dataset_root(root)
            row = {
                "source_dataset": "burned_area",
                "source_split": "train",
                "source_sample_id": "0000010",
                "patch_x": 0,
                "patch_y": 16,
                "patch_width": 16,
                "patch_height": 16,
            }
            swapped_path = dataset_root / "images" / "burned_area" / "train" / "0000010_16_0_16_16.npy"
            swapped_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(swapped_path, np.random.randn(7, 16, 16).astype(np.float32))

            self.assertEqual(routerset_patch_token(row), "0000010_0_16_16_16.npy")
            self.assertEqual(routerset_image_path(root, row), swapped_path)

    def test_routerset_image_path_prefers_materialized_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            materialized = root / "runtime" / "routerset_materialized" / "images" / "fire" / "train" / "tile.npy"
            materialized.parent.mkdir(parents=True, exist_ok=True)
            np.save(materialized, np.random.randn(8, 16, 16).astype(np.float32))
            row = {
                "source_dataset": "fire",
                "source_split": "train",
                "source_sample_id": "0000010",
                "patch_x": 0,
                "patch_y": 0,
                "patch_width": 16,
                "patch_height": 16,
                "materialized_image_path": str(materialized),
            }

            self.assertEqual(routerset_image_path(root, row), materialized)

    def test_materialize_routerset_training_manifest_tiles_oversized_arrays_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            runtime_root = root / "runtime"

            manifest_path = materialize_routerset_training_manifest(
                routerset_dir=root,
                manifest_path=resolve_routerset_dataset_root(root) / "manifest.jsonl",
                materialization_root=runtime_root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                target_size=16,
                target_channels=8,
            )

            rows = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            anomaly_train_rows = [
                row
                for row in rows
                if row["source_dataset"] == "anomaly_detection" and row["source_split"] == "train"
            ]

            self.assertEqual(len(anomaly_train_rows), 4)
            self.assertTrue(all(row["patch_width"] == 16 for row in anomaly_train_rows))
            self.assertTrue(all(row["patch_height"] == 16 for row in anomaly_train_rows))
            self.assertTrue(all("materialized_image_path" in row for row in anomaly_train_rows))
            for row in anomaly_train_rows:
                tile = np.load(Path(row["materialized_image_path"]))
                self.assertEqual(tile.shape, (8, 16, 16))

    def test_materialize_routerset_training_manifest_skips_smaller_patches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            runtime_root = root / "runtime"

            manifest_path = materialize_routerset_training_manifest(
                routerset_dir=root,
                manifest_path=resolve_routerset_dataset_root(root) / "manifest.jsonl",
                materialization_root=runtime_root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                target_size=16,
                target_channels=8,
            )

            rows = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            burned_area_train = next(
                row
                for row in rows
                if row["source_dataset"] == "burned_area" and row["source_split"] == "train"
            )

            self.assertEqual(burned_area_train["patch_width"], 16)
            self.assertEqual(burned_area_train["patch_height"], 16)
            self.assertNotIn("materialized_image_path", burned_area_train)

    def test_materialize_routerset_training_manifest_normalizes_fire_metadata_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            runtime_root = root / "runtime"

            manifest_path = materialize_routerset_training_manifest(
                routerset_dir=root,
                manifest_path=resolve_routerset_dataset_root(root) / "manifest.jsonl",
                materialization_root=runtime_root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                target_size=16,
                target_channels=8,
            )

            rows = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            fire_train = next(
                row
                for row in rows
                if row["source_dataset"] == "fire" and row["source_split"] == "train"
            )

            self.assertEqual(fire_train["patch_width"], 16)
            self.assertEqual(fire_train["patch_height"], 16)
            self.assertIn("materialized_image_path", fire_train)
            self.assertTrue(Path(fire_train["materialized_image_path"]).exists())

    def test_materialize_routerset_dataset_writes_full_npy_export_for_small_patches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            export_root = root / "materialized_16"

            summary = materialize_routerset_dataset(
                routerset_dir=root,
                output_dir=export_root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                target_size=16,
                target_channels=8,
            )

            manifest_path = export_root / "manifest_16.jsonl"
            self.assertEqual(summary["status"], "materialized")
            self.assertEqual(summary["manifest_path"], str(manifest_path))
            self.assertTrue(manifest_path.exists())
            self.assertTrue((export_root / "dataset_report.json").exists())
            self.assertTrue((export_root / "materialization_summary.json").exists())
            self.assertTrue((export_root / "label_vocab.json").exists())

            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 19)
            self.assertTrue(all("materialized_image_path" in row for row in rows))
            self.assertEqual(summary["materialized_unique_file_count"], 19)
            self.assertEqual(summary["materialization_summary"]["single_file_source_rows"], 11)
            self.assertEqual(summary["materialization_summary"]["tiled_source_rows"], 2)

            burned_area_train = next(
                row
                for row in rows
                if row["source_dataset"] == "burned_area" and row["source_split"] == "train"
            )
            roads_train = next(
                row
                for row in rows
                if row["source_dataset"] == "roads" and row["source_split"] == "train"
            )
            burned_area_tile = np.load(Path(burned_area_train["materialized_image_path"]))
            roads_tile = np.load(Path(roads_train["materialized_image_path"]))
            self.assertEqual(tuple(burned_area_tile.shape), (8, 16, 16))
            self.assertEqual(tuple(roads_tile.shape), (8, 16, 16))
            self.assertEqual(float(roads_tile[3].sum()), 0.0)
            self.assertTrue(np.isfinite(roads_tile).all())

    def test_materialize_routerset_dataset_does_not_require_training_ready_positive_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root, broken_fire_validation=True)
            export_root = root / "materialized_16"

            summary = materialize_routerset_dataset(
                routerset_dir=root,
                output_dir=export_root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                target_size=16,
                target_channels=8,
            )

            self.assertEqual(summary["status"], "materialized")
            self.assertTrue((export_root / "manifest_16.jsonl").exists())
            self.assertFalse(summary["fault_report"]["training_ready"])
            self.assertIn(
                {"code": "missing_positive_rows", "count": 0, "expert": "fire", "split": "validation"},
                summary["fault_report"]["training_blockers"],
            )

    def test_materialize_routerset_dataset_clean_excludes_all_zero_tiles_and_writes_fault_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(
                root,
                broken_fire_validation=True,
                all_zero_worldfloods_train=True,
            )
            export_root = root / "materialized_16_clean"

            summary = materialize_routerset_dataset(
                routerset_dir=root,
                output_dir=export_root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                target_size=16,
                target_channels=8,
                clean_export=True,
            )

            manifest_path = export_root / "manifest_16.jsonl"
            fault_rows_path = export_root / "fault_rows_16.jsonl"
            fault_report_path = export_root / "fault_report.json"
            self.assertEqual(summary["export_mode"], "clean")
            self.assertTrue(manifest_path.exists())
            self.assertTrue(fault_rows_path.exists())
            self.assertTrue(fault_report_path.exists())

            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            fault_rows = [json.loads(line) for line in fault_rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 18)
            self.assertEqual(len(fault_rows), 1)
            self.assertEqual(fault_rows[0]["source_dataset"], "worldfloods")
            self.assertEqual(fault_rows[0]["fault_codes"], ["all_zero_materialized_tile"])
            self.assertEqual(fault_rows[0]["fault_action"], "excluded_from_clean_export")
            self.assertEqual(summary["fault_report"]["row_faults_by_code"], {"all_zero_materialized_tile": 1})
            self.assertEqual(summary["fault_report"]["excluded_rows_by_expert"], {"worldfloods": 1})
            self.assertFalse(summary["fault_report"]["training_ready"])
            self.assertIn(
                {"code": "missing_positive_rows", "count": 0, "expert": "fire", "split": "validation"},
                summary["fault_report"]["training_blockers"],
            )

    def test_normalize_routerset_array_adapts_all_source_layouts(self) -> None:
        fire = normalize_routerset_array(np.random.randn(8, 16, 16).astype(np.float32), source_dataset="fire")
        burned_area = normalize_routerset_array(np.random.randn(7, 16, 16).astype(np.float32), source_dataset="burned_area")
        lc = normalize_routerset_array(np.random.randint(0, 10, size=(16, 16, 10), dtype=np.uint16), source_dataset="lc")
        roads = normalize_routerset_array(np.random.randint(0, 10, size=(16, 16, 10), dtype=np.uint16), source_dataset="roads")

        self.assertEqual(tuple(fire.shape), (8, 16, 16))
        self.assertEqual(tuple(burned_area.shape), (8, 16, 16))
        self.assertEqual(tuple(lc.shape), (8, 16, 16))
        self.assertEqual(tuple(roads.shape), (8, 16, 16))

    def test_normalize_routerset_array_matches_phi2fm_student_roads_contract(self) -> None:
        values = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=np.uint16)
        array = np.broadcast_to(values, (4, 4, 10)).copy()

        normalized = normalize_routerset_array(array, source_dataset="roads")

        self.assertTrue(torch.allclose(normalized[0], torch.full((4, 4), 0.01)))
        self.assertTrue(torch.allclose(normalized[1], torch.full((4, 4), 0.02)))
        self.assertTrue(torch.allclose(normalized[2], torch.full((4, 4), 0.03)))
        self.assertTrue(torch.allclose(normalized[3], torch.zeros((4, 4))))
        self.assertTrue(torch.allclose(normalized[4], torch.full((4, 4), 0.04)))
        self.assertTrue(torch.allclose(normalized[5], torch.full((4, 4), 0.05)))
        self.assertTrue(torch.allclose(normalized[6], torch.full((4, 4), 0.06)))
        self.assertTrue(torch.allclose(normalized[7], torch.full((4, 4), 0.07)))

    def test_normalize_routerset_array_copies_non_writable_input(self) -> None:
        array = np.arange(32, dtype=np.float32).reshape(8, 2, 2)
        array.setflags(write=False)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            normalized = normalize_routerset_array(array, source_dataset="fire")

        normalized[0, 0, 0] = 123.0
        self.assertEqual(float(normalized[0, 0, 0]), 123.0)
        self.assertEqual(float(array[0, 0, 0]), 0.0)

    def test_build_routerset_training_tensor_zero_pads_small_patches(self) -> None:
        roads = build_routerset_training_tensor(
            np.full((16, 16, 10), 1000, dtype=np.uint16),
            source_dataset="roads",
            target_size=32,
            target_channels=8,
        )
        burned_area = build_routerset_training_tensor(
            np.full((7, 16, 16), 1.0, dtype=np.float32),
            source_dataset="burned_area",
            target_size=32,
            target_channels=8,
        )

        self.assertEqual(tuple(roads.shape), (8, 32, 32))
        self.assertEqual(tuple(burned_area.shape), (8, 32, 32))
        self.assertTrue(torch.allclose(roads[0, :16, :16], torch.full((16, 16), 0.1)))
        self.assertTrue(torch.allclose(roads[3, :16, :16], torch.zeros((16, 16))))
        self.assertEqual(float(roads[:, 16:, :].sum().item()), 0.0)
        self.assertEqual(float(roads[:, :, 16:].sum().item()), 0.0)
        self.assertTrue(torch.all(burned_area[:7, :16, :16] > 0))
        self.assertEqual(float(burned_area[:, 16:, :].sum().item()), 0.0)
        self.assertEqual(float(burned_area[:, :, 16:].sum().item()), 0.0)

    def test_reduce_expert_output_to_routing_score_prefers_sparse_binary_activation(self) -> None:
        logits = torch.full((1, 1, 16, 16), -12.0)
        logits[0, 0, 5, 7] = 12.0

        score = _reduce_expert_output_to_routing_score(logits)

        self.assertGreater(score, 0.99)

    def test_normalize_routerset_array_sanitizes_non_finite_values(self) -> None:
        array = np.array(
            [
                [[np.nan, np.inf], [-np.inf, 5.0]],
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            dtype=np.float32,
        )

        normalized = normalize_routerset_array(array, source_dataset="fire", target_channels=2)

        self.assertTrue(np.isfinite(normalized.numpy()).all())
        self.assertEqual(float(normalized.min()), 0.0)
        self.assertEqual(float(normalized.max()), 5.0)
        self.assertIn(1.0, normalized.numpy())

    def test_dataset_supports_full_routerset_expert_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            train_dataset = RoutersetMoEDataset(
                root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                split="train",
                target_size=16,
            )
            val_dataset = RoutersetMoEDataset(
                root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                split="validation",
                target_size=16,
            )

            self.assertEqual(len(train_dataset), 10)
            self.assertEqual(len(val_dataset), 9)
            sample = train_dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (8, 16, 16))
            self.assertEqual(tuple(sample["target"].shape), (len(DEFAULT_ROUTERSET_EXPERTS),))
            self.assertEqual(tuple(sample["tile_origin"].shape), (2,))
            summary = train_dataset.summary()
            self.assertEqual(summary["training_shapes_by_expert"]["anomaly_detection"], {"8x16x16": 4})
            self.assertEqual(summary["positive_counts_by_expert"]["burned_area"], 1)
            self.assertEqual(summary["positive_counts_by_expert"]["fire"], 2)
            self.assertEqual(summary["raw_counts_by_expert"]["anomaly_detection"], 1)
            self.assertEqual(summary["normalization_modes_by_expert"]["roads"], {"phi2fm_student_s2_layout_scaled": 1})
            self.assertEqual(summary["normalization_modes_by_expert"]["lc"], {"phi2fm_student_s2_layout_scaled": 1})
            self.assertEqual(summary["manifest_shape_source_records_by_expert"]["roads"], 1)
            self.assertEqual(summary["manifest_shape_raw_samples_by_expert"]["roads"][0]["raw_shape"], [16, 16, 10])
            self.assertGreaterEqual(summary["manifest_shape_raw_samples_by_expert"]["roads"][0]["raw_min"], 0.0)
            self.assertLess(summary["manifest_shape_raw_samples_by_expert"]["roads"][0]["raw_max"], 1000.0)
            self.assertGreaterEqual(summary["manifest_shape_raw_samples_by_expert"]["roads"][0]["raw_zero_fraction"], 0.0)
            self.assertLessEqual(summary["manifest_shape_raw_samples_by_expert"]["roads"][0]["raw_zero_fraction"], 1.0)
            self.assertEqual(summary["num_manifest_shape_source_records"], 4)
            self.assertEqual(summary["num_compatibility_path_source_records"], 0)
            self.assertEqual(summary["manifest_path"], str(resolve_routerset_dataset_root(root) / "manifest.jsonl"))

    def test_dataset_summary_reports_compatibility_path_source_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            dataset_root = resolve_routerset_dataset_root(root)
            manifest_paths = [dataset_root / "manifest.jsonl", default_rebuilt_manifest_path(root)]
            for manifest_path in manifest_paths:
                rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                for row in rows:
                    if row["source_dataset"] == "burned_area" and row["source_split"] == "train":
                        row["patch_x"] = 0
                        row["patch_y"] = 16
                        break
                manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            swapped_path = dataset_root / "images" / "burned_area" / "train" / "0000004_16_0_16_16.npy"
            np.save(swapped_path, np.random.randn(7, 16, 16).astype(np.float32))

            train_dataset = RoutersetMoEDataset(
                root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                split="train",
                target_size=16,
            )

            summary = train_dataset.summary()
            self.assertEqual(summary["compatibility_path_source_records_by_expert"]["burned_area"], 1)
            self.assertEqual(summary["num_compatibility_path_source_records"], 1)
            self.assertEqual(summary["manifest_shape_raw_samples_by_expert"]["burned_area"][0]["raw_shape"], [7, 16, 16])
            self.assertLessEqual(
                summary["manifest_shape_raw_samples_by_expert"]["burned_area"][0]["raw_min"],
                summary["manifest_shape_raw_samples_by_expert"]["burned_area"][0]["raw_max"],
            )
            self.assertTrue(summary["manifest_shape_raw_samples_by_expert"]["burned_area"][0]["used_compatibility_path"])
            self.assertEqual(summary["normalization_modes_by_expert"]["burned_area"], {"channel_adapter": 1})

    def test_generate_routerset_routing_targets_writes_multi_hot_targets_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            manifest_path = resolve_routerset_dataset_root(root) / "manifest.jsonl"
            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            marker_map = {
                ("anomaly_detection", "validation"): 1,
                ("burned_area", "validation"): 2,
                ("fire", "validation"): 3,
                ("lc", "validation"): 4,
                ("roads", "validation"): 5,
                ("worldfloods", "validation"): 6,
            }
            for row in rows:
                marker = marker_map.get((str(row["source_dataset"]), str(row["source_split"])))
                if marker is not None:
                    _set_fixture_marker(root, row, marker)

            fake_models = {
                "anomaly_detection": _FakeExpertModel(score_by_marker={1: 0.9, 2: 0.1, 3: 0.8, 4: 0.1, 5: 0.1, 6: 0.1}),
                "burned_area": _FakeExpertModel(score_by_marker={1: 0.1, 2: 0.9, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1}),
                "fire": _FakeExpertModel(score_by_marker={1: 0.1, 2: 0.1, 3: 0.95, 4: 0.1, 5: 0.1, 6: 0.1}),
                "lc": _FakeExpertModel(score_by_marker={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.9, 5: 0.1, 6: 0.1}),
                "roads": _FakeExpertModel(score_by_marker={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.9, 6: 0.1}, single_channel=True),
                "worldfloods": _FakeExpertModel(score_by_marker={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.9}),
            }
            report_path = root / "routing_target_report.json"

            with patch("hydranet.moe_training._load_student_expert_models", return_value=fake_models), patch(
                "hydranet.moe_training._calibrate_routing_thresholds",
                return_value={
                    "anomaly_detection": 0.7,
                    "burned_area": 0.8,
                    "fire": 0.8,
                    "lc": 0.8,
                    "roads": 0.8,
                    "worldfloods": 0.8,
                },
            ):
                output_manifest = generate_routerset_routing_targets(
                    root,
                    manifest_path=manifest_path,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    report_path=report_path,
                    target_size=16,
                    target_channels=8,
                    weights_dir=str(root / "weights"),
                )

            rewritten_rows = [json.loads(line) for line in output_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            fire_validation = next(
                row for row in rewritten_rows if row["source_dataset"] == "fire" and row["source_split"] == "validation"
            )
            self.assertEqual(fire_validation["routing_target_source"], "expert_inference_v1")
            self.assertEqual(fire_validation["routing_target_experts"], ["anomaly_detection", "fire"])
            self.assertEqual(fire_validation["routing_target"], [1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            self.assertEqual(list(fire_validation["routing_scores"].keys()), list(DEFAULT_ROUTERSET_EXPERTS))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "generated")
            self.assertGreater(report["average_active_experts"], 1.0)

    def test_generate_routerset_routing_targets_falls_back_to_top1_when_no_threshold_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            manifest_path = resolve_routerset_dataset_root(root) / "manifest.jsonl"
            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            fire_validation = next(
                row for row in rows if row["source_dataset"] == "fire" and row["source_split"] == "validation"
            )
            _set_fixture_marker(root, fire_validation, 7)

            fake_models = {
                "anomaly_detection": _FakeExpertModel(score_by_marker={7: 0.3}),
                "burned_area": _FakeExpertModel(score_by_marker={7: 0.2}),
                "fire": _FakeExpertModel(score_by_marker={7: 0.4}),
                "lc": _FakeExpertModel(score_by_marker={7: 0.1}),
                "roads": _FakeExpertModel(score_by_marker={7: 0.05}, single_channel=True),
                "worldfloods": _FakeExpertModel(score_by_marker={7: 0.25}),
            }

            with patch("hydranet.moe_training._load_student_expert_models", return_value=fake_models), patch(
                "hydranet.moe_training._calibrate_routing_thresholds",
                return_value={expert: 0.95 for expert in DEFAULT_ROUTERSET_EXPERTS},
            ):
                output_manifest = generate_routerset_routing_targets(
                    root,
                    manifest_path=manifest_path,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    target_size=16,
                    target_channels=8,
                )

            rewritten_rows = [json.loads(line) for line in output_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            updated_fire = next(
                row for row in rewritten_rows if row["source_dataset"] == "fire" and row["source_split"] == "validation"
            )
            self.assertEqual(updated_fire["routing_target_experts"], ["fire"])
            self.assertEqual(sum(updated_fire["routing_target"]), 1.0)

    def test_routerset_dataset_uses_generated_routing_target_instead_of_source_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            manifest_path = resolve_routerset_dataset_root(root) / "manifest.jsonl"
            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                if row["source_dataset"] == "fire" and row["source_split"] == "train":
                    row["routing_target"] = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    row["routing_target_experts"] = ["anomaly_detection", "fire"]
                    row["routing_scores"] = {expert: 0.0 for expert in DEFAULT_ROUTERSET_EXPERTS}
                    row["routing_target_source"] = "expert_inference_v1"
                    break
            manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            train_dataset = RoutersetMoEDataset(
                root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                split="train",
                target_size=16,
            )
            fire_sample = next(sample for sample in train_dataset if sample["expert_name"] == "fire")
            self.assertEqual(fire_sample["target"].tolist(), [1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def test_dataset_sanitizes_non_finite_anomaly_tiles_and_reports_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            _inject_non_finite_routerset_sample(
                root,
                source_dataset="anomaly_detection",
                source_split="train",
                source_sample_id="0000002",
            )

            train_dataset = RoutersetMoEDataset(
                root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                split="train",
                target_size=16,
            )
            corrupt_samples = [
                train_dataset[index]
                for index, record in enumerate(train_dataset.records)
                if record.base_source_sample_id == "0000002"
            ]

            self.assertTrue(corrupt_samples)
            self.assertTrue(all(np.isfinite(sample["image"].numpy()).all() for sample in corrupt_samples))
            summary = train_dataset.summary()
            self.assertEqual(summary["non_finite_source_records_by_expert"]["anomaly_detection"], 1)
            self.assertEqual(summary["sanitized_training_tensors_by_expert"]["anomaly_detection"], 4)
            self.assertEqual(summary["num_non_finite_source_records"], 1)
            self.assertEqual(summary["num_sanitized_training_tensors"], 4)
            self.assertEqual(summary["num_non_finite_values"], 96)

    def test_rebuild_routerset_split_manifest_repairs_fire_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root, broken_fire_validation=True)
            rebuilt = rebuild_routerset_split_manifest(
                root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
            )
            rows = [json.loads(line) for line in rebuilt.read_text(encoding="utf-8").splitlines()]
            fire_train = [row for row in rows if row["source_dataset"] == "fire" and row.get("moe_split") == "train" and row["record_status"] == "positive"]
            fire_val = [row for row in rows if row["source_dataset"] == "fire" and row.get("moe_split") == "validation" and row["record_status"] == "positive"]

            self.assertTrue(fire_train)
            self.assertTrue(fire_val)
            self.assertEqual(fire_val[0]["source_split"], "train")

    def test_preflight_routerset_training_generates_targets_before_dataset_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"
            observed: list[str] = []

            def _fake_generate(**kwargs):
                observed.append("generate")
                report_path = kwargs.get("report_path")
                if report_path is not None:
                    Path(report_path).write_text("{}", encoding="utf-8")
                return Path(kwargs["output_path"])

            def _fake_collect(*args, **kwargs):
                observed.append("dataset_report")
                return {
                    "train": {"counts_by_expert": {name: 1 for name in DEFAULT_ROUTERSET_EXPERTS},
                              "positive_counts_by_expert": {name: 1 for name in DEFAULT_ROUTERSET_EXPERTS},
                              "training_shapes_by_expert": {name: {"8x16x16": 1} for name in DEFAULT_ROUTERSET_EXPERTS}},
                    "validation": {"counts_by_expert": {name: 1 for name in DEFAULT_ROUTERSET_EXPERTS},
                                   "positive_counts_by_expert": {name: 1 for name in DEFAULT_ROUTERSET_EXPERTS},
                                   "training_shapes_by_expert": {name: {"8x16x16": 1} for name in DEFAULT_ROUTERSET_EXPERTS}},
                    "sampling": {"balanced_sampling": True, "expanded_counts_by_expert": {}, "class_weights": {}, "num_samples": 1},
                }

            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "weights")),
            ), patch(
                "hydranet.moe_training.generate_routerset_routing_targets",
                side_effect=_fake_generate,
            ), patch(
                "hydranet.moe_training.collect_routerset_dataset_report",
                side_effect=_fake_collect,
            ), patch("hydranet.moe_training.validate_routerset_dataset_report", return_value=None):
                report = preflight_routerset_training(
                    routerset_dir=root,
                    output_dir=output_dir,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    target_size=16,
                    target_channels=8,
                    runtime_root=root / "runtime",
                )

            self.assertEqual(observed, ["generate", "dataset_report"])
            self.assertEqual(
                report["manifest_path"],
                str(root / "runtime" / "routerset_materialized" / "manifest_16.jsonl"),
            )
            self.assertTrue((output_dir / "routing_target_report.json").exists())

    def test_preflight_routerset_training_does_not_force_if_skip_routing_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"
            captured: dict[str, bool] = {}

            def _fake_generate(**kwargs) -> Path:
                captured["force"] = kwargs["force"]
                Path(kwargs["output_path"]).write_text("{}", encoding="utf-8")
                if kwargs.get("report_path") is not None:
                    Path(kwargs["report_path"]).write_text("{}", encoding="utf-8")
                return Path(kwargs["output_path"])

            def _fake_collect(*args, **kwargs):
                return {
                    "train": {"counts_by_expert": {name: 1 for name in DEFAULT_ROUTERSET_EXPERTS},
                              "positive_counts_by_expert": {name: 1 for name in DEFAULT_ROUTERSET_EXPERTS},
                              "training_shapes_by_expert": {name: {"8x16x16": 1} for name in DEFAULT_ROUTERSET_EXPERTS}},
                    "validation": {"counts_by_expert": {name: 1 for name in DEFAULT_ROUTERSET_EXPERTS},
                                   "positive_counts_by_expert": {name: 1 for name in DEFAULT_ROUTERSET_EXPERTS},
                                   "training_shapes_by_expert": {name: {"8x16x16": 1} for name in DEFAULT_ROUTERSET_EXPERTS}},
                    "sampling": {"balanced_sampling": True, "expanded_counts_by_expert": {}, "class_weights": {}, "num_samples": 1},
                }

            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "weights")),
            ), patch(
                "hydranet.moe_training.generate_routerset_routing_targets",
                side_effect=_fake_generate,
            ), patch(
                "hydranet.moe_training.collect_routerset_dataset_report",
                side_effect=_fake_collect,
            ), patch("hydranet.moe_training.validate_routerset_dataset_report", return_value=None):
                preflight_routerset_training(
                    routerset_dir=root,
                    output_dir=output_dir,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    target_size=16,
                    target_channels=8,
                    runtime_root=root / "runtime",
                    skip_existing_routing_targets=True,
                )

            self.assertIn("force", captured)
            self.assertFalse(captured["force"])

    def test_train_switcher_generates_targets_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"
            release_dir = output_dir / "bundle" / "phidranet_demo"
            observed: list[str] = []

            class FakeModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.expert_names = list(DEFAULT_ROUTERSET_EXPERTS)
                    self.encoder_source_task = "anomaly_detection"
                    self.threshold = 0.5
                    self.top_k = len(DEFAULT_ROUTERSET_EXPERTS)
                    self.encoder = torch.nn.Linear(1, 1)
                    self.switcher = torch.nn.Linear(1, len(DEFAULT_ROUTERSET_EXPERTS))
                    self.experts = torch.nn.ModuleDict({name: torch.nn.Identity() for name in DEFAULT_ROUTERSET_EXPERTS})

            class FakeLightningModule:
                def __init__(self, model, *, learning_rate: float, weight_decay: float) -> None:
                    self.model = model

            class FakeTrainer:
                callback_metrics = {"val_loss": 0.25}

                def fit(self, lightning_module, train_dataloaders=None, val_dataloaders=None) -> None:
                    _ = (lightning_module, train_dataloaders, val_dataloaders)
                    observed.append("fit")

            def _fake_generate(**kwargs):
                observed.append("generate")
                report_path = kwargs.get("report_path")
                if report_path is not None:
                    Path(report_path).write_text("{}", encoding="utf-8")
                return Path(kwargs["output_path"])

            def _fake_collect(*args, **kwargs):
                observed.append("dataset_report")
                return {
                    "train": {"counts_by_expert": {name: 1 for name in DEFAULT_ROUTERSET_EXPERTS},
                              "positive_counts_by_expert": {name: 1 for name in DEFAULT_ROUTERSET_EXPERTS},
                              "training_shapes_by_expert": {name: {"8x16x16": 1} for name in DEFAULT_ROUTERSET_EXPERTS}},
                    "validation": {"counts_by_expert": {name: 1 for name in DEFAULT_ROUTERSET_EXPERTS},
                                   "positive_counts_by_expert": {name: 1 for name in DEFAULT_ROUTERSET_EXPERTS},
                                   "training_shapes_by_expert": {name: {"8x16x16": 1} for name in DEFAULT_ROUTERSET_EXPERTS}},
                    "sampling": {"balanced_sampling": True, "expanded_counts_by_expert": {}, "class_weights": {}, "num_samples": 1},
                }

            def _fake_capture(model, out_dir):
                path = Path(out_dir) / "baseline_summary.json"
                path.write_text("{}", encoding="utf-8")
                return path

            def _fake_save_bundle(model, path, **kwargs):
                Path(path).write_bytes(b"bundle")
                return Path(path)

            def _fake_write_predictions(model, dataloader, output_path):
                Path(output_path).write_text("{}\n", encoding="utf-8")
                return Path(output_path)

            def _fake_release(**kwargs):
                release_dir.mkdir(parents=True, exist_ok=True)
                for filename in ["student_moe_bundle.pt", "config.json", "metrics.json", "baseline_summary.json", "routing_predictions.jsonl", "dataset_report.json", "checkpoint_report.json", "release_manifest.json", "DEPLOY.md"]:
                    path = release_dir / filename
                    if path.suffix == ".pt":
                        path.write_bytes(b"bundle")
                    else:
                        path.write_text("{}", encoding="utf-8")
                return {
                    "release_dir": str(release_dir),
                    "bundle_path": str(release_dir / "student_moe_bundle.pt"),
                    "config_path": str(release_dir / "config.json"),
                    "metrics_path": str(release_dir / "metrics.json"),
                    "baseline_summary_path": str(release_dir / "baseline_summary.json"),
                    "routing_predictions_path": str(release_dir / "routing_predictions.jsonl"),
                    "dataset_report_path": str(release_dir / "dataset_report.json"),
                    "checkpoint_report_path": str(release_dir / "checkpoint_report.json"),
                    "release_manifest_path": str(release_dir / "release_manifest.json"),
                    "deployment_path": str(release_dir / "DEPLOY.md"),
                }

            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "weights")),
            ), patch("hydranet.moe_training.generate_routerset_routing_targets", side_effect=_fake_generate), patch(
                "hydranet.moe_training.collect_routerset_dataset_report",
                side_effect=_fake_collect,
            ), patch("hydranet.moe_training.validate_routerset_dataset_report", return_value=None), patch(
                "hydranet.moe_training.probe_lightning_import",
                return_value=None,
            ), patch("hydranet.moe_training.build_routerset_moe", return_value=FakeModel()), patch(
                "hydranet.moe_training.capture_baseline_summary",
                side_effect=_fake_capture,
            ), patch("hydranet.moe_training._loading_module", return_value=_FakeLoadingModule(save_bundle_side_effect=_fake_save_bundle)), patch(
                "hydranet.moe_training.write_routing_predictions",
                side_effect=_fake_write_predictions,
            ), patch("hydranet.moe_training.create_phidranet_release", side_effect=_fake_release), patch(
                "hydranet.moe_training.load_lightning_training_components",
                return_value=(FakeLightningModule, lambda *args, **kwargs: FakeTrainer(), lambda *args, **kwargs: None),
            ):
                summary = train_switcher(
                    routerset_dir=root,
                    output_dir=output_dir,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    runtime_root=root / "runtime",
                    target_size=16,
                    target_channels=8,
                    release_name="demo",
                    startup_timeout_seconds=1,
                    max_epochs=1,
                )

            self.assertEqual(observed[:2], ["generate", "dataset_report"])
            self.assertEqual(summary["status"], "completed")

    def test_rebuild_routerset_split_manifest_syncs_legacy_root_copy_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root, broken_fire_validation=True)
            legacy_root_rebuilt_manifest_path(root).write_text("stale\n", encoding="utf-8")

            rebuilt = rebuild_routerset_split_manifest(
                root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
            )

            self.assertEqual(
                legacy_root_rebuilt_manifest_path(root).read_text(encoding="utf-8"),
                rebuilt.read_text(encoding="utf-8"),
            )

    def test_preflight_reports_dataset_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "weights")),
            ):
                report = preflight_routerset_training(
                    routerset_dir=root,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    target_size=16,
                    output_dir=root / "out",
                    release_name="demo",
                )

            self.assertEqual(report["release_name"], "phidranet_demo")
            self.assertIn("dataset_report", report)
            self.assertIn("checkpoint_report", report)
            self.assertTrue((root / "out" / "dataset_report.json").exists())
            self.assertTrue((root / "out" / "checkpoint_report.json").exists())
            self.assertTrue((root / "out" / "preflight_report.json").exists())
            checkpoint_report = json.loads((root / "out" / "checkpoint_report.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(checkpoint_report), list(DEFAULT_ROUTERSET_EXPERTS))
            self.assertEqual(checkpoint_report["fire"]["status"], "ok")
            self.assertEqual(
                report["dataset_report"]["train"]["normalization_modes_by_expert"]["roads"],
                {"phi2fm_student_s2_layout_scaled": 1},
            )
            self.assertEqual(
                report["dataset_report"]["train"]["manifest_shape_raw_samples_by_expert"]["roads"][0]["raw_shape"],
                [16, 16, 10],
            )
            self.assertGreaterEqual(
                report["dataset_report"]["train"]["manifest_shape_raw_samples_by_expert"]["roads"][0]["raw_min"],
                0.0,
            )
            self.assertLess(
                report["dataset_report"]["train"]["manifest_shape_raw_samples_by_expert"]["roads"][0]["raw_max"],
                1000.0,
            )
            self.assertEqual(report["dataset_report"]["train"]["num_manifest_shape_source_records"], 10)
            self.assertEqual(report["dataset_report"]["train"]["num_compatibility_path_source_records"], 0)

    def test_preflight_reports_sanitized_non_finite_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            _inject_non_finite_routerset_sample(
                root,
                source_dataset="anomaly_detection",
                source_split="train",
                source_sample_id="0000002",
            )
            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "weights")),
            ):
                report = preflight_routerset_training(
                    routerset_dir=root,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    target_size=16,
                    output_dir=root / "out",
                    release_name="demo",
                )

            train_report = report["dataset_report"]["train"]
            self.assertEqual(train_report["non_finite_source_records_by_expert"]["anomaly_detection"], 1)
            self.assertEqual(train_report["sanitized_training_tensors_by_expert"]["anomaly_detection"], 4)
            self.assertEqual(train_report["num_non_finite_source_records"], 1)
            self.assertEqual(train_report["num_sanitized_training_tensors"], 4)

    def test_preflight_rejects_missing_positive_experts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root, broken_fire_validation=True)

            with self.assertRaisesRegex(ValueError, "split rebuild required"):
                preflight_routerset_training(
                    routerset_dir=root,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    target_size=16,
                    output_dir=root / "out",
                    release_name="demo",
                )

    def test_preflight_rejects_missing_required_expert_in_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            _drop_manifest_rows(root, expert="roads", split="validation")

            with self.assertRaisesRegex(ValueError, "roads \\(validation\\)"):
                preflight_routerset_training(
                    routerset_dir=root,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    target_size=16,
                    output_dir=root / "out",
                    release_name="demo",
                )

    def test_preflight_rebuilds_broken_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root, broken_fire_validation=True)
            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "weights")),
            ):
                report = preflight_routerset_training(
                    routerset_dir=root,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    target_size=16,
                    output_dir=root / "out",
                    release_name="demo",
                    rebuild_splits=True,
                    rebuilt_manifest_out=default_rebuilt_manifest_path(root),
                )

            self.assertEqual(report["manifest_path"], str(default_rebuilt_manifest_path(root)))
            self.assertEqual(report["dataset_report"]["validation"]["raw_positive_counts_by_expert"]["fire"], 1)

    def test_preflight_writes_failed_checkpoint_report_before_aborting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"

            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(failed_expert="fire", weights_dir=str(root / "weights")),
            ):
                with self.assertRaisesRegex(ValueError, "fire from catalog/fire/student.pt"):
                    preflight_routerset_training(
                        routerset_dir=root,
                        expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                        target_size=16,
                        output_dir=output_dir,
                        release_name="demo",
                    )

            checkpoint_report = json.loads((output_dir / "checkpoint_report.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint_report["fire"]["status"], "failed")
            self.assertEqual(checkpoint_report["roads"]["status"], "ok")
            self.assertTrue((output_dir / "dataset_report.json").exists())

    def test_configure_local_runtime_environment_writes_local_cache_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = configure_local_runtime_environment(
                output_dir=root / "out",
                runtime_root=root / "runtime",
            )

            self.assertEqual(report["runtime_root"], str(root / "runtime"))
            self.assertEqual(report["weights_dir"], str(root / "runtime" / "weights"))
            self.assertTrue((root / "runtime" / "cache" / "huggingface").exists())
            self.assertEqual(report["env"]["HF_HUB_DISABLE_XET"], "1")

    def test_configure_local_runtime_environment_rejects_weights_outside_runtime_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            with self.assertRaisesRegex(ValueError, "weights_dir must stay under"):
                configure_local_runtime_environment(
                    output_dir=root / "out",
                    runtime_root=root / "runtime",
                    weights_dir=root / "weights",
                )

    def test_run_training_startup_gate_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"
            with patch("hydranet.moe_training.probe_lightning_import", return_value=None):
                report = run_training_startup_gate(
                    routerset_dir=root,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    target_size=16,
                    target_channels=8,
                    output_dir=output_dir,
                    startup_timeout_seconds=1,
                )

            self.assertEqual(report["lightning_probe"]["status"], "ok")
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["timeout_seconds"], 1)
            self.assertTrue((output_dir / "startup_gate.json").exists())
            self.assertEqual(tuple(report["sample_image_shape"]), (8, 16, 16))

    def test_run_training_startup_gate_probe_failure_writes_timeout_and_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"

            with patch(
                "hydranet.moe_training.probe_lightning_import",
                side_effect=RuntimeError("No module named pytorch_lightning"),
            ):
                with self.assertRaisesRegex(RuntimeError, "No module named pytorch_lightning"):
                    run_training_startup_gate(
                        routerset_dir=root,
                        expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                        target_size=16,
                        target_channels=8,
                        output_dir=output_dir,
                        startup_timeout_seconds=7,
                    )

            report = json.loads((output_dir / "startup_gate.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["timeout_seconds"], 7)
            self.assertEqual(report["error_type"], "RuntimeError")
            self.assertIn("No module named pytorch_lightning", report["error"])
            self.assertIn("RuntimeError", report["traceback"])
            self.assertEqual(report["lightning_probe"]["status"], "failed")
            self.assertEqual(report["lightning_probe"]["timeout_seconds"], 7)
            self.assertIn("RuntimeError", report["lightning_probe"]["traceback"])

    def test_prepare_routerset_training_writes_prepare_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "weights")),
            ), patch("hydranet.moe_training.probe_lightning_import", return_value=None):
                report = prepare_routerset_training(
                    routerset_dir=root,
                    output_dir=root / "out",
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    runtime_root=root / "runtime",
                    target_size=16,
                    target_channels=8,
                    rebuild_splits=True,
                    rebuilt_manifest_out=default_rebuilt_manifest_path(root),
                    release_name="demo",
                    startup_timeout_seconds=1,
                )

            self.assertEqual(report["status"], "prepared")
            self.assertTrue((root / "out" / "runtime_environment.json").exists())
            self.assertTrue((root / "out" / "startup_gate.json").exists())
            self.assertTrue((root / "out" / "prepare_report.json").exists())

    def test_train_switcher_summary_records_contract_and_artifact_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"
            release_dir = output_dir / "bundle" / "phidranet_demo"

            class FakeLightningModule:
                def __init__(self, model, *, learning_rate: float, weight_decay: float) -> None:
                    self.model = model
                    self.learning_rate = learning_rate
                    self.weight_decay = weight_decay

            class FakeTrainer:
                callback_metrics = {"val_loss": 0.25}

                def fit(self, lightning_module, train_dataloaders=None, val_dataloaders=None) -> None:
                    _ = (lightning_module, train_dataloaders, val_dataloaders)

            class FakeModel:
                encoder_source_task = "fire"

            def _fake_capture_baseline_summary(model, output_dir) -> Path:
                _ = model
                path = Path(output_dir) / "baseline_summary.json"
                path.write_text("{}", encoding="utf-8")
                return path

            def _fake_save_bundle(model, output_path, metadata=None) -> Path:
                _ = (model, metadata)
                path = Path(output_path)
                path.write_bytes(b"bundle")
                return path

            def _fake_write_predictions(model, dataloader, output_path) -> Path:
                _ = (model, dataloader)
                path = Path(output_path)
                path.write_text("{}\n", encoding="utf-8")
                return path

            def _fake_create_release(**kwargs):
                _ = kwargs
                release_dir.mkdir(parents=True, exist_ok=True)
                for filename in [
                    "student_moe_bundle.pt",
                    "config.json",
                    "metrics.json",
                    "baseline_summary.json",
                    "routing_predictions.jsonl",
                    "dataset_report.json",
                    "checkpoint_report.json",
                    "release_manifest.json",
                    "DEPLOY.md",
                ]:
                    (release_dir / filename).write_text("{}", encoding="utf-8")
                return {
                    "release_dir": str(release_dir),
                    "bundle_path": str(release_dir / "student_moe_bundle.pt"),
                    "config_path": str(release_dir / "config.json"),
                    "metrics_path": str(release_dir / "metrics.json"),
                    "baseline_summary_path": str(release_dir / "baseline_summary.json"),
                    "routing_predictions_path": str(release_dir / "routing_predictions.jsonl"),
                    "dataset_report_path": str(release_dir / "dataset_report.json"),
                    "checkpoint_report_path": str(release_dir / "checkpoint_report.json"),
                    "release_manifest_path": str(release_dir / "release_manifest.json"),
                    "deployment_path": str(release_dir / "DEPLOY.md"),
                }

            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "weights")),
            ), patch("hydranet.moe_training.probe_lightning_import", return_value=None), patch(
                "hydranet.moe_training.build_routerset_moe",
                return_value=FakeModel(),
            ), patch(
                "hydranet.moe_training.capture_baseline_summary",
                side_effect=_fake_capture_baseline_summary,
            ), patch(
                "hydranet.moe_training._loading_module",
                return_value=_FakeLoadingModule(save_bundle_side_effect=_fake_save_bundle),
            ), patch(
                "hydranet.moe_training.write_routing_predictions",
                side_effect=_fake_write_predictions,
            ), patch(
                "hydranet.moe_training.create_phidranet_release",
                side_effect=_fake_create_release,
            ), patch(
                "hydranet.moe_training.load_lightning_training_components",
                return_value=(FakeLightningModule, lambda *args, **kwargs: FakeTrainer(), lambda *args, **kwargs: None),
            ):
                summary = train_switcher(
                    routerset_dir=root,
                    output_dir=output_dir,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    runtime_root=root / "runtime",
                    target_size=16,
                    target_channels=8,
                    release_name="demo",
                    startup_timeout_seconds=1,
                    max_epochs=1,
                )

            self.assertEqual(summary["status"], "completed")
            contract = summary["contract"]
            self.assertEqual(contract["canonical_stages"], list(FULL_TRAINING_STAGES))
            self.assertEqual(contract["completed_stages"], ["preflight", "startup_gate", "training", "export"])
            self.assertEqual(contract["next_stage"], "smoke")
            self.assertEqual(contract["expected_artifact_names"]["run_dir"]["summary"], "summary.json")
            self.assertEqual(contract["expected_artifact_names"]["run_dir"]["startup_gate"], "startup_gate.json")
            self.assertEqual(contract["expected_artifact_names"]["release_dir"]["release_manifest"], "release_manifest.json")
            self.assertEqual(contract["layout"]["directories"]["bundle_root"], str(output_dir / "bundle"))
            self.assertEqual(contract["layout"]["directories"]["release_dir"], str(release_dir))
            self.assertEqual(contract["layout"]["directories"]["checkpoints_dir"], str(root / "runtime" / "weights"))
            self.assertTrue((output_dir / "summary.json").exists())

    def test_train_switcher_records_mocked_stage_sequence_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"
            release_dir = output_dir / "bundle" / "phidranet_demo"
            observed_events: list[str] = []

            class FakeLightningModule:
                def __init__(self, model, *, learning_rate: float, weight_decay: float) -> None:
                    self.model = model
                    self.learning_rate = learning_rate
                    self.weight_decay = weight_decay

            class FakeTrainer:
                callback_metrics = {"val_loss": 0.125}

                def fit(self, lightning_module, train_dataloaders=None, val_dataloaders=None) -> None:
                    _ = (lightning_module, train_dataloaders, val_dataloaders)
                    observed_events.append("fit")

            class FakeModel:
                encoder_source_task = "fire"

            def _fake_probe(timeout_seconds: int) -> None:
                observed_events.append(f"startup_gate:{timeout_seconds}")

            def _fake_capture_baseline_summary(model, out_dir) -> Path:
                _ = model
                observed_events.append("prepare")
                path = Path(out_dir) / "baseline_summary.json"
                path.write_text("{}", encoding="utf-8")
                return path

            def _fake_save_bundle(model, output_path, metadata=None) -> Path:
                _ = (model, metadata)
                observed_events.append("export")
                path = Path(output_path)
                path.write_bytes(b"bundle")
                return path

            def _fake_write_predictions(model, dataloader, output_path) -> Path:
                _ = (model, dataloader)
                path = Path(output_path)
                path.write_text("{}\n", encoding="utf-8")
                return path

            def _fake_create_release(**kwargs):
                _ = kwargs
                for filename in [
                    "student_moe_bundle.pt",
                    "config.json",
                    "metrics.json",
                    "baseline_summary.json",
                    "routing_predictions.jsonl",
                    "dataset_report.json",
                    "checkpoint_report.json",
                    "release_manifest.json",
                    "DEPLOY.md",
                ]:
                    (release_dir / filename).parent.mkdir(parents=True, exist_ok=True)
                    (release_dir / filename).write_text("{}", encoding="utf-8")
                return {
                    "release_dir": str(release_dir),
                    "bundle_path": str(release_dir / "student_moe_bundle.pt"),
                    "config_path": str(release_dir / "config.json"),
                    "metrics_path": str(release_dir / "metrics.json"),
                    "baseline_summary_path": str(release_dir / "baseline_summary.json"),
                    "routing_predictions_path": str(release_dir / "routing_predictions.jsonl"),
                    "dataset_report_path": str(release_dir / "dataset_report.json"),
                    "checkpoint_report_path": str(release_dir / "checkpoint_report.json"),
                    "release_manifest_path": str(release_dir / "release_manifest.json"),
                    "deployment_path": str(release_dir / "DEPLOY.md"),
                }

            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "runtime" / "weights")),
            ), patch(
                "hydranet.moe_training.probe_lightning_import",
                side_effect=_fake_probe,
            ), patch(
                "hydranet.moe_training.build_routerset_moe",
                return_value=FakeModel(),
            ), patch(
                "hydranet.moe_training.capture_baseline_summary",
                side_effect=_fake_capture_baseline_summary,
            ), patch(
                "hydranet.moe_training._loading_module",
                return_value=_FakeLoadingModule(save_bundle_side_effect=_fake_save_bundle),
            ), patch(
                "hydranet.moe_training.write_routing_predictions",
                side_effect=_fake_write_predictions,
            ), patch(
                "hydranet.moe_training.create_phidranet_release",
                side_effect=_fake_create_release,
            ), patch(
                "hydranet.moe_training.load_lightning_training_components",
                return_value=(FakeLightningModule, lambda *args, **kwargs: FakeTrainer(), lambda *args, **kwargs: None),
            ):
                summary = train_switcher(
                    routerset_dir=root,
                    output_dir=output_dir,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    runtime_root=root / "runtime",
                    target_size=16,
                    target_channels=8,
                    release_name="demo",
                    startup_timeout_seconds=5,
                    max_epochs=1,
                )

            self.assertEqual(observed_events, ["startup_gate:5", "prepare", "fit", "export"])
            startup_log_entries = [
                json.loads(line)
                for line in (output_dir / "startup_log.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            stage_order = [entry["stage"] for entry in startup_log_entries]
            self.assertLess(stage_order.index("dataset_report_written"), stage_order.index("startup_gate_started"))
            self.assertLess(stage_order.index("startup_gate_completed"), stage_order.index("fit_started"))
            self.assertLess(stage_order.index("preflight_report_written"), stage_order.index("fit_started"))
            self.assertLess(stage_order.index("fit_started"), stage_order.index("fit_completed"))
            self.assertEqual(summary["contract"]["completed_stages"], ["preflight", "startup_gate", "training", "export"])
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "startup_gate.json").exists())
            self.assertTrue((output_dir / "preflight_report.json").exists())
            self.assertTrue((output_dir / "student_moe_bundle.pt").exists())
            self.assertTrue((release_dir / "release_manifest.json").exists())

    def test_train_switcher_failure_after_fit_started_records_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"

            class FakeLightningModule:
                def __init__(self, model, *, learning_rate: float, weight_decay: float) -> None:
                    self.model = model
                    self.learning_rate = learning_rate
                    self.weight_decay = weight_decay

            class FakeTrainer:
                callback_metrics = {}

                def fit(self, lightning_module, train_dataloaders=None, val_dataloaders=None) -> None:
                    _ = (lightning_module, train_dataloaders, val_dataloaders)
                    raise RuntimeError("fit exploded")

            class FakeModel:
                encoder_source_task = "fire"

            def _fake_baseline_summary(out: Path) -> Path:
                path = Path(out) / "baseline_summary.json"
                path.write_text("{}", encoding="utf-8")
                return path

            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "weights")),
            ), patch("hydranet.moe_training.probe_lightning_import", return_value=None), patch(
                "hydranet.moe_training.build_routerset_moe",
                return_value=FakeModel(),
            ), patch(
                "hydranet.moe_training.capture_baseline_summary",
                side_effect=lambda model, out: _fake_baseline_summary(out),
            ), patch(
                "hydranet.moe_training.load_lightning_training_components",
                return_value=(FakeLightningModule, lambda *args, **kwargs: FakeTrainer(), lambda *args, **kwargs: None),
            ):
                with self.assertRaisesRegex(RuntimeError, "fit exploded"):
                    train_switcher(
                        routerset_dir=root,
                        output_dir=output_dir,
                        expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                        runtime_root=root / "runtime",
                        target_size=16,
                        target_channels=8,
                        release_name="demo",
                        startup_timeout_seconds=1,
                        max_epochs=1,
                    )

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["failure_stage"], "fit_started")
            self.assertEqual(summary["startup_stage"], "fit_started")
            self.assertEqual(summary["config_path"], str(output_dir / "config.json"))
            self.assertEqual(summary["startup_log_path"], str(output_dir / "startup_log.txt"))
            self.assertEqual(summary["startup_stage_path"], str(output_dir / "startup_stage.json"))

    def test_train_switcher_non_finite_loss_records_batch_diagnostics_and_blocks_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"

            class FakeLightningModule:
                def __init__(self, model, *, learning_rate: float, weight_decay: float) -> None:
                    self.model = model
                    self.learning_rate = learning_rate
                    self.weight_decay = weight_decay

            class NonFiniteBatchError(RuntimeError):
                def __init__(self) -> None:
                    super().__init__("Non-finite train loss detected")
                    self.diagnostics = {
                        "stage": "train",
                        "batch_index": 2,
                        "batch_size": 2,
                        "loss_value": "nan",
                        "sample_id": "0000001",
                        "sample_ids": ["0000001", "0000013"],
                        "expert_context": {
                            "sample_expert_names": ["fire", "fire"],
                            "active_experts": [["fire"], ["fire"]],
                        },
                    }

            class FakeTrainer:
                callback_metrics = {}

                def fit(self, lightning_module, train_dataloaders=None, val_dataloaders=None) -> None:
                    _ = (lightning_module, train_dataloaders, val_dataloaders)
                    raise NonFiniteBatchError()

            class FakeModel:
                encoder_source_task = "fire"

            def _fake_baseline_summary(out: Path) -> Path:
                path = Path(out) / "baseline_summary.json"
                path.write_text("{}", encoding="utf-8")
                return path

            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "weights")),
            ), patch("hydranet.moe_training.probe_lightning_import", return_value=None), patch(
                "hydranet.moe_training.build_routerset_moe",
                return_value=FakeModel(),
            ), patch(
                "hydranet.moe_training.capture_baseline_summary",
                side_effect=lambda model, out: _fake_baseline_summary(out),
            ), patch(
                "hydranet.moe_training._loading_module",
            ) as loading_module_mock, patch(
                "hydranet.moe_training.create_phidranet_release",
            ) as create_release_mock, patch(
                "hydranet.moe_training.load_lightning_training_components",
                return_value=(FakeLightningModule, lambda *args, **kwargs: FakeTrainer(), lambda *args, **kwargs: None),
            ):
                with self.assertRaisesRegex(NonFiniteBatchError, "Non-finite train loss detected"):
                    train_switcher(
                        routerset_dir=root,
                        output_dir=output_dir,
                        expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                        runtime_root=root / "runtime",
                        target_size=16,
                        target_channels=8,
                        release_name="demo",
                        startup_timeout_seconds=1,
                        max_epochs=1,
                    )

            loading_module_mock.assert_not_called()
            create_release_mock.assert_not_called()
            self.assertFalse((output_dir / "student_moe_bundle.pt").exists())

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            startup_log_entries = [
                json.loads(line)
                for line in (output_dir / "startup_log.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            non_finite_entry = next(entry for entry in startup_log_entries if entry["stage"] == "non_finite_loss_detected")

            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["failure_stage"], "fit_started")
            self.assertEqual(summary["error_type"], "NonFiniteBatchError")
            self.assertEqual(summary["non_finite_loss"]["batch_index"], 2)
            self.assertEqual(summary["non_finite_loss"]["sample_id"], "0000001")
            self.assertEqual(summary["non_finite_loss"]["sample_ids"], ["0000001", "0000013"])
            self.assertEqual(summary["non_finite_loss"]["expert_context"]["sample_expert_names"], ["fire", "fire"])
            self.assertEqual(summary["non_finite_loss"]["config_snapshot"]["release_name"], "phidranet_demo")
            self.assertEqual(non_finite_entry["error_type"], "NonFiniteBatchError")
            self.assertEqual(non_finite_entry["failure_stage"], "fit_started")
            self.assertEqual(non_finite_entry["non_finite_loss"]["loss_value"], "nan")

    def test_train_switcher_skip_startup_gate_writes_skipped_gate_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"
            release_dir = output_dir / "bundle" / "phidranet_demo"

            class FakeLightningModule:
                def __init__(self, model, *, learning_rate: float, weight_decay: float) -> None:
                    self.model = model
                    self.learning_rate = learning_rate
                    self.weight_decay = weight_decay

            class FakeTrainer:
                callback_metrics = {"val_loss": 0.25}

                def fit(self, lightning_module, train_dataloaders=None, val_dataloaders=None) -> None:
                    _ = (lightning_module, train_dataloaders, val_dataloaders)

            class FakeModel:
                encoder_source_task = "fire"

            def _fake_capture_baseline_summary(model, output_dir) -> Path:
                _ = model
                path = Path(output_dir) / "baseline_summary.json"
                path.write_text("{}", encoding="utf-8")
                return path

            def _fake_save_bundle(model, output_path, metadata=None) -> Path:
                _ = (model, metadata)
                path = Path(output_path)
                path.write_bytes(b"bundle")
                return path

            def _fake_write_predictions(model, dataloader, output_path) -> Path:
                _ = (model, dataloader)
                path = Path(output_path)
                path.write_text("{}\n", encoding="utf-8")
                return path

            def _fake_create_release(**kwargs):
                _ = kwargs
                release_dir.mkdir(parents=True, exist_ok=True)
                for filename in [
                    "student_moe_bundle.pt",
                    "config.json",
                    "metrics.json",
                    "baseline_summary.json",
                    "routing_predictions.jsonl",
                    "dataset_report.json",
                    "checkpoint_report.json",
                    "release_manifest.json",
                    "DEPLOY.md",
                ]:
                    (release_dir / filename).write_text("{}", encoding="utf-8")
                return {
                    "release_dir": str(release_dir),
                    "bundle_path": str(release_dir / "student_moe_bundle.pt"),
                    "config_path": str(release_dir / "config.json"),
                    "metrics_path": str(release_dir / "metrics.json"),
                    "baseline_summary_path": str(release_dir / "baseline_summary.json"),
                    "routing_predictions_path": str(release_dir / "routing_predictions.jsonl"),
                    "dataset_report_path": str(release_dir / "dataset_report.json"),
                    "checkpoint_report_path": str(release_dir / "checkpoint_report.json"),
                    "release_manifest_path": str(release_dir / "release_manifest.json"),
                    "deployment_path": str(release_dir / "DEPLOY.md"),
                }

            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "weights")),
            ), patch("hydranet.moe_training.probe_lightning_import", return_value=None) as probe_mock, patch(
                "hydranet.moe_training.build_routerset_moe",
                return_value=FakeModel(),
            ), patch(
                "hydranet.moe_training.capture_baseline_summary",
                side_effect=_fake_capture_baseline_summary,
            ), patch(
                "hydranet.moe_training._loading_module",
                return_value=_FakeLoadingModule(save_bundle_side_effect=_fake_save_bundle),
            ), patch(
                "hydranet.moe_training.write_routing_predictions",
                side_effect=_fake_write_predictions,
            ), patch(
                "hydranet.moe_training.create_phidranet_release",
                side_effect=_fake_create_release,
            ), patch(
                "hydranet.moe_training.load_lightning_training_components",
                return_value=(FakeLightningModule, lambda *args, **kwargs: FakeTrainer(), lambda *args, **kwargs: None),
            ):
                summary = train_switcher(
                    routerset_dir=root,
                    output_dir=output_dir,
                    expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                    runtime_root=root / "runtime",
                    target_size=16,
                    target_channels=8,
                    release_name="demo",
                    run_startup_gate=False,
                    max_epochs=1,
                )

            gate_report = json.loads((output_dir / "startup_gate.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(gate_report["status"], "skipped")
            self.assertEqual(gate_report["timeout_seconds"], 120)
            self.assertEqual(gate_report["lightning_probe"]["status"], "skipped")
            self.assertEqual(probe_mock.call_count, 1)

    def test_train_switcher_startup_gate_timeout_records_startup_failed_before_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"

            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value=_checkpoint_report(weights_dir=str(root / "weights")),
            ), patch(
                "hydranet.moe_training.probe_lightning_import",
                side_effect=TimeoutError("Lightning import probe timed out before trainer startup completed."),
            ), patch("hydranet.moe_training.build_routerset_moe") as build_model_mock, patch(
                "hydranet.moe_training.load_lightning_training_components"
            ) as lightning_components_mock:
                with self.assertRaisesRegex(TimeoutError, "Lightning import probe timed out"):
                    train_switcher(
                        routerset_dir=root,
                        output_dir=output_dir,
                        expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                        runtime_root=root / "runtime",
                        target_size=16,
                        target_channels=8,
                        release_name="demo",
                        startup_timeout_seconds=3,
                        max_epochs=1,
                    )

            build_model_mock.assert_not_called()
            lightning_components_mock.assert_not_called()
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            gate_report = json.loads((output_dir / "startup_gate.json").read_text(encoding="utf-8"))
            startup_log_lines = (output_dir / "startup_log.txt").read_text(encoding="utf-8").splitlines()
            startup_log_entries = [json.loads(line) for line in startup_log_lines if line.strip()]

            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["failure_stage"], "startup_failed")
            self.assertEqual(summary["startup_stage"], "startup_failed")
            self.assertNotIn("fit_started", [entry["stage"] for entry in startup_log_entries])
            self.assertIn("startup_gate_started", [entry["stage"] for entry in startup_log_entries])
            self.assertIn("startup_gate_failed", [entry["stage"] for entry in startup_log_entries])
            self.assertEqual(gate_report["status"], "failed")
            self.assertEqual(gate_report["timeout_seconds"], 3)
            self.assertEqual(gate_report["error_type"], "TimeoutError")
            self.assertIn("TimeoutError", gate_report["traceback"])
            self.assertTrue(all("timestamp" in entry for entry in startup_log_entries))

    def test_train_switcher_missing_manifest_fails_before_long_running_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "out"
            with patch("hydranet.moe_training.build_routerset_moe") as build_mock, patch(
                "hydranet.moe_training.load_lightning_training_components"
            ) as lightning_mock:
                with self.assertRaises(FileNotFoundError):
                    train_switcher(
                        routerset_dir=root,
                        output_dir=output_dir,
                        manifest_path=root / "missing.jsonl",
                        expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                        runtime_root=root / "runtime",
                        target_size=16,
                        target_channels=8,
                        release_name="demo",
                    )

            build_mock.assert_not_called()
            lightning_mock.assert_not_called()
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertNotEqual(summary["failure_stage"], "fit_started")
            self.assertEqual(summary["manifest_path"], str(root / "missing.jsonl"))
            self.assertEqual(summary["contract"]["completed_stages"], [])

    def test_train_sampling_report_favors_minority_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            datamodule = RoutersetMoEDataModule(
                root,
                expert_names=list(DEFAULT_ROUTERSET_EXPERTS),
                batch_size=1,
                num_workers=0,
                target_size=16,
            )
            datamodule.setup("fit")
            sampling = datamodule.train_sampling_report()

            self.assertGreater(sampling["class_weights"]["fire"], sampling["class_weights"]["anomaly_detection"])
            self.assertGreater(sampling["class_weights"]["burned_area"], sampling["class_weights"]["anomaly_detection"])

    def test_run_moe_smoke_test_wires_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            smoke_dir = root / "smoke"
            release_dir = smoke_dir / "bundle"
            with patch(
                "hydranet.moe_training.preflight_routerset_training",
                return_value={"manifest_path": str(default_rebuilt_manifest_path(root))},
            ) as preflight_mock, patch(
                "hydranet.moe_training.train_switcher",
                return_value={
                    "bundle_path": str(root / "student_moe_bundle.pt"),
                    "release_dir": str(release_dir / "phidranet_smoke_v1"),
                },
            ) as train_mock, patch(
                "hydranet.moe_training.load_student_moe_bundle",
                return_value=object(),
            ) as load_mock, patch(
                "hydranet.moe_training.write_routing_predictions",
                side_effect=lambda model, dataloader, output_path: Path(output_path),
            ) as predict_mock:
                summary = run_moe_smoke_test(
                    routerset_dir=root,
                    output_dir=smoke_dir,
                    release_name="smoke_v1",
                )

            self.assertEqual(preflight_mock.call_args.kwargs["rebuild_splits"], True)
            self.assertEqual(train_mock.call_args.kwargs["max_epochs"], 1)
            self.assertEqual(train_mock.call_args.kwargs["manifest_path"], str(default_rebuilt_manifest_path(root)))
            self.assertTrue(load_mock.called)
            self.assertTrue(predict_mock.called)
            self.assertTrue((smoke_dir / "smoke_test_summary.json").exists())
            self.assertEqual(summary["rebuilt_manifest_path"], str(default_rebuilt_manifest_path(root)))

    def test_run_moe_smoke_test_writes_failure_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            smoke_dir = root / "smoke"
            with patch(
                "hydranet.moe_training.preflight_routerset_training",
                return_value={"manifest_path": str(default_rebuilt_manifest_path(root))},
            ), patch(
                "hydranet.moe_training.train_switcher",
                side_effect=TimeoutError("Lightning import probe timed out"),
            ):
                with self.assertRaises(TimeoutError):
                    run_moe_smoke_test(
                        routerset_dir=root,
                        output_dir=smoke_dir,
                        release_name="smoke_v1",
                    )

            summary = json.loads((smoke_dir / "smoke_test_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["error_type"], "TimeoutError")

    def test_run_full_training_marks_smoke_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"
            release_dir = output_dir / "bundle" / "phidranet_demo"
            summary_path = output_dir / "summary.json"

            def _fake_train_switcher(**kwargs):
                _ = kwargs
                output_dir.mkdir(parents=True, exist_ok=True)
                summary = {
                    "status": "completed",
                    "startup_stage": "fit_completed",
                    "failure_stage": None,
                    "routerset_dir": str(root),
                    "dataset_root": str(resolve_routerset_dataset_root(root)),
                    "manifest_path": str(default_rebuilt_manifest_path(root)),
                    "runtime_root": str(root / "runtime"),
                    "runtime_environment_path": str(output_dir / "runtime_environment.json"),
                    "preflight_report_path": str(output_dir / "preflight_report.json"),
                    "bundle_path": str(output_dir / "student_moe_bundle.pt"),
                    "metrics_path": str(output_dir / "metrics.json"),
                    "prediction_path": str(output_dir / "routing_predictions.jsonl"),
                    "config_path": str(output_dir / "config.json"),
                    "dataset_report_path": str(output_dir / "dataset_report.json"),
                    "checkpoint_report_path": str(output_dir / "checkpoint_report.json"),
                    "startup_log_path": str(output_dir / "startup_log.txt"),
                    "startup_stage_path": str(output_dir / "startup_stage.json"),
                    "startup_gate_path": str(output_dir / "startup_gate.json"),
                    "release_dir": str(release_dir),
                    "release_manifest_path": str(release_dir / "release_manifest.json"),
                    "contract": {"completed_stages": ["preflight", "startup_gate", "training", "export"]},
                }
                for path in [
                    output_dir / "runtime_environment.json",
                    output_dir / "preflight_report.json",
                    output_dir / "config.json",
                    output_dir / "dataset_report.json",
                    output_dir / "checkpoint_report.json",
                    output_dir / "metrics.json",
                    output_dir / "startup_log.txt",
                    output_dir / "startup_stage.json",
                    output_dir / "startup_gate.json",
                    output_dir / "routing_predictions.jsonl",
                    output_dir / "student_moe_bundle.pt",
                ]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix == ".pt":
                        path.write_bytes(b"bundle")
                    else:
                        path.write_text("{}", encoding="utf-8")
                release_dir.mkdir(parents=True, exist_ok=True)
                (release_dir / "release_manifest.json").write_text("{}", encoding="utf-8")
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                return summary

            with patch("hydranet.moe_training.train_switcher", side_effect=_fake_train_switcher), patch(
                "hydranet.moe_training.run_exported_moe_inference",
                return_value={
                    "summary_path": str(output_dir / "inference" / "summary.json"),
                    "prediction_path": str(output_dir / "inference" / "routing_predictions.jsonl"),
                },
            ):
                summary = run_full_training(
                    routerset_dir=root,
                    output_dir=output_dir,
                    release_name="demo",
                )

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["contract"]["completed_stages"], list(FULL_TRAINING_STAGES))
            self.assertIsNone(summary["contract"]["next_stage"])
            self.assertEqual(summary["smoke_test_summary_path"], str(output_dir / "smoke_test_summary.json"))
            self.assertTrue((output_dir / "smoke_test_summary.json").exists())
            written = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(written["contract"]["completed_stages"], list(FULL_TRAINING_STAGES))

    def test_run_full_training_records_smoke_failure_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"
            release_dir = output_dir / "bundle" / "phidranet_demo"
            summary_path = output_dir / "summary.json"

            def _fake_train_switcher(**kwargs):
                _ = kwargs
                output_dir.mkdir(parents=True, exist_ok=True)
                summary = {
                    "status": "completed",
                    "startup_stage": "fit_completed",
                    "failure_stage": None,
                    "routerset_dir": str(root),
                    "dataset_root": str(resolve_routerset_dataset_root(root)),
                    "manifest_path": str(default_rebuilt_manifest_path(root)),
                    "runtime_root": str(root / "runtime"),
                    "runtime_environment_path": str(output_dir / "runtime_environment.json"),
                    "preflight_report_path": str(output_dir / "preflight_report.json"),
                    "bundle_path": str(output_dir / "student_moe_bundle.pt"),
                    "metrics_path": str(output_dir / "metrics.json"),
                    "prediction_path": str(output_dir / "routing_predictions.jsonl"),
                    "config_path": str(output_dir / "config.json"),
                    "dataset_report_path": str(output_dir / "dataset_report.json"),
                    "checkpoint_report_path": str(output_dir / "checkpoint_report.json"),
                    "startup_log_path": str(output_dir / "startup_log.txt"),
                    "startup_stage_path": str(output_dir / "startup_stage.json"),
                    "startup_gate_path": str(output_dir / "startup_gate.json"),
                    "release_dir": str(release_dir),
                    "release_manifest_path": str(release_dir / "release_manifest.json"),
                    "contract": {"completed_stages": ["preflight", "startup_gate", "training", "export"]},
                }
                for path in [
                    output_dir / "runtime_environment.json",
                    output_dir / "preflight_report.json",
                    output_dir / "config.json",
                    output_dir / "dataset_report.json",
                    output_dir / "checkpoint_report.json",
                    output_dir / "metrics.json",
                    output_dir / "startup_log.txt",
                    output_dir / "startup_stage.json",
                    output_dir / "startup_gate.json",
                    output_dir / "student_moe_bundle.pt",
                ]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.suffix == ".pt":
                        path.write_bytes(b"bundle")
                    else:
                        path.write_text("{}", encoding="utf-8")
                release_dir.mkdir(parents=True, exist_ok=True)
                (release_dir / "release_manifest.json").write_text("{}", encoding="utf-8")
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                return summary

            with patch("hydranet.moe_training.train_switcher", side_effect=_fake_train_switcher), patch(
                "hydranet.moe_training.run_exported_moe_inference",
                side_effect=RuntimeError("smoke inference exploded"),
            ):
                with self.assertRaisesRegex(RuntimeError, "smoke inference exploded"):
                    run_full_training(
                        routerset_dir=root,
                        output_dir=output_dir,
                        release_name="demo",
                    )

            written = json.loads(summary_path.read_text(encoding="utf-8"))
            smoke_summary = json.loads((output_dir / "smoke_test_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(written["status"], "failed")
            self.assertEqual(written["failure_stage"], "smoke")
            self.assertEqual(smoke_summary["failure_stage"], "smoke")
            self.assertEqual(smoke_summary["error_type"], "RuntimeError")

    def test_run_full_training_preserves_training_failure_stage_before_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            output_dir = root / "out"
            release_dir = output_dir / "bundle" / "phidranet_demo"
            summary_path = output_dir / "summary.json"

            failed_training_summary = {
                "status": "failed",
                "startup_stage": "startup_failed",
                "failure_stage": "startup_failed",
                "routerset_dir": str(root),
                "dataset_root": str(resolve_routerset_dataset_root(root)),
                "manifest_path": str(default_rebuilt_manifest_path(root)),
                "runtime_root": str(root / "runtime"),
                "runtime_environment_path": str(output_dir / "runtime_environment.json"),
                "preflight_report_path": str(output_dir / "preflight_report.json"),
                "bundle_path": "",
                "metrics_path": "",
                "prediction_path": "",
                "config_path": str(output_dir / "config.json"),
                "dataset_report_path": str(output_dir / "dataset_report.json"),
                "checkpoint_report_path": str(output_dir / "checkpoint_report.json"),
                "startup_log_path": str(output_dir / "startup_log.txt"),
                "startup_stage_path": str(output_dir / "startup_stage.json"),
                "startup_gate_path": str(output_dir / "startup_gate.json"),
                "release_dir": str(release_dir),
                "release_manifest_path": str(release_dir / "release_manifest.json"),
                "contract": {"completed_stages": ["preflight"]},
                "error_type": "TimeoutError",
                "error": "Lightning import probe timed out",
            }

            _write_mock_training_artifacts(output_dir, release_dir)
            (output_dir / "student_moe_bundle.pt").unlink()
            (output_dir / "routing_predictions.jsonl").unlink()
            (output_dir / "metrics.json").unlink()
            summary_path.write_text(json.dumps(failed_training_summary), encoding="utf-8")

            with patch(
                "hydranet.moe_training.train_switcher",
                side_effect=TimeoutError("Lightning import probe timed out"),
            ):
                with self.assertRaisesRegex(TimeoutError, "timed out"):
                    run_full_training(
                        routerset_dir=root,
                        output_dir=output_dir,
                        release_name="demo",
                    )

            smoke_summary = json.loads((output_dir / "smoke_test_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(smoke_summary["status"], "failed")
            self.assertEqual(smoke_summary["failure_stage"], "startup_failed")
            self.assertEqual(smoke_summary["training_summary_path"], str(summary_path))
            self.assertEqual(smoke_summary["startup_stage_path"], str(output_dir / "startup_stage.json"))
            self.assertEqual(smoke_summary["startup_log_path"], str(output_dir / "startup_log.txt"))
            self.assertEqual(smoke_summary["manifest_path"], str(default_rebuilt_manifest_path(root)))
            self.assertEqual(smoke_summary["bundle_path"], "")
            self.assertEqual(smoke_summary["error_type"], "TimeoutError")

    def test_full_train_cli_requires_output_dir_and_release_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "full_train_moe.py"
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("--output-dir", result.stderr)
            self.assertIn("--release-name", result.stderr)
            self.assertEqual(list(root.iterdir()), [])

    def test_public_package_imports_finish_for_cli_bootstrap(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "src")

        result = subprocess.run(
            [sys.executable, "-c", "import hydranet; import hydranet.moe_training; print('ok')"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("ok", result.stdout)

    def test_lazy_lightning_proxy_uses_loader(self) -> None:
        models = {name: create_phisatnet("checkpoint", n_classes=1) for name in DEFAULT_ROUTERSET_EXPERTS}
        moe_model = build_moe_student_from_models(models, threshold=0.5, top_k=1)

        class FakeLightningModule:
            def __init__(self, model, *, learning_rate: float, weight_decay: float) -> None:
                self.model = model
                self.learning_rate = learning_rate
                self.weight_decay = weight_decay

        with patch(
            "hydranet.moe_training.load_lightning_training_components",
            return_value=(FakeLightningModule, object(), object()),
        ):
            lightning_module = MoESwitcherLightningModule(moe_model, learning_rate=1e-3, weight_decay=0.0)

        self.assertIsInstance(lightning_module, FakeLightningModule)
        self.assertIs(lightning_module.model, moe_model)

    def test_probe_lightning_import_translates_timeout(self) -> None:
        with patch(
            "hydranet.moe_training.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=1),
        ):
            with self.assertRaises(TimeoutError):
                probe_lightning_import(timeout_seconds=1)

    def test_moe_training_source_has_no_top_level_lightning_import(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src/hydranet/moe_training.py").read_text(encoding="utf-8")
        self.assertNotIn("import pytorch_lightning as pl", source)
        self.assertNotIn("from pytorch_lightning.callbacks import EarlyStopping", source)
        self.assertIn("load_lightning_training_components", source)

    def test_moe_lightning_source_does_not_mask_cuda_devices(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src/hydranet/moe_lightning.py").read_text(encoding="utf-8")
        self.assertNotIn('CUDA_VISIBLE_DEVICES', source)

    def test_release_helpers_create_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "run"
            release_dir = output_dir / "bundle"
            release_dir.mkdir(parents=True, exist_ok=True)
            bundle = root / "student_moe_bundle.pt"
            bundle.write_bytes(b"bundle")
            run_dir = output_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            for filename in ["config.json", "metrics.json", "baseline_summary.json"]:
                (run_dir / filename).write_text("{}", encoding="utf-8")
            prediction = root / "routing_predictions.jsonl"
            prediction.write_text("{}", encoding="utf-8")

            models = {name: create_phisatnet("checkpoint", n_classes=1) for name in DEFAULT_ROUTERSET_EXPERTS}
            moe_model = build_moe_student_from_models(models)
            release = create_phidranet_release(
                output_dir=output_dir,
                release_root=release_dir,
                release_name="phidranet_demo",
                model=moe_model,
                config={"release_name": "phidranet_demo", "target_channels": 8, "target_size": 16},
                dataset_report={"train": {}, "validation": {}},
                checkpoint_report={"fire": {"checkpoint_path": "x"}},
                run_output_dir=run_dir,
                bundle_path=bundle,
                prediction_path=prediction,
            )

            self.assertTrue(Path(release["release_manifest_path"]).exists())
            self.assertTrue(Path(release["deployment_path"]).exists())
            self.assertTrue(Path(release["bundle_path"]).exists())
            deployment = write_deployment_readme(release_dir / "phidranet_demo", release_name="phidranet_demo")
            self.assertTrue(deployment.exists())

    def test_artifact_layout_rejects_release_root_outside_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "out"
            runtime_root = root / "runtime"

            with self.assertRaisesRegex(ValueError, "release_root must stay under"):
                build_artifact_layout(
                    output_dir=output_dir,
                    runtime_root=runtime_root,
                    release_name="phidranet_demo",
                    release_root=root / "elsewhere",
                )


if __name__ == "__main__":
    unittest.main()
