from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import numpy as np

from hydranet.moe_training import (
    DEFAULT_ROUTERSET_EXPERTS,
    FULL_TRAINING_STAGES,
    build_artifact_layout,
    configure_local_runtime_environment,
    default_rebuilt_manifest_path,
    MoESwitcherLightningModule,
    prepare_routerset_training,
    RoutersetMoEDataModule,
    RoutersetMoEDataset,
    create_phidranet_release,
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


def _write_routerset_fixture(root: Path, *, broken_fire_validation: bool = False) -> None:
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
        else:
            arr = np.random.randn(8, 16, 16).astype(np.float32)
        np.save(image_path, arr)

        if broken_fire_validation and src == "fire" and row["source_split"] == "validation":
            row["record_status"] = "explicit_negative"

    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_text = "\n".join(json.dumps(row) for row in rows) + "\n"
    (dataset_root / "manifest.jsonl").write_text(manifest_text, encoding="utf-8")
    default_rebuilt_manifest_path(root).write_text(manifest_text, encoding="utf-8")


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

    def test_normalize_routerset_array_adapts_all_source_layouts(self) -> None:
        fire = normalize_routerset_array(np.random.randn(8, 16, 16).astype(np.float32), source_dataset="fire")
        burned_area = normalize_routerset_array(np.random.randn(7, 16, 16).astype(np.float32), source_dataset="burned_area")
        lc = normalize_routerset_array(np.random.randint(0, 10, size=(16, 16, 10), dtype=np.uint16), source_dataset="lc")
        roads = normalize_routerset_array(np.random.randint(0, 10, size=(16, 16, 10), dtype=np.uint16), source_dataset="roads")

        self.assertEqual(tuple(fire.shape), (8, 16, 16))
        self.assertEqual(tuple(burned_area.shape), (8, 16, 16))
        self.assertEqual(tuple(lc.shape), (8, 16, 16))
        self.assertEqual(tuple(roads.shape), (8, 16, 16))

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
            self.assertEqual(summary["manifest_path"], str(resolve_routerset_dataset_root(root) / "manifest.jsonl"))

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
                "hydranet.moe_training.save_student_moe_bundle",
                side_effect=_fake_save_bundle,
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
                "hydranet.moe_training.save_student_moe_bundle",
                side_effect=_fake_save_bundle,
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
                "hydranet.moe_training.save_student_moe_bundle",
            ) as save_bundle_mock, patch(
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

            save_bundle_mock.assert_not_called()
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
                "hydranet.moe_training.save_student_moe_bundle",
                side_effect=_fake_save_bundle,
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
            self.assertEqual(gate_report["timeout_seconds"], 60)
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
