from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import numpy as np

from hydranet.moe_training import (
    DEFAULT_ROUTERSET_EXPERTS,
    FULL_TRAINING_STAGES,
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
                return_value={"fire": {"checkpoint_path": "dummy.pt", "training": "finetuning", "n_shots": 5000}},
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
            self.assertTrue((root / "out" / "preflight_report.json").exists())

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

    def test_preflight_rebuilds_broken_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root, broken_fire_validation=True)
            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value={"fire": {"checkpoint_path": "dummy.pt", "training": "finetuning", "n_shots": 5000}},
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

    def test_configure_local_runtime_environment_writes_local_cache_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = configure_local_runtime_environment(
                output_dir=root / "out",
                runtime_root=root / "runtime",
                weights_dir=root / "weights",
            )

            self.assertEqual(report["runtime_root"], str(root / "runtime"))
            self.assertEqual(report["weights_dir"], str(root / "weights"))
            self.assertTrue((root / "runtime" / "cache" / "huggingface").exists())
            self.assertEqual(report["env"]["HF_HUB_DISABLE_XET"], "1")

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
            self.assertTrue((output_dir / "startup_gate.json").exists())
            self.assertEqual(tuple(report["sample_image_shape"]), (8, 16, 16))

    def test_prepare_routerset_training_writes_prepare_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            with patch(
                "hydranet.moe_training.resolve_student_checkpoint_report",
                return_value={"fire": {"checkpoint_path": "dummy.pt", "training": "finetuning", "n_shots": 5000}},
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
            release_dir = root / "release" / "phidranet_demo"

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
                return_value={"fire": {"checkpoint_path": "dummy.pt", "training": "finetuning", "n_shots": 5000}},
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
            self.assertTrue((output_dir / "summary.json").exists())

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
                return_value={"fire": {"checkpoint_path": "dummy.pt", "training": "finetuning", "n_shots": 5000}},
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
            release_dir = root / "release"
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
            release_dir = root / "release"
            release_dir.mkdir(parents=True, exist_ok=True)
            bundle = root / "student_moe_bundle.pt"
            bundle.write_bytes(b"bundle")
            run_dir = root / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            for filename in ["config.json", "metrics.json", "baseline_summary.json"]:
                (run_dir / filename).write_text("{}", encoding="utf-8")
            prediction = root / "routing_predictions.jsonl"
            prediction.write_text("{}", encoding="utf-8")

            models = {name: create_phisatnet("checkpoint", n_classes=1) for name in DEFAULT_ROUTERSET_EXPERTS}
            moe_model = build_moe_student_from_models(models)
            release = create_phidranet_release(
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


if __name__ == "__main__":
    unittest.main()
