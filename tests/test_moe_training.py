from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from hydranet.moe_training import (
    MoESwitcherLightningModule,
    RoutersetMoEDataModule,
    RoutersetMoEDataset,
    routerset_image_path,
    routerset_patch_token,
)
from hydranet.models.moe_student import build_moe_student_from_models
from hydranet.models.student import create_phisatnet


def _write_routerset_fixture(root: Path) -> None:
    rows = [
        {
            "dataset_path": "unused",
            "image_ref": "unused",
            "label_coverages": {},
            "label_names": ["active_fire"],
            "label_source": "native",
            "labels": [0],
            "native_label_names": ["active_fire"],
            "patch_height": None,
            "patch_width": None,
            "patch_x": 0,
            "patch_y": 0,
            "record_status": "positive",
            "selection_bucket": "fire",
            "source_dataset": "fire",
            "source_sample_id": "0000001",
            "source_split": "train",
            "source_storage_group": "test",
            "weak_label_names": [],
        },
        {
            "dataset_path": "unused",
            "image_ref": "unused",
            "label_coverages": {},
            "label_names": ["water"],
            "label_source": "native",
            "labels": [0],
            "native_label_names": ["water"],
            "patch_height": None,
            "patch_width": None,
            "patch_x": 0,
            "patch_y": 0,
            "record_status": "positive",
            "selection_bucket": "anomaly",
            "source_dataset": "anomaly_detection",
            "source_sample_id": "0000002",
            "source_split": "train",
            "source_storage_group": "test",
            "weak_label_names": [],
        },
        {
            "dataset_path": "unused",
            "image_ref": "unused",
            "label_coverages": {},
            "label_names": ["cloud"],
            "label_source": "native",
            "labels": [0],
            "native_label_names": ["cloud"],
            "patch_height": 16,
            "patch_width": 16,
            "patch_x": 0,
            "patch_y": 0,
            "record_status": "positive",
            "selection_bucket": "wf",
            "source_dataset": "worldfloods",
            "source_sample_id": "0000003",
            "source_split": "validation",
            "source_storage_group": "test",
            "weak_label_names": [],
        },
        {
            "dataset_path": "unused",
            "image_ref": "unused",
            "label_coverages": {},
            "label_names": ["road_present"],
            "label_source": "native",
            "labels": [0],
            "native_label_names": ["road_present"],
            "patch_height": 16,
            "patch_width": 16,
            "patch_x": 0,
            "patch_y": 0,
            "record_status": "positive",
            "selection_bucket": "roads",
            "source_dataset": "roads",
            "source_sample_id": "0000004",
            "source_split": "train",
            "source_storage_group": "test",
            "weak_label_names": [],
        },
    ]

    (root / "images").mkdir(parents=True, exist_ok=True)
    for row in rows:
        image_path = routerset_image_path(root, row)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        shape = (8, 16, 16) if row["source_dataset"] != "roads" else (10, 16, 16)
        np.save(image_path, np.random.randn(*shape).astype(np.float32))

    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


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
        self.assertTrue(str(routerset_image_path("routerset", row)).endswith("routerset/images/fire/train/0000010_0_0_full_full.npy"))

    def test_dataset_filters_routerset_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)
            dataset = RoutersetMoEDataset(root, expert_names=["anomaly_detection", "fire", "worldfloods"], split="train", target_size=16)

            self.assertEqual(len(dataset), 2)
            summary = dataset.summary()
            self.assertEqual(summary["counts_by_expert"]["fire"], 1)
            self.assertEqual(summary["counts_by_expert"]["anomaly_detection"], 1)
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (8, 16, 16))
            self.assertEqual(tuple(sample["target"].shape), (3,))

    def test_tiny_lightning_fit_and_bundle_ready_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_routerset_fixture(root)

            datamodule = RoutersetMoEDataModule(
                root,
                expert_names=["anomaly_detection", "fire", "worldfloods"],
                batch_size=1,
                num_workers=0,
                target_size=16,
            )
            models = {
                "anomaly_detection": create_phisatnet("checkpoint", n_classes=1),
                "fire": create_phisatnet("checkpoint", n_classes=1),
                "worldfloods": create_phisatnet("checkpoint", n_classes=1),
            }
            moe_model = build_moe_student_from_models(models, threshold=0.5, top_k=1)
            lightning_module = MoESwitcherLightningModule(moe_model, learning_rate=1e-3, weight_decay=0.0)

            trainer = pl.Trainer(
                default_root_dir=str(root / "trainer"),
                max_epochs=1,
                accelerator="cpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
            )
            trainer.fit(lightning_module, datamodule=datamodule)

            batch = next(iter(datamodule.val_dataloader()))
            outputs = moe_model(batch["image"])
            self.assertIn("routing_logits", outputs)
            self.assertTrue(outputs["expert_outputs"])


if __name__ == "__main__":
    unittest.main()
