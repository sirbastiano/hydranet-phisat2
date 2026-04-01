from __future__ import annotations

import unittest

import numpy as np

from hydranet.routerset_roads_anomaly import (
    build_roads_mosaic_record,
    derive_anomaly_payload,
    derive_roads_weak_payload,
    mosaic_roads_images,
    replace_dataset_rows,
)


class TestRoutersetRoadsAnomaly(unittest.TestCase):
    def test_mosaic_roads_images_assembles_2x2_grid(self) -> None:
        images = [
            np.full((128, 128, 10), fill_value=index, dtype=np.uint16)
            for index in range(4)
        ]
        mosaic = mosaic_roads_images(images)
        self.assertEqual(mosaic.shape, (256, 256, 10))
        self.assertTrue(np.all(mosaic[:128, :128] == 0))
        self.assertTrue(np.all(mosaic[:128, 128:] == 1))
        self.assertTrue(np.all(mosaic[128:, :128] == 2))
        self.assertTrue(np.all(mosaic[128:, 128:] == 3))

    def test_derive_roads_weak_payload_returns_expected_scene_tags(self) -> None:
        image = np.zeros((256, 256, 10), dtype=np.uint16)
        image[..., 1] = 600  # green
        image[..., 3] = 1400  # nir
        image[64:128, 64:128, 1] = 2000  # green
        image[64:128, 64:128, 3] = 500  # nir
        image[:64, :64, 0] = 5000
        image[:64, :64, 1] = 5000
        image[:64, :64, 2] = 5000
        labels, coverages = derive_roads_weak_payload(image)
        self.assertIn("land", labels)
        self.assertIn("water", labels)
        self.assertIn("cloud", labels)
        self.assertGreaterEqual(coverages["water"], 0.05)
        self.assertGreaterEqual(coverages["cloud"], 0.05)

    def test_derive_anomaly_payload_uses_scene_thresholds(self) -> None:
        label_patch = np.zeros((1, 256, 256), dtype=np.float32)
        label_patch[:, :128, :128] = 1  # water
        label_patch[:, :64, 128:256] = 8  # cloud
        labels, coverages, status = derive_anomaly_payload(label_patch)
        self.assertEqual(status, "positive")
        self.assertEqual(labels, ["cloud", "water"])
        self.assertGreater(coverages["water"], coverages["cloud"])

    def test_derive_anomaly_payload_rejects_non_integral_masks(self) -> None:
        label_patch = np.zeros((256, 256), dtype=np.float32)
        label_patch[0, 0] = 1.5
        with self.assertRaisesRegex(ValueError, "integral class ids"):
            derive_anomaly_payload(label_patch)

    def test_replace_dataset_rows_swaps_only_target_dataset(self) -> None:
        rows = [
            {"source_dataset": "roads", "source_split": "train", "source_sample_id": "a", "patch_y": 0, "patch_x": 0},
            {"source_dataset": "fire", "source_split": "train", "source_sample_id": "b", "patch_y": 0, "patch_x": 0},
        ]
        replacements = [
            {"source_dataset": "roads", "source_split": "train", "source_sample_id": "mosaic", "patch_y": 0, "patch_x": 0},
        ]
        updated = replace_dataset_rows(rows, dataset="roads", replacements=replacements)
        self.assertEqual([row["source_dataset"] for row in updated], ["fire", "roads"])
        self.assertEqual(updated[-1]["source_sample_id"], "mosaic")

    def test_build_roads_mosaic_record_preserves_native_and_weak_labels(self) -> None:
        rows = [
            {
                "source_split": "train",
                "dataset_path": "/tmp/downstream_datasets_nshot.zip",
                "label_coverages": {"road_present": 0.01},
            }
            for _ in range(4)
        ]
        record = build_roads_mosaic_record(
            rows,
            dataset_path="/tmp/downstream_datasets_nshot.zip",
            label_vocab=["cloud", "land", "water", "road_present"],
            mosaic_index=0,
            weak_label_names=["land", "cloud"],
            weak_coverages={"land": 0.8, "cloud": 0.1},
            moe_split="train",
        )
        self.assertEqual(record["patch_height"], 256)
        self.assertEqual(record["native_label_names"], ["road_present"])
        self.assertEqual(record["weak_label_names"], ["cloud", "land"])
        self.assertEqual(record["label_names"], ["cloud", "land", "road_present"])


if __name__ == "__main__":
    unittest.main()
