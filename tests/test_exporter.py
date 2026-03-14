from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from hydranet.export import ModelExporter


class _DummyExportConfig:
    n_channels = 3
    input_size = 224


class TestModelExporter(unittest.TestCase):
    def test_default_host_mount_root_is_experiment_parent(self) -> None:
        logger = logging.getLogger("hydranet.export.test")
        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "experiment"
            experiment_dir.mkdir(parents=True)

            exporter = ModelExporter(
                config=_DummyExportConfig(),
                experiment_dir=experiment_dir,
                logger=logger,
                onnx_dir=experiment_dir / "onnx",
                openvino_dir=experiment_dir / "openvino",
            )

            expected_mount_root = experiment_dir.resolve().parent
            self.assertEqual(exporter.host_mount_root, expected_mount_root)
            self.assertEqual(
                exporter._to_container_path(experiment_dir / "model.onnx"),
                str(exporter.container_mount_root / experiment_dir.name / "model.onnx"),
            )


if __name__ == "__main__":
    unittest.main()
