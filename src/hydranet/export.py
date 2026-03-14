#!/usr/bin/env python3
"""ONNX and OpenVINO IR export helpers for PyTorch models."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np
import torch

try:
    import onnx
except ImportError:  # pragma: no cover - optional at import time
    onnx = None


OPENVINO_2020_3_MAX_OPSET = 11


class ExportConfig(Protocol):
    """Minimal configuration contract required by the exporter."""

    n_channels: int
    input_size: int


class ModelExporter:
    """Handle PyTorch -> ONNX -> OpenVINO IR export workflow."""

    def __init__(
        self,
        config: ExportConfig,
        experiment_dir: str | Path,
        logger: logging.Logger,
        onnx_dir: str | Path | None = None,
        openvino_dir: str | Path | None = None,
        host_mount_root: str | Path | None = None,
        container_mount_root: str | Path = "/home/mount",
        docker_image: str = "openvino/ubuntu18_dev:2020.3",
        opset_version: int = OPENVINO_2020_3_MAX_OPSET,
    ) -> None:
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.logger = logger
        self.onnx_dir = Path(onnx_dir) if onnx_dir is not None else self.experiment_dir / "onnx"
        self.openvino_dir = (
            Path(openvino_dir) if openvino_dir is not None else self.experiment_dir / "openvino"
        )
        self.host_mount_root = (
            Path(host_mount_root)
            if host_mount_root is not None
            else self.experiment_dir.resolve().parent
        )
        self.container_mount_root = Path(container_mount_root)
        self.docker_image = docker_image
        self.opset_version = opset_version

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        self.openvino_dir.mkdir(parents=True, exist_ok=True)

        self._validate_opset_version(self.opset_version)

    def _to_container_path(self, host_path: Path) -> str:
        """Map a host path into the container mount path."""
        try:
            rel_path = host_path.resolve().relative_to(self.host_mount_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Path '{host_path}' is outside host_mount_root '{self.host_mount_root}'."
            ) from exc
        return str(self.container_mount_root / rel_path)

    def _validate_opset_version(self, opset_version: int) -> None:
        """Reject ONNX opsets unsupported by OpenVINO 2020.3 conversion flow."""
        if opset_version < 1:
            raise ValueError(f"Invalid ONNX opset_version={opset_version}.")
        if opset_version > OPENVINO_2020_3_MAX_OPSET:
            raise ValueError(
                "OpenVINO 2020.3 export requires ONNX opset <= "
                f"{OPENVINO_2020_3_MAX_OPSET}. Received opset_version={opset_version}."
            )

    def _validate_onnx_model(self, onnx_path: Path) -> None:
        """Run a local ONNX checker pass and verify the exported opset."""
        if onnx is None:
            self.logger.warning("onnx is not installed; skipping exported graph validation.")
            return

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        opset_imports = [entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")]
        if not opset_imports:
            raise RuntimeError(f"Unable to determine ONNX opset for '{onnx_path}'.")

        exported_opset = max(opset_imports)
        if exported_opset > OPENVINO_2020_3_MAX_OPSET:
            raise RuntimeError(
                f"Exported ONNX opset {exported_opset} exceeds OpenVINO 2020.3 support."
            )
        self.logger.info("ONNX checker passed. Exported opset: %s", exported_opset)

    def export_model_to_onnx(
        self,
        model: torch.nn.Module,
        output_path: str,
        dummy_input: torch.Tensor,
        opset_version: int = 11,
    ) -> None:
        """
        Export a PyTorch model to ONNX format with diagnostics.

        Args:
            model: Model to export.
            output_path: Path to save the ONNX model.
            dummy_input: Example input tensor for tracing.
            opset_version: ONNX opset version.
        """
        self._validate_opset_version(opset_version)
        self.logger.info("Starting ONNX export with opset version %s", opset_version)
        self.logger.info("Model input shape: %s", tuple(dummy_input.shape))
        self.logger.info("Model input dtype: %s", dummy_input.dtype)

        model.eval()

        try:
            with torch.no_grad():
                test_output = model(dummy_input)
            self.logger.info("Forward pass successful. Output shape: %s", tuple(test_output.shape))
            self.logger.info("Output dtype: %s", test_output.dtype)
            self.logger.info(
                "Output value range: [%.4f, %.4f]",
                test_output.min().item(),
                test_output.max().item(),
            )
        except Exception as exc:
            raise RuntimeError(f"Forward pass failed: {exc}") from exc

        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=opset_version,
                do_constant_folding=True,
                keep_initializers_as_inputs=False,
                export_params=True,
                verbose=False,
                dynamic_axes=None,
            )
            self.logger.info("Model successfully exported to %s", output_path)
            self._validate_onnx_model(Path(output_path))

            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            self.logger.info("ONNX model size: %.2f MB", file_size_mb)
        except Exception as exc:
            raise RuntimeError(f"Failed to export model: {exc}") from exc

    def convert_to_onnx(self, model: torch.nn.Module) -> Tuple[Optional[str], float]:
        """Convert a PyTorch model to ONNX format."""
        self.logger.info("=== ONNX Conversion Phase ===")

        dummy_input = torch.randn(
            1,
            self.config.n_channels,
            self.config.input_size,
            self.config.input_size,
        )

        onnx_path = self.onnx_dir / "model.onnx"

        self.logger.info("=== Model Export Diagnostics ===")
        self.logger.info("PyTorch version: %s", torch.__version__)
        self.logger.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
        self.logger.info(
            "Model memory: %.2f MB",
            sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2,
        )

        start_time = time.perf_counter()
        try:
            self.export_model_to_onnx(
                model=model,
                dummy_input=dummy_input,
                output_path=str(onnx_path),
                opset_version=self.opset_version,
            )
        except Exception as exc:
            self.logger.error("ONNX export failed: %s", exc)
            return None, 0.0

        end_time = time.perf_counter()
        onnx_conversion_time = end_time - start_time

        sample_input_path = self.onnx_dir / "sample_input.npy"
        dummy_input_np = dummy_input.cpu().numpy()
        np.save(sample_input_path, dummy_input_np)

        self.logger.info(
            "Saved dummy input with shape %s to %s", dummy_input_np.shape, sample_input_path
        )
        self.logger.info("Input data type: %s", dummy_input_np.dtype)
        self.logger.info(
            "Input value range: [%.4f, %.4f]", dummy_input_np.min(), dummy_input_np.max()
        )

        return str(onnx_path), onnx_conversion_time

    def convert_to_openvino(self) -> Tuple[Optional[str], float]:
        """Convert an ONNX model to OpenVINO IR format in Docker."""
        self.logger.info("=== OpenVINO Conversion Phase ===")

        onnx_model_path = self.onnx_dir / "model.onnx"
        if not onnx_model_path.exists():
            self.logger.error("OpenVINO conversion skipped: missing ONNX model at %s", onnx_model_path)
            return None, 0.0
        if shutil.which("docker") is None:
            self.logger.error("OpenVINO conversion skipped: docker not available in PATH.")
            return None, 0.0

        convert_script_path = self.experiment_dir / "convert_openvino.sh"
        onnx_model_container_path = self._to_container_path(onnx_model_path)
        output_dir_container_path = self._to_container_path(self.openvino_dir)
        script_container_path = self._to_container_path(convert_script_path)

        mean_values = ",".join(["0"] * self.config.n_channels)
        scale_values = ",".join(["1"] * self.config.n_channels)

        convert_script_content = f"""#!/bin/bash
set -euo pipefail

# Source OpenVINO environment
source /opt/intel/openvino/bin/setupvars.sh

# OpenVINO Model Optimizer script for ONNX model
PYPATH=/opt/intel/openvino_2020.3.194/deployment_tools/model_optimizer/mo.py
ONNX_MODEL={onnx_model_container_path}
OUTPUT_DIR={output_dir_container_path}

echo "Converting ONNX model to OpenVINO IR format..."
echo "Input model: $ONNX_MODEL"
echo "Output directory: $OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

python3 $PYPATH \\
    --input_model "$ONNX_MODEL" \\
    --data_type FP16 \\
    --input input \\
    --input_shape "[1,{self.config.n_channels},{self.config.input_size},{self.config.input_size}]" \\
    --mean_values "[{mean_values}]" \\
    --scale_values "[{scale_values}]" \\
    --progress \\
    --stream_output \\
    --output_dir "$OUTPUT_DIR" \\
    --model_name model

echo "OpenVINO conversion completed!"
echo "Generated files in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR/"
"""

        with open(convert_script_path, "w", encoding="utf-8") as f:
            f.write(convert_script_content)

        os.chmod(convert_script_path, 0o755)

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--net=host",
            "--privileged",
            "-v",
            "/dev:/dev",
            "-u",
            "root",
            "-v",
            f"{self.host_mount_root}:{self.container_mount_root}/",
            "-w",
            str(self.container_mount_root),
            self.docker_image,
            "/bin/bash",
            script_container_path,
        ]

        self.logger.info("Starting OpenVINO conversion in Docker container...")

        start_time = time.perf_counter()
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            end_time = time.perf_counter()
            openvino_conversion_time = end_time - start_time

            if result.returncode == 0:
                self.logger.info(
                    "OpenVINO conversion completed in %.2f seconds", openvino_conversion_time
                )

                xml_path = self.openvino_dir / "model.xml"
                bin_path = self.openvino_dir / "model.bin"

                if xml_path.exists() and bin_path.exists():
                    xml_size_mb = os.path.getsize(xml_path) / (1024 * 1024)
                    bin_size_mb = os.path.getsize(bin_path) / (1024 * 1024)
                    self.logger.info("OpenVINO model files created:")
                    self.logger.info("  XML file: %s (%.2f MB)", xml_path, xml_size_mb)
                    self.logger.info("  BIN file: %s (%.2f MB)", bin_path, bin_size_mb)
                else:
                    self.logger.error("OpenVINO model files not found after conversion")
                    self.logger.error("Docker STDOUT: %s", result.stdout)
                    self.logger.error("Docker STDERR: %s", result.stderr)
                    return None, openvino_conversion_time
            else:
                self.logger.error(
                    "OpenVINO conversion failed with return code: %s", result.returncode
                )
                self.logger.error("STDOUT: %s", result.stdout)
                self.logger.error("STDERR: %s", result.stderr)
                return None, openvino_conversion_time

        except subprocess.TimeoutExpired:
            self.logger.error("OpenVINO conversion timed out")
            return None, 300.0
        except Exception as exc:
            self.logger.error("OpenVINO conversion failed: %s", exc)
            return None, 0.0

        return str(self.openvino_dir / "model.xml"), openvino_conversion_time
