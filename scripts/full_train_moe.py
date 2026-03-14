#!/usr/bin/env python3
"""Run the full HydraNet student MoE training contract in one invocation."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_RELEASE_ROOT = "outputs/phidranet"
DEFAULT_ROUTERSET_EXPERTS = (
    "anomaly_detection",
    "burned_area",
    "fire",
    "lc",
    "roads",
    "worldfloods",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routerset-dir", default="routerset")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--experts", nargs="+", default=list(DEFAULT_ROUTERSET_EXPERTS))
    parser.add_argument("--training", default="finetuning")
    parser.add_argument("--n-shots", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--target-channels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--release-root", default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--runtime-root", default=None)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--startup-timeout-seconds", type=int, default=20)
    parser.add_argument("--skip-startup-gate", action="store_true")
    parser.add_argument("--rebuild-splits", action="store_true")
    parser.add_argument("--rebuilt-manifest-out", default=None)
    parser.add_argument("--disable-balanced-sampling", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    from hydranet.moe_training import run_full_training

    summary = run_full_training(
        routerset_dir=args.routerset_dir,
        manifest_path=args.manifest_path,
        output_dir=output_dir,
        weights_dir=args.weights_dir,
        expert_names=args.experts,
        training=args.training,
        n_shots=args.n_shots,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        top_k=args.top_k,
        target_size=args.target_size,
        target_channels=args.target_channels,
        seed=args.seed,
        release_name=args.release_name,
        release_root=args.release_root,
        runtime_root=args.runtime_root,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        rebuild_splits=args.rebuild_splits,
        rebuilt_manifest_out=args.rebuilt_manifest_out,
        balanced_sampling=not args.disable_balanced_sampling,
        run_startup_gate=not args.skip_startup_gate,
        startup_timeout_seconds=args.startup_timeout_seconds,
    )
    print(summary)


if __name__ == "__main__":
    main()
