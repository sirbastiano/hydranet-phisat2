#!/usr/bin/env python3
"""Run bundled HydraNet student MoE inference and write routing reports."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--routerset-dir", default="routerset")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experts", nargs="+", default=list(DEFAULT_ROUTERSET_EXPERTS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--target-channels", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datetime import datetime

    from hydranet import load_student_moe_bundle
    from hydranet.moe_training import RoutersetMoEDataModule, save_json, write_routing_predictions

    output_dir = Path(args.output_dir or Path("outputs/moe") / f"infer_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_student_moe_bundle(args.bundle_path)
    datamodule = RoutersetMoEDataModule(
        args.routerset_dir,
        manifest_path=args.manifest_path,
        expert_names=args.experts,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=args.target_size,
        target_channels=args.target_channels,
    )
    datamodule.setup("validate")
    prediction_path = write_routing_predictions(
        model,
        datamodule.val_dataloader(),
        output_dir / "routing_predictions.jsonl",
    )
    summary = {
        "bundle_path": args.bundle_path,
        "prediction_path": str(prediction_path),
        "experts": args.experts,
    }
    save_json(output_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
