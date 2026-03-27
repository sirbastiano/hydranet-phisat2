#!/usr/bin/env python3
"""Materialize a canonical 8x256x256 routerset export as concrete .npy files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routerset-dir", default="routerset")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--output-dir", default="outputs/routerset/materialized_256")
    parser.add_argument(
        "--experts",
        nargs="+",
        default=[
            "anomaly_detection",
            "burned_area",
            "fire",
            "lc",
            "roads",
            "worldfloods",
        ],
    )
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--target-channels", type=int, default=8)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Exclude objective row-level artifact faults from the exported manifest and emit fault_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from hydranet.moe_training import materialize_routerset_dataset

    summary = materialize_routerset_dataset(
        routerset_dir=args.routerset_dir,
        output_dir=Path(args.output_dir),
        manifest_path=args.manifest_path,
        expert_names=args.experts,
        target_size=args.target_size,
        target_channels=args.target_channels,
        clean_export=args.clean,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
