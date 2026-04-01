#!/usr/bin/env python3
"""Run a full per-file audit over the raw routerset dataset snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routerset-dir", default="routerset")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from hydranet.routerset_raw_audit import audit_raw_routerset_dataset

    summary = audit_raw_routerset_dataset(
        args.routerset_dir,
        output_dir=Path(args.output_dir) if args.output_dir is not None else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
