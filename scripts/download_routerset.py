#!/usr/bin/env python3
"""Download the canonical routerset dataset snapshot from Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="sirbastiano94/routerset")
    parser.add_argument("--local-dir", default="routerset")
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional glob pattern to limit downloaded files. Repeat to provide multiple patterns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=args.allow_pattern,
        resume_download=True,
    )

    dataset_root = local_dir / "multilabel_dataset"
    manifest_path = dataset_root / "manifest.jsonl"
    summary = {
        "repo_id": args.repo_id,
        "local_dir": str(local_dir),
        "snapshot_path": str(snapshot_path),
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
    }
    if not manifest_path.exists():
        raise SystemExit(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
