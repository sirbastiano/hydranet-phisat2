#!/usr/bin/env python3
"""Train the HydraNet student MoE switcher on local routerset labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydranet.moe_training import DEFAULT_ROUTERSET_EXPERTS, train_switcher, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routerset-dir", default="routerset")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--experts", nargs="+", default=list(DEFAULT_ROUTERSET_EXPERTS))
    parser.add_argument("--training", default="finetuning")
    parser.add_argument("--n-shots", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or str(Path("outputs/moe") / utc_timestamp())
    summary = train_switcher(
        routerset_dir=args.routerset_dir,
        output_dir=output_dir,
        expert_names=args.experts,
        weights_dir=args.weights_dir,
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
        seed=args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main()
