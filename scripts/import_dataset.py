#!/usr/bin/env python3
"""Import a dataset and create train/dev/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import (
    fixed_train_split,
    load_examples,
    renumber_records,
    stratified_split,
    summarize_dataset,
    write_jsonl,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        help="Path to a dataset file in Python `examples = [...]` or JSONL format.",
    )
    parser.add_argument(
        "--train-file",
        help="Path to the dataset used as the fixed training split.",
    )
    parser.add_argument(
        "--eval-file",
        help="Path to the dataset used as the dev/test sampling pool.",
    )
    parser.add_argument("--output-dir", default="data/processed", help="Directory for processed files.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Target train ratio.")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Target dev ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_file and (args.train_file or args.eval_file):
        raise ValueError("Use either --input-file or --train-file/--eval-file, not both.")

    if args.input_file:
        records = load_examples(args.input_file)
        train, dev, test = stratified_split(
            records,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
        )
        summary = {
            "split_mode": "single_source",
            "full": summarize_dataset(records),
            "train_size": len(train),
            "dev_size": len(dev),
            "test_size": len(test),
        }
    elif args.train_file and args.eval_file:
        train_source = load_examples(args.train_file)
        eval_source = load_examples(args.eval_file)
        train, dev, test = fixed_train_split(
            train_source,
            eval_source,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
        )
        train = renumber_records(train, start=1)
        dev = renumber_records(dev, start=len(train) + 1)
        test = renumber_records(test, start=len(train) + len(dev) + 1)
        records = train + dev + test
        total = len(records)
        summary = {
            "split_mode": "fixed_train_eval_pool",
            "full": summarize_dataset(records),
            "train_source_size": len(train_source),
            "eval_source_size": len(eval_source),
            "train_size": len(train),
            "dev_size": len(dev),
            "test_size": len(test),
            "actual_train_ratio": len(train) / total if total else 0.0,
            "actual_dev_ratio": len(dev) / total if total else 0.0,
            "actual_test_ratio": len(test) / total if total else 0.0,
        }
    else:
        raise ValueError("Provide either --input-file or both --train-file and --eval-file.")

    write_jsonl(records, output_dir / "full_dataset.jsonl")
    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(dev, output_dir / "dev.jsonl")
    write_jsonl(test, output_dir / "test.jsonl")
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
