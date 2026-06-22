#!/usr/bin/env python3
"""Import the user's Python-format dataset and create train/dev/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import parse_python_examples, stratified_split, summarize_dataset, write_jsonl


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="Path to the pasted Python dataset file.")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for processed files.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = parse_python_examples(args.input_file)
    train, dev, test = stratified_split(records)
    summary = {
        "full": summarize_dataset(records),
        "train_size": len(train),
        "dev_size": len(dev),
        "test_size": len(test),
    }

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
