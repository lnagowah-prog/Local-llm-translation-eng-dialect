#!/usr/bin/env python3
"""Normalize Venetian dataset records with a configurable profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import read_jsonl, write_jsonl
from normalization import DEFAULT_CONFIG_PATH, VenetianNormalizer, load_profile, normalize_records


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="Input JSONL dataset.")
    parser.add_argument("--output-file", required=True, help="Output JSONL dataset.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Normalization profile JSON file.",
    )
    parser.add_argument("--report-file", help="Optional JSON report path.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    profile = load_profile(args.config)
    normalizer = VenetianNormalizer(profile)
    records = read_jsonl(args.input_file)
    normalized_records, stats = normalize_records(records, normalizer)

    write_jsonl(normalized_records, args.output_file)

    if args.report_file:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"output_file": args.output_file, **stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
