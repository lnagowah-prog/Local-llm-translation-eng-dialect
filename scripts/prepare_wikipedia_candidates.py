#!/usr/bin/env python3
"""Convert raw Wikipedia candidates into standard dataset records."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import normalize_record, read_jsonl, write_jsonl


DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "wikipedia_candidates.jsonl"
DEFAULT_REFERENCE_FILE = PROJECT_ROOT / "data" / "full_dataset.jsonl"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "wikipedia_candidates.jsonl"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE), help="Raw Wikipedia candidates JSONL.")
    parser.add_argument(
        "--reference-file",
        default=str(DEFAULT_REFERENCE_FILE),
        help="Existing dataset used to reserve already-taken ids.",
    )
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Standardized Wikipedia candidates JSONL.",
    )
    return parser


def extract_numeric_id(raw_id: str) -> int:
    if raw_id.isdigit():
        return int(raw_id)

    parts = raw_id.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)

    raise ValueError(f"Could not extract a numeric id from {raw_id!r}.")


def next_available_id(base_id: int, used_ids: set[str]) -> str:
    candidate = base_id
    while f"{candidate:04d}" in used_ids:
        candidate += 1
    normalized_id = f"{candidate:04d}"
    used_ids.add(normalized_id)
    return normalized_id


def standardize_record(record: dict, used_ids: set[str]) -> dict:
    standardized = dict(record)
    standardized.update(normalize_record(record))
    standardized["id"] = next_available_id(extract_numeric_id(str(record.get("id", ""))), used_ids)
    return standardized


def main() -> None:
    args = build_argparser().parse_args()
    raw_records = read_jsonl(args.input_file)
    reference_records = read_jsonl(args.reference_file)
    used_ids = {str(record["id"]) for record in reference_records}

    standardized_records = [standardize_record(record, used_ids) for record in raw_records]
    write_jsonl(standardized_records, args.output_file)

    print(
        f"Prepared {len(standardized_records)} Wikipedia candidate rows -> {args.output_file} "
        f"(first id: {standardized_records[0]['id'] if standardized_records else 'n/a'}, "
        f"last id: {standardized_records[-1]['id'] if standardized_records else 'n/a'})"
    )


if __name__ == "__main__":
    main()
