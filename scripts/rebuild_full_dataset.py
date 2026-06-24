#!/usr/bin/env python3
"""Rebuild the merged full dataset from the curated prefix and vec sentences."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import read_jsonl, write_jsonl

FULL_DATASET_PATH = PROJECT_ROOT / "data" / "full_dataset.jsonl"
STANDARD_VEC_SENTENCES_PATH = PROJECT_ROOT / "data" / "normalized" / "vec_sentences.jsonl"
STANDARD_WIKIPEDIA_CANDIDATES_PATH = PROJECT_ROOT / "data" / "wikipedia_candidates.jsonl"
SYNC_START_ID = "0201"


def main() -> None:
    full_records = read_jsonl(FULL_DATASET_PATH)
    standard_vec_records = read_jsonl(STANDARD_VEC_SENTENCES_PATH)
    standard_wikipedia_records = read_jsonl(STANDARD_WIKIPEDIA_CANDIDATES_PATH)

    prefix_records = [record for record in full_records if record["id"] < SYNC_START_ID]
    records = prefix_records + standard_vec_records + standard_wikipedia_records

    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Duplicate ids detected while rebuilding full_dataset.jsonl.")

    write_jsonl(records, FULL_DATASET_PATH)
    print(
        f"Rebuilt {FULL_DATASET_PATH} with {len(prefix_records)} prefix rows "
        f"{len(standard_vec_records)} standardized vec rows, and "
        f"{len(standard_wikipedia_records)} standardized Wikipedia rows."
    )


if __name__ == "__main__":
    main()
