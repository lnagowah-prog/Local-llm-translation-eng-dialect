#!/usr/bin/env python3
"""Rebuild the merged full dataset from the curated prefix and vec sentences."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_DATASET_PATH = PROJECT_ROOT / "data" / "full_dataset.jsonl"
VEC_SENTENCES_PATH = PROJECT_ROOT / "data" / "vec_sentences.jsonl"
SYNC_START_ID = "0201"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line.lstrip("\ufeff"))
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8-sig",
    )


def main() -> None:
    full_records = read_jsonl(FULL_DATASET_PATH)
    vec_records = read_jsonl(VEC_SENTENCES_PATH)

    prefix_records = [record for record in full_records if record["id"] < SYNC_START_ID]
    merged_records = prefix_records + vec_records

    if len({record["id"] for record in merged_records}) != len(merged_records):
        raise ValueError("Duplicate ids detected while rebuilding full_dataset.jsonl.")

    write_jsonl(FULL_DATASET_PATH, merged_records)
    print(
        f"Rebuilt {FULL_DATASET_PATH} with {len(prefix_records)} prefix rows "
        f"and {len(vec_records)} vec rows."
    )


if __name__ == "__main__":
    main()
