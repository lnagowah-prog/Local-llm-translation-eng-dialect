"""Dataset loading and splitting helpers."""

from __future__ import annotations

import ast
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from constants import TASK_PREFIX, TEXT_FIELDS


def parse_python_examples(input_file: str | Path) -> list[dict]:
    """Parse a Python file that defines `examples = [...]`."""
    path = Path(input_file)
    module = ast.parse(path.read_text(encoding="utf-8"))

    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "examples":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, list):
                        raise ValueError("`examples` must be a list of records.")
                    return [normalize_record(record) for record in value]

    raise ValueError("Could not find `examples = [...]` in the input file.")


def parse_jsonl_examples(input_file: str | Path) -> list[dict]:
    """Parse a JSONL file containing one record per line."""
    path = Path(input_file)
    return [normalize_record(record) for record in read_jsonl(path)]


def load_examples(input_file: str | Path) -> list[dict]:
    """Load records from either a Python dataset file or a JSONL file."""
    path = Path(input_file)
    if path.suffix.lower() == ".jsonl":
        return parse_jsonl_examples(path)
    return parse_python_examples(path)


def normalize_record(record: dict) -> dict:
    """Keep only expected fields and normalize whitespace."""
    normalized = {}
    for field in TEXT_FIELDS:
        value = record.get(field, "")
        if isinstance(value, str):
            normalized[field] = " ".join(value.split())
        else:
            normalized[field] = value

    prompt = record.get("translation_prompt", "")
    if isinstance(prompt, str) and prompt.strip():
        normalized["translation_prompt"] = " ".join(prompt.split())
    else:
        normalized["translation_prompt"] = TASK_PREFIX + normalized["source_text"]
    return normalized


def stratified_split(
    records: list[dict],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records by domain while approximating 80/10/10 overall."""
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["domain"]].append(record)

    train: list[dict] = []
    dev: list[dict] = []
    test: list[dict] = []

    for domain in sorted(grouped):
        examples = list(grouped[domain])
        rng.shuffle(examples)
        size = len(examples)

        dev_count = round(size * dev_ratio)
        test_count = round(size * (1.0 - train_ratio - dev_ratio))

        if size >= 10:
            dev_count = max(1, dev_count)
            test_count = max(1, test_count)

        if dev_count + test_count >= size:
            overflow = dev_count + test_count - size + 1
            test_count = max(0, test_count - overflow)

        train_cutoff = size - dev_count - test_count
        dev_cutoff = train_cutoff + dev_count

        train.extend(examples[:train_cutoff])
        dev.extend(examples[train_cutoff:dev_cutoff])
        test.extend(examples[dev_cutoff:])

    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return train, dev, test


def fixed_train_split(
    train_records: list[dict],
    eval_records: list[dict],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Keep the training split fixed and sample dev/test from a separate pool."""
    test_ratio = 1.0 - train_ratio - dev_ratio
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if dev_ratio < 0.0 or test_ratio < 0.0:
        raise ValueError("dev_ratio and test_ratio must be non-negative.")

    rng = random.Random(seed)
    shuffled_eval = list(eval_records)
    rng.shuffle(shuffled_eval)

    dev_count = round(len(train_records) * dev_ratio / train_ratio)
    test_count = round(len(train_records) * test_ratio / train_ratio)

    if dev_count + test_count > len(shuffled_eval):
        raise ValueError(
            "Evaluation pool is too small for the requested split: "
            f"need {dev_count + test_count} records, found {len(shuffled_eval)}."
        )

    train = list(train_records)
    dev = shuffled_eval[:dev_count]
    test = shuffled_eval[dev_count : dev_count + test_count]
    return train, dev, test


def write_jsonl(records: Iterable[dict], output_file: str | Path) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def renumber_records(records: Iterable[dict], start: int = 1) -> list[dict]:
    renumbered = []
    for index, record in enumerate(records, start=start):
        updated = dict(record)
        updated["id"] = f"{index:04d}"
        renumbered.append(updated)
    return renumbered


def read_jsonl(input_file: str | Path) -> list[dict]:
    path = Path(input_file)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def summarize_dataset(records: list[dict]) -> dict:
    domain_counts = Counter(record["domain"] for record in records)
    return {
        "num_examples": len(records),
        "domains": dict(sorted(domain_counts.items())),
        "source_language_codes": sorted({record["source_lang"] for record in records}),
        "target_language_codes": sorted({record["target_lang"] for record in records}),
    }
