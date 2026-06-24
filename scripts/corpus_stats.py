#!/usr/bin/env python3
"""Compute word and token statistics (min, max, mean) for English vs Venetian
across each split and the full dataset combined.

Token counts use the NLLB-200 sentencepiece tokenizer (no special tokens),
so they reflect actual subword fragmentation and validate the --max-length
setting used during training.

Usage:
    python scripts/corpus_stats.py
    python scripts/corpus_stats.py --data-dir data/normalized --max-length 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "facebook/nllb-200-distilled-600M"
SPLITS = ["train", "dev", "test"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def word_counts(records: list[dict], field: str) -> list[int]:
    return [len(r[field].split()) for r in records]


def token_counts(texts: list[str], tokenizer: AutoTokenizer) -> list[int]:
    batch_size = 128
    counts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)
        counts.extend(len(ids) for ids in encoded["input_ids"])
    return counts


def stats(counts: list[int]) -> dict:
    return {
        "min":   min(counts),
        "max":   max(counts),
        "mean":  round(sum(counts) / len(counts), 1),
        "total": sum(counts),
        "n":     len(counts),
    }


def print_section(
    label: str,
    en_words: list[int],
    vec_words: list[int],
    en_tokens: list[int],
    vec_tokens: list[int],
    max_length: int,
) -> None:
    ew = stats(en_words)
    vw = stats(vec_words)
    et = stats(en_tokens)
    vt = stats(vec_tokens)

    word_ratio  = round(vw["mean"] / ew["mean"], 3) if ew["mean"] else 0
    token_ratio = round(vt["mean"] / et["mean"], 3) if et["mean"] else 0

    en_over  = sum(1 for t in en_tokens  if t > max_length)
    vec_over = sum(1 for t in vec_tokens if t > max_length)

    n = ew["n"]
    w = 56

    print(f"\n{'─' * w}")
    print(f"  {label}  (n={n})")
    print(f"{'─' * w}")
    print(f"  {'':22}  {'English':>10}  {'Venetian':>10}")
    print(f"  {'--- WORDS ---':22}  {'':>10}  {'':>10}")
    print(f"  {'Min':22}  {ew['min']:>10}  {vw['min']:>10}")
    print(f"  {'Max':22}  {ew['max']:>10}  {vw['max']:>10}")
    print(f"  {'Mean':22}  {ew['mean']:>10}  {vw['mean']:>10}")
    print(f"  {'Vec/Eng ratio':22}  {word_ratio:>10}")
    print(f"  {'--- TOKENS (NLLB) ---':22}  {'':>10}  {'':>10}")
    print(f"  {'Min':22}  {et['min']:>10}  {vt['min']:>10}")
    print(f"  {'Max':22}  {et['max']:>10}  {vt['max']:>10}")
    print(f"  {'Mean':22}  {et['mean']:>10}  {vt['mean']:>10}")
    print(f"  {'Vec/Eng ratio':22}  {token_ratio:>10}")
    print(f"  {f'> {max_length} tokens':22}  {en_over:>10}  {vec_over:>10}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/normalized")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--max-length", type=int, default=256,
                   help="Training max-length threshold for truncation warning.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    data_dir = PROJECT_ROOT / args.data_dir

    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    all_records: list[dict] = []

    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"  ⚠  {path.name} not found — skipping")
            continue

        records = load_jsonl(path)
        all_records.extend(records)

        en_texts  = [r["source_text"] for r in records]
        vec_texts = [r["target_text"] for r in records]

        print_section(
            split.upper(),
            word_counts(records, "source_text"),
            word_counts(records, "target_text"),
            token_counts(en_texts,  tokenizer),
            token_counts(vec_texts, tokenizer),
            args.max_length,
        )

    if all_records:
        en_texts  = [r["source_text"] for r in all_records]
        vec_texts = [r["target_text"] for r in all_records]

        print_section(
            "ALL SPLITS COMBINED",
            word_counts(all_records, "source_text"),
            word_counts(all_records, "target_text"),
            token_counts(en_texts,  tokenizer),
            token_counts(vec_texts, tokenizer),
            args.max_length,
        )

    print()


if __name__ == "__main__":
    main()
