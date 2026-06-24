#!/usr/bin/env python3
"""Baseline 1: zero-shot translation with NLLB-200.

Translates the test set without any fine-tuning and saves one prediction
record per line to the output file. The same script can be reused after
fine-tuning by pointing --model at the local checkpoint:

    python scripts/baseline_nllb_zeroshot.py \\
        --model models/nllb_finetuned \\
        --output-file data/results/nllb_finetuned_predictions.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import read_jsonl, write_jsonl

MODEL_ID = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "vec_Latn"
DEFAULT_BATCH = 8
DEFAULT_MAX_LEN = 256
DEFAULT_BEAMS = 4


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--test-file", default="data/normalized/test.jsonl")
    p.add_argument(
        "--output-file", default="data/results/nllb_zeroshot_predictions.jsonl"
    )
    p.add_argument(
        "--model", default=MODEL_ID, help="HuggingFace model ID or local path"
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--num-beams", type=int, default=DEFAULT_BEAMS)
    return p


def translate_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    tgt_lang_id: int,
    max_length: int,
    num_beams: int,
    device: torch.device,
) -> list[str]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=max_length,
            num_beams=num_beams,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def main() -> None:
    args = build_argparser().parse_args()

    test_path = PROJECT_ROOT / args.test_file
    out_path = PROJECT_ROOT / args.output_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {args.model} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    tgt_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    records = read_jsonl(test_path)
    print(f"Translating {len(records)} sentences (batch size {args.batch_size}) ...")

    predictions: list[dict] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        hypotheses = translate_batch(
            [r["source_text"] for r in batch],
            tokenizer,
            model,
            tgt_lang_id,
            args.max_length,
            args.num_beams,
            device,
        )
        for record, hyp in zip(batch, hypotheses):
            predictions.append(
                {
                    "id": record["id"],
                    "source_text": record["source_text"],
                    "reference": record["target_text"],
                    "hypothesis": hyp,
                    "model": args.model,
                    "system": "nllb_zeroshot",
                }
            )
        done = min(start + args.batch_size, len(records))
        print(f"  {done}/{len(records)}", end="\r")

    print()
    write_jsonl(predictions, out_path)
    print(f"Saved {len(predictions)} predictions → {out_path}")


if __name__ == "__main__":
    main()
