#!/usr/bin/env python3
"""Baseline 3: zero-shot translation with an OpenAI chat model (gpt-5-mini).

Reads OPENAI_API_KEY from the environment or from a .env file at the repo root.
Translates every sentence in the test set sequentially, with automatic
retry on rate-limit errors, and writes one prediction record per line.

Usage:
    python scripts/baseline_llm_zeroshot.py
    python scripts/baseline_llm_zeroshot.py --model gpt-4o --delay 0.5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

import openai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import read_jsonl, write_jsonl

MODEL_ID = "gpt-5-mini"
DEFAULT_MAX_TOKENS = 256
DEFAULT_DELAY = 0.2  # seconds between requests; increase if hitting rate limits

SYSTEM_PROMPT = (
    "You are a professional translator specialising in the Venetian dialect "
    "of northeastern Italy (Veneto region). "
    "Translate the sentence the user sends into Venetian. "
    "Reply with only the translation — no explanations, no alternatives."
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--test-file", default="data/normalized/test.jsonl")
    p.add_argument(
        "--output-file", default="data/results/llm_zeroshot_predictions.jsonl"
    )
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Seconds to wait between API calls.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retry attempts on rate-limit errors (exponential backoff).",
    )
    return p


def translate_one(
    client: openai.OpenAI,
    text: str,
    model: str,
    max_tokens: int,
    max_retries: int,
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 1 s, 2 s, 4 s, 8 s …
            print(f"  Rate limit — waiting {wait}s before retry {attempt + 1}/{max_retries - 1} ...")
            time.sleep(wait)
        except openai.APIError as exc:
            raise RuntimeError(f"OpenAI API error on '{text[:60]}…': {exc}") from exc
    return ""  # unreachable, but satisfies type checker


def main() -> None:
    args = build_argparser().parse_args()

    # Load .env if dotenv is installed; OPENAI_API_KEY may already be set in env
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "Error: OPENAI_API_KEY is not set. "
            "Add it to your shell environment or to a .env file at the repo root."
        )

    test_path = PROJECT_ROOT / args.test_file
    out_path = PROJECT_ROOT / args.output_file

    client = openai.OpenAI(api_key=api_key)
    records = read_jsonl(test_path)

    print(f"Model : {args.model}")
    print(f"Input : {test_path}  ({len(records)} sentences)")

    predictions: list[dict] = []
    for i, record in enumerate(records, start=1):
        hyp = translate_one(
            client,
            record["source_text"],
            args.model,
            args.max_tokens,
            args.max_retries,
        )
        predictions.append(
            {
                "id": record["id"],
                "source_text": record["source_text"],
                "reference": record["target_text"],
                "hypothesis": hyp,
                "model": args.model,
                "system": "llm_zeroshot",
            }
        )
        print(f"  [{i:>3}/{len(records)}]  {record['source_text'][:55]!r}")
        print(f"           → {hyp[:70]!r}")
        if args.delay > 0:
            time.sleep(args.delay)

    write_jsonl(predictions, out_path)
    print(f"\nSaved {len(predictions)} predictions → {out_path}")


if __name__ == "__main__":
    main()
