#!/usr/bin/env python3
"""Scrape candidate parallel sentences from Venetian and English Wikipedia.

Strategy: fetch lead sections of short stub articles that exist in both
vec.wikipedia.org and en.wikipedia.org. Short stubs (under MAX_CHARS
characters) are targeted because their lead paragraphs are most likely to
be near-direct translations rather than independently written content.

Output is a JSONL of CANDIDATES — every pair needs human review before
being added to the training corpus. The source_type field is set to
'wikipedia_candidate' to distinguish them from verified translations.

Usage:
    python scripts/scrape_wikipedia.py
    python scripts/scrape_wikipedia.py --pages 200 --output data/raw/wikipedia_candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VEC_API = "https://vec.wikipedia.org/w/api.php"
EN_API  = "https://en.wikipedia.org/w/api.php"

# Only keep articles whose Venetian intro is under this length —
# longer articles diverge too much from the English for easy alignment.
MAX_CHARS    = 800
MIN_CHARS    = 40    # skip redirect stubs / empty pages
DELAY        = 1.0   # seconds between API calls — Wikipedia enforces rate limits
MAX_RETRIES  = 4


def api_get(
    session: requests.Session,
    url: str,
    params: dict,
    max_retries: int = MAX_RETRIES,
) -> requests.Response:
    """GET with exponential backoff on 429 / 5xx, honouring Retry-After."""
    for attempt in range(max_retries):
        r = session.get(url, params=params, timeout=15)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2 ** (attempt + 1)))
            print(f"\n    rate-limited — waiting {wait}s ...", end="  ")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r  # unreachable


def get_random_vec_titles(n: int, session: requests.Session) -> list[str]:
    """Return up to n random Venetian Wikipedia article titles."""
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": min(n, 500),
        "format": "json",
    }
    r = api_get(session, VEC_API, params)
    return [p["title"] for p in r.json()["query"]["random"]]


def get_english_title(vec_title: str, session: requests.Session) -> str | None:
    """Return the English Wikipedia title linked from a Venetian article."""
    params = {
        "action": "query",
        "titles": vec_title,
        "prop": "langlinks",
        "lllang": "en",
        "format": "json",
    }
    r = api_get(session, VEC_API, params)
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    langlinks = page.get("langlinks", [])
    return langlinks[0]["*"] if langlinks else None


def get_intro(api_url: str, title: str, session: requests.Session) -> str:
    """Return the plain-text lead section of a Wikipedia article."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "format": "json",
    }
    r = api_get(session, api_url, params)
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "").strip()


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, keeping each sentence clean."""
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if len(s.strip()) > MIN_CHARS]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pages", type=int, default=100,
                   help="Number of random Venetian pages to fetch.")
    p.add_argument("--output", default="data/raw/wikipedia_candidates.jsonl")
    p.add_argument("--max-chars", type=int, default=MAX_CHARS,
                   help="Max Venetian intro length — longer articles are skipped.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_path = PROJECT_ROOT / args.output

    session = requests.Session()
    session.headers.update({"User-Agent": "llm-translation-eng-dialect/1.0 (research)"})

    print(f"Fetching {args.pages} random Venetian Wikipedia titles ...")
    titles = get_random_vec_titles(args.pages, session)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    skipped_no_en = 0
    skipped_too_long = 0
    skipped_empty = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for i, vec_title in enumerate(titles, start=1):
            print(f"  [{i:>3}/{len(titles)}] {vec_title!r}", end="  ")

            en_title = get_english_title(vec_title, session)
            time.sleep(DELAY)

            if not en_title:
                print("→ no English link, skip")
                skipped_no_en += 1
                continue

            vec_text = get_intro(VEC_API, vec_title, session)
            time.sleep(DELAY)

            if not vec_text or len(vec_text) < MIN_CHARS:
                print("→ empty Venetian intro, skip")
                skipped_empty += 1
                continue

            if len(vec_text) > args.max_chars:
                print(f"→ too long ({len(vec_text)} chars), skip")
                skipped_too_long += 1
                continue

            en_text = get_intro(EN_API, en_title, session)
            time.sleep(DELAY)

            if not en_text or len(en_text) < MIN_CHARS:
                print("→ empty English intro, skip")
                skipped_empty += 1
                continue

            vec_sentences = split_sentences(vec_text)
            en_sentences  = split_sentences(en_text)

            pairs = list(zip(en_sentences, vec_sentences))

            for j, (en_sent, vec_sent) in enumerate(pairs):
                record = {
                    "id": f"wiki_{i:04d}_{j:02d}",
                    "source_lang": "eng_Latn",
                    "target_lang": "vec_Latn",
                    "source_text": en_sent,
                    "target_text": vec_sent,
                    "domain": "wikipedia",
                    "dialect_label": "wikipedia_venetian",
                    "source_type": "wikipedia_candidate",
                    "en_article": en_title,
                    "vec_article": vec_title,
                    "en_intro": en_text,
                    "vec_intro": vec_text,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()

            total_pairs += len(pairs)
            print(f"→ {len(pairs)} pair(s) extracted  [total saved: {total_pairs}]")

    print(f"\nDone.")
    print(f"  Extracted  : {total_pairs} candidate pairs")
    print(f"  No EN link : {skipped_no_en}")
    print(f"  Too long   : {skipped_too_long}")
    print(f"  Empty      : {skipped_empty}")
    print(f"  Output     : {out_path}")
    print()
    print("Next step: open the output file and manually verify each pair before")
    print("adding to the training corpus. Discard pairs where the sentences are")
    print("not translations of each other (Wikipedia articles diverge a lot).")


if __name__ == "__main__":
    main()
