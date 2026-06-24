#!/usr/bin/env python3
"""Compute BLEU and chrF for all translation systems and print a results table.

Expects prediction files produced by the three baseline scripts:
    data/results/nllb_zeroshot_predictions.jsonl
    data/results/nllb_finetuned_predictions.jsonl
    data/results/llm_zeroshot_predictions.jsonl

Missing files are skipped with a warning. Human ratings (1–5) can be added
via --human-ratings so that the final saved JSON includes the full table.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py \\
        --human-ratings nllb_zeroshot=2.4 nllb_finetuned=3.5 llm_zeroshot=2.9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sacrebleu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import read_jsonl

DISPLAY_NAMES: dict[str, str] = {
    "nllb_zeroshot":  "NLLB zero-shot",
    "nllb_finetuned": "NLLB fine-tuned",
    "llm_zeroshot":   "LLM zero-shot",
}
_KNOWN_ORDER = list(DISPLAY_NAMES)


def system_label(system_id: str) -> str:
    return DISPLAY_NAMES.get(system_id, system_id.replace("_", " "))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results-dir",
        default="data/results",
        help="Directory containing the *_predictions.jsonl files.",
    )
    p.add_argument(
        "--output-file",
        default="data/results/evaluation_results.json",
    )
    p.add_argument(
        "--human-ratings",
        nargs="*",
        metavar="SYSTEM=SCORE",
        default=[],
        help='Optional human acceptability scores, e.g. nllb_zeroshot=2.4',
    )
    return p


def parse_human_ratings(pairs: list[str]) -> dict[str, float]:
    ratings: dict[str, float] = {}
    for pair in pairs:
        try:
            key, value = pair.split("=", 1)
            ratings[key.strip()] = float(value.strip())
        except ValueError:
            sys.exit(f"Error: --human-ratings entry {pair!r} must be in SYSTEM=SCORE format.")
    return ratings


def score_system(predictions: list[dict]) -> dict:
    hyps = [p["hypothesis"] for p in predictions]
    refs = [p["reference"] for p in predictions]

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])

    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "n": len(predictions),
        "human_rating": None,
    }


def print_table(results: dict[str, dict], ordered_systems: list[tuple[str, str]]) -> None:
    col_w = max((len(label) for _, label in ordered_systems), default=22)
    col_w = max(col_w, len("System"))
    print()
    print(f"{'System':<{col_w}}  {'BLEU':>6}  {'chrF':>6}  {'Human':>6}  {'n':>4}")
    print("─" * col_w + "  " + "─" * 6 + "  " + "─" * 6 + "  " + "─" * 6 + "  " + "─" * 4)
    for system_id, label in ordered_systems:
        if system_id not in results:
            continue
        r = results[system_id]
        human = f"{r['human_rating']:.1f}" if r["human_rating"] is not None else "—"
        print(
            f"{label:<{col_w}}  {r['bleu']:>6.2f}  {r['chrf']:>6.2f}  {human:>6}  {r['n']:>4}"
        )
    print()


def main() -> None:
    args = build_argparser().parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    out_path = PROJECT_ROOT / args.output_file
    human_ratings = parse_human_ratings(args.human_ratings)

    results: dict[str, dict] = {}

    pred_files = sorted(results_dir.glob("*_predictions.jsonl"))
    if not pred_files:
        sys.exit(f"No *_predictions.jsonl files found in {results_dir}. Run the baseline scripts first.")

    def _sort_key(path: Path) -> tuple:
        sid = path.stem.replace("_predictions", "")
        idx = _KNOWN_ORDER.index(sid) if sid in _KNOWN_ORDER else len(_KNOWN_ORDER)
        return (idx, sid)

    pred_files = sorted(pred_files, key=_sort_key)
    ordered_systems: list[tuple[str, str]] = []

    for pred_file in pred_files:
        system_id = pred_file.stem.replace("_predictions", "")
        label = system_label(system_id)

        predictions = read_jsonl(pred_file)
        if not predictions:
            print(f"  ⚠  {pred_file.name} is empty — skipping")
            continue

        scores = score_system(predictions)
        if system_id in human_ratings:
            scores["human_rating"] = human_ratings[system_id]

        results[system_id] = scores
        ordered_systems.append((system_id, label))
        print(f"  Scored {len(predictions)} sentences for {label}")

    if not results:
        sys.exit("All prediction files were empty.")

    print_table(results, ordered_systems)

    output = {
        "results_dir": str(results_dir),
        "systems": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved → {out_path}")

    if any(r["human_rating"] is None for r in results.values()):
        systems_str = " ".join(f"{sid}=?" for sid in results)
        print(
            "Tip: add human acceptability scores (1–5) with:\n"
            f"  python scripts/evaluate.py --human-ratings {systems_str}"
        )


if __name__ == "__main__":
    main()
