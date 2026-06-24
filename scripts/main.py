#!/usr/bin/env python3
"""Run the project pipeline in execution order.

By default this script runs the full local workflow:

1. dataset preparation
2. zero-shot NLLB baseline
3. NLLB fine-tuning
4. fine-tuned NLLB inference
5. aggregate evaluation
6. Streamlit demo launch

The OpenAI zero-shot baseline remains opt-in because it requires an API key
and can incur cost.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NORMALIZED_DIR = DATA_DIR / "normalized"
REPORT_DIR = DATA_DIR / "reports"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_step(title: str, command: list[str]) -> None:
    print(f"\n==> {title}", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Stop after dataset preparation and split generation.",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Do not launch the Streamlit demo at the end.",
    )
    parser.add_argument(
        "--run-llm-baseline",
        action="store_true",
        help="Run the OpenAI zero-shot baseline before evaluation.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip aggregate scoring of the prediction files.",
    )
    return parser


def add_data_prep_steps(steps: list[tuple[str, list[str]]], python: str) -> None:
    steps.extend(
        [
            (
                "Generate FLORES dataset",
                [python, "-B", str(SCRIPTS_DIR / "dataset_flores.py")],
            ),
            (
                "Normalize vec_sentences",
                [
                    python,
                    "-B",
                    str(SCRIPTS_DIR / "normalize_dataset.py"),
                    "--input-file",
                    str(DATA_DIR / "vec_sentences.jsonl"),
                    "--output-file",
                    str(NORMALIZED_DIR / "vec_sentences.jsonl"),
                    "--report-file",
                    str(REPORT_DIR / "vec_sentences.report.json"),
                ],
            ),
            (
                "Prepare Wikipedia candidates",
                [python, "-B", str(SCRIPTS_DIR / "prepare_wikipedia_candidates.py")],
            ),
            (
                "Rebuild merged full dataset",
                [python, "-B", str(SCRIPTS_DIR / "rebuild_full_dataset.py")],
            ),
            (
                "Normalize full_dataset",
                [
                    python,
                    "-B",
                    str(SCRIPTS_DIR / "normalize_dataset.py"),
                    "--input-file",
                    str(DATA_DIR / "full_dataset.jsonl"),
                    "--output-file",
                    str(NORMALIZED_DIR / "full_dataset.jsonl"),
                    "--report-file",
                    str(REPORT_DIR / "full_dataset.report.json"),
                ],
            ),
            (
                "Normalize FLORES dataset",
                [
                    python,
                    "-B",
                    str(SCRIPTS_DIR / "normalize_dataset.py"),
                    "--input-file",
                    str(DATA_DIR / "eng_Latn_vec_Latn_dataset.jsonl"),
                    "--output-file",
                    str(NORMALIZED_DIR / "eng_Latn_vec_Latn_dataset.jsonl"),
                    "--report-file",
                    str(REPORT_DIR / "eng_Latn_vec_Latn_dataset.report.json"),
                ],
            ),
            (
                "Create processed train/dev/test splits",
                [
                    python,
                    "-B",
                    str(SCRIPTS_DIR / "import_dataset.py"),
                    "--train-file",
                    str(NORMALIZED_DIR / "full_dataset.jsonl"),
                    "--eval-file",
                    str(NORMALIZED_DIR / "eng_Latn_vec_Latn_dataset.jsonl"),
                    "--output-dir",
                    str(DATA_DIR / "processed"),
                    "--train-ratio",
                    "0.8",
                    "--dev-ratio",
                    "0.1",
                    "--seed",
                    "42",
                ],
            ),
            (
                "Export normalized train split",
                [
                    python,
                    "-B",
                    str(SCRIPTS_DIR / "normalize_dataset.py"),
                    "--input-file",
                    str(DATA_DIR / "processed" / "train.jsonl"),
                    "--output-file",
                    str(NORMALIZED_DIR / "train.jsonl"),
                    "--report-file",
                    str(REPORT_DIR / "train.report.json"),
                ],
            ),
            (
                "Export normalized dev split",
                [
                    python,
                    "-B",
                    str(SCRIPTS_DIR / "normalize_dataset.py"),
                    "--input-file",
                    str(DATA_DIR / "processed" / "dev.jsonl"),
                    "--output-file",
                    str(NORMALIZED_DIR / "dev.jsonl"),
                    "--report-file",
                    str(REPORT_DIR / "dev.report.json"),
                ],
            ),
            (
                "Export normalized test split",
                [
                    python,
                    "-B",
                    str(SCRIPTS_DIR / "normalize_dataset.py"),
                    "--input-file",
                    str(DATA_DIR / "processed" / "test.jsonl"),
                    "--output-file",
                    str(NORMALIZED_DIR / "test.jsonl"),
                    "--report-file",
                    str(REPORT_DIR / "test.report.json"),
                ],
            ),
        ]
    )


def add_nllb_baseline_steps(steps: list[tuple[str, list[str]]], python: str) -> None:
    steps.append(
        (
            "Run NLLB zero-shot baseline",
            [
                python,
                "-B",
                str(SCRIPTS_DIR / "baseline_nllb_zeroshot.py"),
                "--test-file",
                str(NORMALIZED_DIR / "test.jsonl"),
                "--output-file",
                str(RESULTS_DIR / "nllb_zeroshot_predictions.jsonl"),
            ],
        )
    )


def add_finetune_steps(steps: list[tuple[str, list[str]]], python: str) -> None:
    steps.extend(
        [
            (
                "Fine-tune NLLB model",
                [
                    python,
                    "-B",
                    str(SCRIPTS_DIR / "finetune_nllb.py"),
                    "--train-file",
                    str(NORMALIZED_DIR / "train.jsonl"),
                    "--dev-file",
                    str(NORMALIZED_DIR / "dev.jsonl"),
                    "--output-dir",
                    str(MODELS_DIR / "nllb_finetuned"),
                ],
            ),
            (
                "Run fine-tuned NLLB inference",
                [
                    python,
                    "-B",
                    str(SCRIPTS_DIR / "baseline_nllb_zeroshot.py"),
                    "--test-file",
                    str(NORMALIZED_DIR / "test.jsonl"),
                    "--model",
                    str(MODELS_DIR / "nllb_finetuned"),
                    "--output-file",
                    str(RESULTS_DIR / "nllb_finetuned_predictions.jsonl"),
                ],
            ),
        ]
    )


def add_llm_baseline_steps(steps: list[tuple[str, list[str]]], python: str) -> None:
    steps.append(
        (
            "Run LLM zero-shot baseline",
            [
                python,
                "-B",
                str(SCRIPTS_DIR / "baseline_llm_zeroshot.py"),
                "--test-file",
                str(NORMALIZED_DIR / "test.jsonl"),
                "--output-file",
                str(RESULTS_DIR / "llm_zeroshot_predictions.jsonl"),
            ],
        )
    )


def add_evaluation_steps(steps: list[tuple[str, list[str]]], python: str) -> None:
    steps.append(
        (
            "Evaluate available prediction files",
            [
                python,
                "-B",
                str(SCRIPTS_DIR / "evaluate.py"),
                "--results-dir",
                str(RESULTS_DIR),
                "--output-file",
                str(RESULTS_DIR / "evaluation_results.json"),
            ],
        )
    )


def add_demo_steps(steps: list[tuple[str, list[str]]], python: str) -> None:
    steps.append(
        (
            "Launch Streamlit demo",
            [
                python,
                "-m",
                "streamlit",
                "run",
                str(SCRIPTS_DIR / "demo_app.py"),
            ],
        )
    )


def main() -> None:
    args = build_argparser().parse_args()
    python = sys.executable
    steps: list[tuple[str, list[str]]] = []

    add_data_prep_steps(steps, python)

    if not args.data_only:
        add_nllb_baseline_steps(steps, python)
        add_finetune_steps(steps, python)

        if args.run_llm_baseline:
            add_llm_baseline_steps(steps, python)
        if not args.skip_evaluation:
            add_evaluation_steps(steps, python)
        if not args.skip_demo:
            add_demo_steps(steps, python)

    for title, command in steps:
        run_step(title, command)


if __name__ == "__main__":
    main()
