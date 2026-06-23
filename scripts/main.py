#!/usr/bin/env python3
"""Run the dataset creation pipeline end to end."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
NORMALIZED_DIR = DATA_DIR / "normalized"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_step(title: str, command: list[str]) -> None:
    print(f"\n==> {title}", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    python = sys.executable
    run_step(
        "Generate FLORES dataset",
        [python, "-B", str(SCRIPTS_DIR / "dataset_flores.py")],
    )
    run_step(
        "Rebuild merged full dataset",
        [python, "-B", str(SCRIPTS_DIR / "rebuild_full_dataset.py")],
    )
    run_step(
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
            str(NORMALIZED_DIR / "vec_sentences.report.json"),
        ],
    )
    run_step(
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
            str(NORMALIZED_DIR / "full_dataset.report.json"),
        ],
    )
    run_step(
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
            str(NORMALIZED_DIR / "eng_Latn_vec_Latn_dataset.report.json"),
        ],
    )
    run_step(
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
    )
    run_step(
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
            str(NORMALIZED_DIR / "train.report.json"),
        ],
    )
    run_step(
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
            str(NORMALIZED_DIR / "dev.report.json"),
        ],
    )
    run_step(
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
            str(NORMALIZED_DIR / "test.report.json"),
        ],
    )


if __name__ == "__main__":
    main()
