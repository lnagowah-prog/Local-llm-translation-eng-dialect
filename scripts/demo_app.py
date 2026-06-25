#!/usr/bin/env python3
"""Interactive demo for the English-to-Venetian translation project.

Loads the current prediction and evaluation files, compares NLLB zero-shot,
an LLM zero-shot baseline, and the fine-tuned NLLB model,
and supports interactive English-to-Venetian translation.

Run from the repository root with:

    python -m streamlit run scripts/demo_app.py

The evaluation sections use saved predictions. The free-text translation
section loads a model only when the user presses the Translate button.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import sacrebleu
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import read_jsonl

DATA_DIR = PROJECT_ROOT / "data" / "normalized"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

NLLB_ZERO_SHOT_FILE = RESULTS_DIR / "nllb_zeroshot_predictions.jsonl"
LLM_ZERO_SHOT_FILE = RESULTS_DIR / "llm_zeroshot_predictions.jsonl"
ABLATION_A0_FILE = RESULTS_DIR / "ablation_A0_predictions.jsonl"
EVALUATION_FILE = RESULTS_DIR / "evaluation_results.json"

BASE_MODEL_ID = "facebook/nllb-200-distilled-600M"
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "vec_Latn"
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MAX_INPUT_LENGTH = 128
DEFAULT_MAX_NEW_TOKENS = 128

FINE_TUNED_MODEL_CANDIDATES = [
    MODEL_DIR / "nllb_finetuned_updated",
    MODEL_DIR / "ablation_A0",
    MODEL_DIR / "nllb_finetuned_lr1e5",
    MODEL_DIR / "nllb_finetuned",
    MODEL_DIR / "nllb_finetuned_lr5e5_1epoch",
]

SYSTEM_SPECS = [
    {
        "id": "nllb_zeroshot",
        "label": "NLLB zero-shot",
        "file": NLLB_ZERO_SHOT_FILE,
    },
    {
        "id": "llm_zeroshot",
        "label": "LLM zero-shot",
        "file": LLM_ZERO_SHOT_FILE,
    },
    {
        "id": "ablation_A0",
        "label": "Fine-tuned NLLB",
        "file": ABLATION_A0_FILE,
    },
]
SYSTEM_LABELS = {
    spec["id"]: spec["label"]
    for spec in SYSTEM_SPECS
}
SYSTEM_ORDER = [
    spec["label"]
    for spec in SYSTEM_SPECS
]

BASELINE_SYSTEM_ID = "nllb_zeroshot"
LLM_SYSTEM_ID = "llm_zeroshot"
BEST_SYSTEM_ID = "ablation_A0"

PRESENTATION_VIEW = "Presentation example"
IMPROVEMENT_VIEW = "Largest fine-tuned improvements"
REGRESSION_VIEW = "Largest fine-tuned regressions"
ALL_EXAMPLES_VIEW = "All test examples"

# Selected manually from the updated test set. It is the same clear,
# neutral example previously used in the presentation, now evaluated
# with the current systems and test split.
PRESENTATION_EXAMPLE_ID = "0737"


def count_jsonl(path: Path) -> int:
    """Count non-empty records in a JSONL file."""
    if not path.exists():
        return 0
    return len(read_jsonl(path))


def require_files(paths: list[Path]) -> None:
    """Raise an informative error when required demo files are missing."""
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in paths
        if not path.exists()
    ]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required demo files: {joined}. "
            "Update the repository data and evaluation artefacts first."
        )


def file_signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    """Return a cache key that changes when an input file changes."""
    return tuple(
        (
            str(path),
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in paths
    )


def sentence_chrf(hypothesis: str, reference: str) -> float:
    """Compute an exploratory sentence-level chrF score."""
    if not hypothesis.strip():
        return 0.0
    return sacrebleu.sentence_chrf(
        hypothesis,
        [reference],
    ).score


def build_comparisons(
    predictions_by_system: dict[str, list[dict]],
) -> list[dict]:
    """Join prediction files by record ID and compute sentence-level chrF."""
    records_by_system = {
        system_id: {
            str(record["id"]): record
            for record in records
        }
        for system_id, records in predictions_by_system.items()
    }

    baseline_records = predictions_by_system[BASELINE_SYSTEM_ID]
    comparisons: list[dict] = []

    for baseline_record in baseline_records:
        record_id = str(baseline_record["id"])
        a0_record = records_by_system[BEST_SYSTEM_ID].get(record_id)
        llm_record = records_by_system[LLM_SYSTEM_ID].get(record_id)

        if a0_record is None or llm_record is None:
            continue

        reference = baseline_record["reference"]
        hypotheses = {
            BASELINE_SYSTEM_ID: baseline_record.get("hypothesis", ""),
            LLM_SYSTEM_ID: llm_record.get("hypothesis", ""),
            BEST_SYSTEM_ID: a0_record.get("hypothesis", ""),
        }
        scores = {
            system_id: sentence_chrf(hypothesis, reference)
            for system_id, hypothesis in hypotheses.items()
        }

        comparisons.append(
            {
                "id": record_id,
                "source_text": baseline_record["source_text"],
                "reference": reference,
                "hypotheses": hypotheses,
                "scores": scores,
                "models": {
                    BASELINE_SYSTEM_ID: baseline_record.get(
                        "model",
                        BASE_MODEL_ID,
                    ),
                    LLM_SYSTEM_ID: llm_record.get(
                        "model",
                        "LLM zero-shot",
                    ),
                    BEST_SYSTEM_ID: a0_record.get(
                        "model",
                        "Fine-tuned NLLB",
                    ),
                },
                "a0_delta_chrf": (
                    scores[BEST_SYSTEM_ID]
                    - scores[BASELINE_SYSTEM_ID]
                ),
                "llm_delta_chrf": (
                    scores[LLM_SYSTEM_ID]
                    - scores[BASELINE_SYSTEM_ID]
                ),
            }
        )

    return comparisons


@st.cache_data
def load_demo_data(
    _signature: tuple[tuple[str, int, int], ...],
) -> tuple[list[dict], dict, dict[str, int]]:
    """Load current predictions, evaluation results, and split sizes."""
    predictions_by_system = {
        spec["id"]: read_jsonl(spec["file"])
        for spec in SYSTEM_SPECS
    }
    evaluation = json.loads(
        EVALUATION_FILE.read_text(encoding="utf-8")
    )
    comparisons = build_comparisons(predictions_by_system)

    split_sizes = {
        "train": count_jsonl(DATA_DIR / "train.jsonl"),
        "dev": count_jsonl(DATA_DIR / "dev.jsonl"),
        "test": count_jsonl(DATA_DIR / "test.jsonl"),
    }
    return comparisons, evaluation, split_sizes


def build_metric_table(evaluation: dict) -> pd.DataFrame:
    """Convert saved evaluation results into an ordered display table."""
    systems = evaluation.get("systems", {})
    rows = []

    for spec in SYSTEM_SPECS:
        metrics = systems.get(spec["id"])
        if not metrics:
            continue
        rows.append(
            {
                "System": spec["label"],
                "BLEU": metrics.get("bleu"),
                "chrF": metrics.get("chrf"),
                "Sentences": metrics.get("n"),
            }
        )

    return pd.DataFrame(rows)


def filter_comparisons(
    comparisons: list[dict],
    view: str,
) -> list[dict]:
    """Filter and order examples for the selected presentation view."""
    if view == PRESENTATION_VIEW:
        return [
            row
            for row in comparisons
            if row["id"] == PRESENTATION_EXAMPLE_ID
        ]

    if view == IMPROVEMENT_VIEW:
        rows = [
            row
            for row in comparisons
            if row["a0_delta_chrf"] > 0
        ]
        return sorted(
            rows,
            key=lambda row: row["a0_delta_chrf"],
            reverse=True,
        )

    if view == REGRESSION_VIEW:
        rows = [
            row
            for row in comparisons
            if row["a0_delta_chrf"] < 0
        ]
        return sorted(
            rows,
            key=lambda row: row["a0_delta_chrf"],
        )

    return sorted(
        comparisons,
        key=lambda row: row["id"],
    )


def format_example(row: dict) -> str:
    """Build a compact label for the sentence selector."""
    source_text = row["source_text"].replace("\n", " ")
    preview = source_text[:95]
    suffix = "..." if len(source_text) > 95 else ""
    return f'{row["id"]} — {preview}{suffix}'


def render_dataset_summary(split_sizes: dict[str, int]) -> None:
    """Render train, development, and test split sizes."""
    st.subheader("1. Dataset")
    train_col, dev_col, test_col = st.columns(3)
    train_col.metric(
        "Training examples",
        split_sizes["train"],
        help="Parallel sentences used for fine-tuning.",
    )
    dev_col.metric(
        "Development examples",
        split_sizes["dev"],
        help="Held-out sentences used during model development.",
    )
    test_col.metric(
        "Test examples",
        split_sizes["test"],
        help="Unseen sentences used for the final comparison.",
    )


def render_metric_chart(
    metric_table: pd.DataFrame,
    metric: str,
) -> None:
    """Render one metric with an explicit, stable system order."""
    chart_data = metric_table[["System", metric]].copy()
    st.vega_lite_chart(
        chart_data,
        {
            "mark": "bar",
            "encoding": {
                "y": {
                    "field": "System",
                    "type": "nominal",
                    "sort": SYSTEM_ORDER,
                    "title": None,
                },
                "x": {
                    "field": metric,
                    "type": "quantitative",
                    "title": metric,
                    "scale": {"zero": True},
                },
                "tooltip": [
                    {
                        "field": "System",
                        "type": "nominal",
                    },
                    {
                        "field": metric,
                        "type": "quantitative",
                        "format": ".2f",
                    },
                ],
            },
            "height": 190,
        },
        width="stretch",
    )


def metric_delta(
    systems: dict,
    system_id: str,
    baseline_id: str,
    metric: str,
) -> float | None:
    """Return a corpus-level metric delta when both values are present."""
    system = systems.get(system_id)
    baseline = systems.get(baseline_id)
    if not system or not baseline:
        return None

    system_value = system.get(metric)
    baseline_value = baseline.get(metric)
    if system_value is None or baseline_value is None:
        return None

    return float(system_value) - float(baseline_value)


def render_aggregate_results(evaluation: dict) -> None:
    """Render corpus-level BLEU and chrF results."""
    st.subheader("2. Aggregate test results")
    metric_table = build_metric_table(evaluation)

    if metric_table.empty:
        st.warning(
            "No aggregate results were found in the evaluation file."
        )
        return

    table_col, bleu_col, chrf_col = st.columns([1.35, 1, 1])
    with table_col:
        st.dataframe(
            metric_table,
            hide_index=True,
            width="stretch",
        )
    with bleu_col:
        render_metric_chart(metric_table, "BLEU")
    with chrf_col:
        render_metric_chart(metric_table, "chrF")

    systems = evaluation.get("systems", {})
    a0_bleu_delta = metric_delta(
        systems,
        BEST_SYSTEM_ID,
        BASELINE_SYSTEM_ID,
        "bleu",
    )
    a0_chrf_delta = metric_delta(
        systems,
        BEST_SYSTEM_ID,
        BASELINE_SYSTEM_ID,
        "chrf",
    )
    llm_bleu_delta = metric_delta(
        systems,
        LLM_SYSTEM_ID,
        BASELINE_SYSTEM_ID,
        "bleu",
    )
    llm_chrf_delta = metric_delta(
        systems,
        LLM_SYSTEM_ID,
        BASELINE_SYSTEM_ID,
        "chrf",
    )

    delta_columns = st.columns(4)
    delta_values = [
        (
            "Fine-tuned vs NLLB: BLEU",
            a0_bleu_delta,
        ),
        (
            "Fine-tuned vs NLLB: chrF",
            a0_chrf_delta,
        ),
        (
            "LLM vs NLLB: BLEU",
            llm_bleu_delta,
        ),
        (
            "LLM vs NLLB: chrF",
            llm_chrf_delta,
        ),
    ]
    for column, (label, value) in zip(
        delta_columns,
        delta_values,
    ):
        column.metric(
            label,
            "n/a" if value is None else f"{value:+.2f}",
        )

    if (
        a0_bleu_delta is not None
        and a0_chrf_delta is not None
        and a0_bleu_delta > 0
        and a0_chrf_delta > 0
    ):
        st.success(
            "The fine-tuned NLLB model improves both aggregate metrics over the NLLB "
            "zero-shot baseline. The LLM baseline has the highest BLEU, "
            "while the fine-tuned NLLB model has the highest chrF."
        )


def render_system_prediction(
    selected: dict,
    system_id: str,
    delta: float | None = None,
) -> None:
    """Render one system prediction and its sentence-level chrF."""
    st.markdown(f'**{SYSTEM_LABELS[system_id]}**')
    hypothesis = selected["hypotheses"][system_id]

    if hypothesis.strip():
        st.write(hypothesis)
    else:
        st.warning("No prediction was produced for this sentence.")

    metric_kwargs = {
        "label": "Sentence chrF",
        "value": f'{selected["scores"][system_id]:.2f}',
        "help": "Exploratory sentence-level score.",
    }
    if delta is not None:
        metric_kwargs["delta"] = f"{delta:+.2f}"

    st.metric(**metric_kwargs)


def render_example_comparison(
    comparisons: list[dict],
) -> None:
    """Render an interactive sentence-level comparison."""
    st.subheader("3. Sentence-level comparison")

    improved = sum(
        row["a0_delta_chrf"] > 0
        for row in comparisons
    )
    worsened = sum(
        row["a0_delta_chrf"] < 0
        for row in comparisons
    )
    unchanged = len(comparisons) - improved - worsened

    improved_col, worsened_col, unchanged_col = st.columns(3)
    improved_col.metric("Fine-tuned better than NLLB", improved)
    worsened_col.metric("Fine-tuned worse than NLLB", worsened)
    unchanged_col.metric("Equal sentence score", unchanged)

    view = st.radio(
        "Example view",
        [
            PRESENTATION_VIEW,
            IMPROVEMENT_VIEW,
            REGRESSION_VIEW,
            ALL_EXAMPLES_VIEW,
        ],
        horizontal=True,
        help=(
            "The presentation example is fixed and manually selected. "
            "The other views rank the fine-tuned model relative to NLLB zero-shot."
        ),
    )
    filtered = filter_comparisons(comparisons, view)

    if not filtered:
        st.warning(
            f"Presentation example {PRESENTATION_EXAMPLE_ID} "
            "was not found in the current prediction files."
        )
        return

    selected_index = st.selectbox(
        "Select a test sentence",
        range(len(filtered)),
        format_func=lambda index: format_example(
            filtered[index]
        ),
    )
    selected = filtered[selected_index]

    st.markdown("**English source**")
    st.info(selected["source_text"])

    st.markdown("**Venetian reference**")
    st.write(selected["reference"])

    nllb_col, llm_col, a0_col = st.columns(3)
    with nllb_col:
        render_system_prediction(
            selected,
            BASELINE_SYSTEM_ID,
        )
    with llm_col:
        render_system_prediction(
            selected,
            LLM_SYSTEM_ID,
            selected["llm_delta_chrf"],
        )
    with a0_col:
        render_system_prediction(
            selected,
            BEST_SYSTEM_ID,
            selected["a0_delta_chrf"],
        )

    with st.expander("Technical details"):
        for system_id in [
            BASELINE_SYSTEM_ID,
            LLM_SYSTEM_ID,
            BEST_SYSTEM_ID,
        ]:
            st.write(
                f'{SYSTEM_LABELS[system_id]} model: '
                f'`{selected["models"][system_id]}`'
            )
        st.write(
            "Sentence-level chrF is shown for exploration. "
            "The main conclusion is based on corpus-level test metrics."
        )


def has_model_weights(path: Path) -> bool:
    """Return whether a directory appears to contain a saved model."""
    if not path.is_dir() or not (path / "config.json").exists():
        return False

    expected_files = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]
    return any(
        (path / filename).exists()
        for filename in expected_files
    )


def discover_fine_tuned_model() -> Path | None:
    """Find a locally saved fine-tuned model."""
    for candidate in FINE_TUNED_MODEL_CANDIDATES:
        if has_model_weights(candidate):
            return candidate

        if candidate.is_dir():
            checkpoints = sorted(
                candidate.glob("checkpoint-*"),
                key=lambda path: path.name,
                reverse=True,
            )
            for checkpoint in checkpoints:
                if has_model_weights(checkpoint):
                    return checkpoint

    if MODEL_DIR.is_dir():
        for config_path in sorted(
            MODEL_DIR.rglob("config.json")
        ):
            candidate = config_path.parent
            if has_model_weights(candidate):
                return candidate

    return None


def model_cache_token(model_reference: str) -> str:
    """Return a cache token that changes if local model weights change."""
    path = Path(model_reference).expanduser()

    if not path.is_dir():
        return model_reference

    model_files = [
        path / "config.json",
        path / "model.safetensors",
        path / "pytorch_model.bin",
    ]
    existing = [
        model_file
        for model_file in model_files
        if model_file.exists()
    ]
    signature = "|".join(
        (
            f"{model_file.name}:"
            f"{model_file.stat().st_mtime_ns}:"
            f"{model_file.stat().st_size}"
        )
        for model_file in existing
    )
    return f"{path.resolve()}|{signature}"


@st.cache_resource(show_spinner=False)
def load_translation_model(
    model_reference: str,
    _cache_token: str,
) -> tuple[
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    torch.device,
]:
    """Load and cache one translation model for interactive inference."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_reference,
        src_lang=SOURCE_LANG,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_reference,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device


@torch.inference_mode()
def translate_to_venetian(
    source_text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
) -> str:
    """Translate one English sentence into Venetian."""
    tokenizer.src_lang = SOURCE_LANG
    target_lang_id = tokenizer.convert_tokens_to_ids(
        TARGET_LANG
    )

    if (
        target_lang_id is None
        or target_lang_id == tokenizer.unk_token_id
    ):
        raise ValueError(
            f"The tokenizer does not recognize the target "
            f"language code {TARGET_LANG!r}."
        )

    encoded = tokenizer(
        source_text,
        return_tensors="pt",
        truncation=True,
        max_length=DEFAULT_MAX_INPUT_LENGTH,
    )
    encoded = {
        name: tensor.to(device)
        for name, tensor in encoded.items()
    }

    generated = model.generate(
        **encoded,
        forced_bos_token_id=target_lang_id,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()


def display_model_reference(model_reference: str) -> str:
    """Return a presentation-safe model name."""
    path = Path(model_reference).expanduser()
    if path.is_dir():
        return path.name
    return model_reference


def render_free_text_translation() -> None:
    """Render an interactive English-to-Venetian translation form."""
    st.subheader("4. Try your own sentence")
    st.write(
        "Enter an English sentence and generate a new Venetian "
        "translation at runtime."
    )

    discovered_model = discover_fine_tuned_model()
    default_fine_tuned_reference = (
        str(discovered_model)
        if discovered_model is not None
        else ""
    )

    with st.expander("Translation model settings"):
        st.caption(
            "The zero-shot model is downloaded from Hugging Face "
            "when first used. The fine-tuned system uses a local checkpoint "
            "or a Hugging Face model ID."
        )
        fine_tuned_reference = st.text_input(
            "Fine-tuned model path or Hugging Face ID",
            value=default_fine_tuned_reference,
            placeholder=(
                "models/nllb_finetuned_updated "
                "or organization/model-name"
            ),
        )

    choices = [
        "NLLB zero-shot",
        "Fine-tuned NLLB",
    ]
    default_choice = (
        1
        if default_fine_tuned_reference
        else 0
    )
    model_choice = st.radio(
        "Translation system",
        choices,
        index=default_choice,
        horizontal=True,
    )
    source_text = st.text_area(
        "English text",
        placeholder="Type an English sentence here...",
        height=100,
        max_chars=1000,
    )

    translate_clicked = st.button(
        "Translate into Venetian",
        type="primary",
        disabled=not source_text.strip(),
    )

    if translate_clicked:
        if model_choice == "NLLB zero-shot":
            model_reference = BASE_MODEL_ID
            model_label = "NLLB zero-shot"
        else:
            model_reference = fine_tuned_reference.strip()
            model_label = "Fine-tuned NLLB"
            if not model_reference:
                st.error(
                    "Provide the fine-tuned checkpoint path or a "
                    "Hugging Face model ID in the model settings."
                )
                return

        try:
            with st.spinner(
                "Loading the model and translating. "
                "The first run may take several minutes..."
            ):
                tokenizer, model, device = load_translation_model(
                    model_reference,
                    model_cache_token(model_reference),
                )
                translation = translate_to_venetian(
                    source_text.strip(),
                    tokenizer,
                    model,
                    device,
                )
        except Exception as exc:
            st.error(f"Translation failed: {exc}")
            return

        st.session_state["free_text_translation"] = {
            "source": source_text.strip(),
            "translation": translation,
            "model_label": model_label,
            "model_reference": model_reference,
        }

    result = st.session_state.get(
        "free_text_translation"
    )
    if result:
        st.markdown("**Venetian translation**")
        st.success(result["translation"])
        st.caption(
            f'Model: {result["model_label"]} — '
            f'`{display_model_reference(result["model_reference"])}`'
        )


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="English-to-Venetian Translation Demo",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("English-to-Venetian Machine Translation")
    st.caption(
        "A comparison of NLLB-200 zero-shot translation, "
        "an LLM zero-shot baseline, and the best fine-tuned "
        "NLLB configuration."
    )

    required_files = [
        DATA_DIR / "train.jsonl",
        DATA_DIR / "dev.jsonl",
        DATA_DIR / "test.jsonl",
        NLLB_ZERO_SHOT_FILE,
        LLM_ZERO_SHOT_FILE,
        ABLATION_A0_FILE,
        EVALUATION_FILE,
    ]

    try:
        require_files(required_files)
        signature = file_signature(required_files)
        comparisons, evaluation, split_sizes = load_demo_data(
            signature
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as exc:
        st.error(str(exc))
        st.stop()

    if split_sizes != {
        "train": 653,
        "dev": 82,
        "test": 82,
    }:
        st.error(
            "The demo did not find the expected updated split sizes "
            "(653 train, 82 development, 82 test). "
            "It will not display results from a different experiment."
        )
        st.stop()

    if len(comparisons) != 82:
        st.error(
            "The three prediction files do not contain 82 matching "
            "test examples. Check that they belong to the same experiment."
        )
        st.stop()

    render_dataset_summary(split_sizes)
    render_aggregate_results(evaluation)
    render_example_comparison(comparisons)
    render_free_text_translation()

    st.divider()
    st.markdown(
        "**Takeaway:** the fine-tuned NLLB model improves the "
        "NLLB zero-shot baseline on both aggregate metrics. "
        "The LLM zero-shot baseline obtains the highest BLEU, "
        "while the fine-tuned NLLB model obtains the highest chrF."
    )


if __name__ == "__main__":
    main()
