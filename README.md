# llm-translation-eng-dialect
## AIM
Build a small Machine Translation (MT) system for a low-resource language. The team picks one language they have access to (an Italian regional language like Sardinian or Neapolitan, a North-African dialect like Tunisian Arabic, a small Berber language like Tamazight, or any minority language they know personally). The team curates a tiny parallel corpus, fine-tunes a massively multilingual MT model (NLLB-200), and compares against the same model used zero-shot. Reports BLEU/chrF plus an honest writeup of what is hard about low-resource MT.


## Dataset pipeline

Run the full pipeline with:

```bash
python scripts/main.py
```

This will:

1. generate the FLORES dataset in `data/`
2. normalize `data/vec_sentences.jsonl` into the standard record file in `data/normalized/`
3. convert `data/raw/wikipedia_candidates.jsonl` into standard dataset records in `data/wikipedia_candidates.jsonl`
4. rebuild `data/full_dataset.jsonl` from those standardized records
5. create normalized datasets in `data/normalized/`
6. create processed `train/dev/test` splits in `data/processed/`
7. export normalized `train/dev/test` files in `data/normalized/`

## Normalization

The normalization profile lives in [config/venetian_normalization.json](<your_path>/llm-translation-eng-dialect/config/venetian_normalization.json).

It currently performs conservative normalization:

- orthographic cleanup such as Unicode normalization, apostrophe cleanup, `ł -> l`, `xè -> xe`
- light lexical canonicalization such as `voialtri/valtri/vialtri -> vualtri`
- metadata cleanup such as `coversation -> conversation`
- prompt standardization so every record uses `translate English to Venetian: ...`

To normalize a single dataset manually:

```bash
python scripts/normalize_dataset.py \
  --input-file data/vec_sentences.jsonl \
  --output-file data/normalized/vec_sentences.jsonl \
  --report-file data/normalized/vec_sentences.report.json
```

## Data Provenance

The dataset sources are split by role:

- `train`: manual translations collected in [data/full_dataset.jsonl](/llm-translation-eng-dialect/data/full_dataset.jsonl)
- `dev`: sampled from FLORES, generated through [scripts/dataset_flores.py](/llm-translation-eng-dialect/scripts/dataset_flores.py)
- `test`: sampled from FLORES, generated through [scripts/dataset_flores.py](/llm-translation-eng-dialect/scripts/dataset_flores.py)

In practice:

- the manual translation side is the curated training source
- `data/normalized/vec_sentences.jsonl` is the standardized manual-record source used to rebuild `full_dataset`
- `data/wikipedia_candidates.jsonl` is the standardized Wikipedia-candidate source used to extend `full_dataset`
- the FLORES English to Venetian dataset is the evaluation pool
- `dev` and `test` are created from that FLORES pool during the processing pipeline

Relevant files:

- [data/full_dataset.jsonl](llm-translation-eng-dialect/data/full_dataset.jsonl): merged manual-translation training source
- [data/eng_Latn_vec_Latn_dataset.jsonl](llm-translation-eng-dialect/data/eng_Latn_vec_Latn_dataset.jsonl): FLORES-derived dataset
- [data/processed/train.jsonl](llm-translation-eng-dialect/data/processed/train.jsonl): final train split
- [data/processed/dev.jsonl](llm-translation-eng-dialect/data/processed/dev.jsonl): final dev split from FLORES
- [data/processed/test.jsonl](llm-translation-eng-dialect/data/processed/test.jsonl): final test split from FLORES
- [data/normalized/train.jsonl](llm-translation-eng-dialect/data/normalized/train.jsonl): normalized train split
- [data/normalized/dev.jsonl](llm-translation-eng-dialect/data/normalized/dev.jsonl): normalized dev split
- [data/normalized/test.jsonl](llm-translation-eng-dialect/data/normalized/test.jsonl): normalized test split

# STRUCTURE

- `data/full_dataset.jsonl`: manual-translation dataset used as the training source.
- `data/eng_Latn_vec_Latn_dataset.jsonl`: FLORES-derived dataset used as the evaluation source.
- `data/normalized/`: normalized versions of the raw datasets.
- `data/normalized/train.jsonl`, `dev.jsonl`, `test.jsonl`: normalized split files.
- `data/processed/`: final train/dev/test splits and summary files.
- `scripts/dataset_flores.py`: generates the FLORES-based English to Venetian dataset.
- `scripts/rebuild_full_dataset.py`: rebuilds the merged manual dataset.
- `scripts/normalize_dataset.py`: applies normalization rules.
- `scripts/import_dataset.py`: creates processed splits.
- `scripts/main.py`: runs the whole pipeline.
- `config/venetian_normalization.json`: normalization rules for orthography, lexical variants, and metadata.
- `src/dataset.py`: dataset loading, writing, splitting, and renumbering helpers.
- `src/normalization.py`: implementation of the Venetian normalizer.

- `scripts/demo_app.py`: Streamlit demo for aggregate metrics and sentence-level zero-shot versus fine-tuned comparisons.

## Interactive demo

The interactive demo is implemented in `scripts/demo_app.py` with Streamlit. It uses the saved prediction and evaluation files, so it does not load either NLLB model at runtime and starts quickly enough for a short live presentation.

The demo expects these files:

- `data/normalized/train.jsonl`
- `data/normalized/dev.jsonl`
- `data/normalized/test.jsonl`
- `data/results/nllb_zeroshot_predictions.jsonl`
- `data/results/nllb_finetuned_predictions.jsonl`
- `data/results/evaluation_results.json`

Generate the baseline predictions, fine-tuned predictions, and aggregate evaluation results before starting the demo. If these files already exist, this step can be skipped.

Install the project dependencies, including Streamlit, with:

```bash
python -m pip install -r requirements.txt
```

Run the demo from the repository root with:

```bash
python -m streamlit run scripts/demo_app.py
```

Streamlit will open the application in the browser. By default, it is available at:

```text
http://localhost:8501
```

The demo displays:

1. the train, development, and test split sizes
2. aggregate BLEU and chrF scores for NLLB zero-shot and NLLB fine-tuned
3. the number of sentence-level improvements, regressions, and equal scores
4. a fixed presentation example plus the largest improvements and regressions
5. the main experimental takeaway

The demo reads saved results only. It does not run training or inference during the presentation.
