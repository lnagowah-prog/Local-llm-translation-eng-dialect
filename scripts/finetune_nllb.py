#!/usr/bin/env python3
"""Baseline 2: fine-tune NLLB-200 on the Venetian parallel corpus.

Trains the model on train.jsonl, evaluates chrF on dev.jsonl after each
epoch, and saves the best checkpoint. After training, run inference with
the zero-shot script repointed at the saved checkpoint:

    python scripts/baseline_nllb_zeroshot.py \\
        --model models/nllb_finetuned \\
        --output-file data/results/nllb_finetuned_predictions.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import read_jsonl

MODEL_ID = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "vec_Latn"
DEFAULT_MAX_LEN = 256


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--train-file", default="data/normalized/train.jsonl")
    p.add_argument("--dev-file", default="data/normalized/dev.jsonl")
    p.add_argument("--output-dir", default="models/nllb_finetuned")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2,
                   help="Gradient accumulation steps. Effective batch = batch-size × grad-accum.")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--num-beams", type=int, default=4,
                   help="Beam size for generation during dev eval. Must match baseline_nllb_zeroshot.py.")
    p.add_argument("--no-fp16", action="store_true",
                   help="Disable fp16 (use on CPU or when GPU does not support it).")
    p.add_argument("--lora", action="store_true",
                   help="Apply LoRA adapters (requires peft). Fewer trainable params, less forgetting.")
    p.add_argument("--lora-rank", type=int, default=16, metavar="R",
                   help="LoRA rank r (default 16).")
    p.add_argument("--lora-alpha", type=int, default=32, metavar="A",
                   help="LoRA scaling alpha (default 32).")
    return p


def make_hf_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_list(
        [{"source_text": r["source_text"], "target_text": r["target_text"]} for r in records]
    )


def make_tokenize_fn(tokenizer: AutoTokenizer, max_length: int):
    def tokenize(batch: dict) -> dict:
        model_inputs = tokenizer(
            batch["source_text"],
            text_target=batch["target_text"],
            max_length=max_length,
            truncation=True,
        )
        return model_inputs

    return tokenize


def make_compute_metrics(tokenizer: AutoTokenizer):
    def compute_metrics(eval_preds) -> dict:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # -100 is the HuggingFace ignore index used for padding in labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        chrf_score = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels]).score
        bleu_score = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        return {"chrf": round(chrf_score, 2), "bleu": round(bleu_score, 2)}

    return compute_metrics


def main() -> None:
    args = build_argparser().parse_args()

    train_path = PROJECT_ROOT / args.train_file
    dev_path = PROJECT_ROOT / args.dev_file
    output_dir = PROJECT_ROOT / args.output_dir

    use_fp16 = not args.no_fp16 and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  fp16: {use_fp16}")
    print(f"Loading {args.model} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    tgt_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    model.generation_config.forced_bos_token_id = tgt_lang_id

    if args.lora:
        if not _PEFT_AVAILABLE:
            sys.exit("LoRA requires peft. Install it with: pip install peft")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print("Tokenising datasets ...")
    tokenize = make_tokenize_fn(tokenizer, args.max_length)
    cols_to_remove = ["source_text", "target_text"]

    train_ds = make_hf_dataset(read_jsonl(train_path)).map(
        tokenize, batched=True, remove_columns=cols_to_remove
    )
    dev_ds = make_hf_dataset(read_jsonl(dev_path)).map(
        tokenize, batched=True, remove_columns=cols_to_remove
    )
    print(f"  train: {len(train_ds)}  dev: {len(dev_ds)}")

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=use_fp16,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=args.num_beams,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"Training for up to {args.epochs} epochs ...")
    trainer.train()

    print("\nFinal dev evaluation:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"Saving checkpoint → {output_dir}")
    if args.lora:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(output_dir))
    else:
        trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
