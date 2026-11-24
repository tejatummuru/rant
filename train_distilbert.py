"""Fine-tune distilbert-base-uncased on the Rant emotion dataset.

Same data and 28->7 label mapping as train.py, so numbers are directly
comparable to the sklearn stacked-ensemble baseline (weighted F1 ~0.55).
Expected F1 after fine-tuning: ~0.72-0.78.

Device auto-detection: CUDA > Apple MPS > CPU.
On Colab free T4 this takes ~45-60 min. On Apple MPS (M1/M2/M3) expect
2-4 hours. CPU-only is not recommended.

Usage:
    python train_distilbert.py                      # train, save to ./distilbert-rant/
    python train_distilbert.py --epochs 2           # fewer epochs
    python train_distilbert.py --push tt26586/rant-distilbert
                                                    # train + push to HF Hub
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from preprocess import EMOTION_MAPPING

MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = Path("data")
OUT_DIR = Path("distilbert-rant")

GO_EMOTION_COLS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_raw_data() -> pd.DataFrame:
    """Load the three sources with mapped labels.

    Unlike train.py (sklearn), we keep the ORIGINAL text — DistilBERT
    doesn't need the spaCy preprocessing. The tokenizer handles it.
    """
    frames = []

    dell = pd.read_csv(DATA_DIR / "sentiment-emotion-labelled_Dell_tweets.csv")
    dell = dell.dropna(subset=["Text", "emotion"])
    dell["mapped_emotion"] = dell["emotion"].map(EMOTION_MAPPING)
    dell = dell.loc[dell["mapped_emotion"].notna(), ["Text", "mapped_emotion"]]
    dell = dell.rename(columns={"Text": "text"})
    frames.append(dell)

    go = pd.read_csv(DATA_DIR / "go_emotions_dataset.csv")
    go["emotion"] = go[GO_EMOTION_COLS].idxmax(axis=1)
    go["mapped_emotion"] = go["emotion"].map(EMOTION_MAPPING)
    go = go.loc[go["mapped_emotion"].notna(), ["text", "mapped_emotion"]]
    frames.append(go)

    val_rows = []
    with open(DATA_DIR / "val.txt") as f:
        for line in f:
            text, emotion = line.strip().split(";")
            mapped = EMOTION_MAPPING.get(emotion)
            if mapped is None:
                continue
            val_rows.append({"text": text, "mapped_emotion": mapped})
    frames.append(pd.DataFrame(val_rows))

    combined = pd.concat(frames, ignore_index=True).dropna()
    combined = combined[combined["text"].str.strip().str.len() > 0]
    return combined


def compute_metrics_factory(label_names):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        p_m, r_m, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        return {
            "weighted_f1": f1_weighted,
            "macro_f1": f1_macro,
            "weighted_precision": p,
            "weighted_recall": r,
        }
    return compute_metrics


def main(args: argparse.Namespace) -> None:
    device = detect_device()
    print(f"device: {device}")

    print("Loading data...")
    df = load_raw_data()
    print(f"  {len(df)} rows, classes: {sorted(df['mapped_emotion'].unique())}")

    labels = sorted(df["mapped_emotion"].unique())
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    df["label"] = df["mapped_emotion"].map(label2id)

    train_df, test_df = train_test_split(
        df[["text", "label"]], test_size=0.1, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df["label"]
    )
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    print(f"Tokenizing with {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize, batched=True, remove_columns=["text"])
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True)).map(tokenize, batched=True, remove_columns=["text"])
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize, batched=True, remove_columns=["text"])

    print(f"Loading {MODEL_NAME} for sequence classification ({len(labels)} labels)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # Build TrainingArguments defensively — some kwargs have been renamed/
    # removed across transformers versions (e.g. warmup_ratio dropped in newer
    # releases, eval_strategy vs evaluation_strategy). We feed them through
    # a kwarg filter so this script works across versions.
    import inspect
    desired = dict(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        evaluation_strategy="epoch",  # older alias
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=100,
        report_to=[],
        fp16=(device == "cuda"),
        dataloader_num_workers=2 if device != "cpu" else 0,
        push_to_hub=False,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    accepted = {k: v for k, v in desired.items() if k in sig.parameters}
    training_args = TrainingArguments(**accepted)

    # Trainer: `tokenizer=` is deprecated in newer transformers in favor of
    # `processing_class=`. Pick whichever the installed version supports.
    trainer_sig = inspect.signature(Trainer.__init__)
    tokenizer_kwarg = {
        "processing_class": tokenizer
    } if "processing_class" in trainer_sig.parameters else {
        "tokenizer": tokenizer
    }
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(labels),
        **tokenizer_kwarg,
    )

    print("Training...")
    trainer.train()

    print("Evaluating on held-out test set...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    y_true = np.array(test_ds["label"])
    print("\nPer-class report (held-out test):")
    print(classification_report(y_true, preds, target_names=labels, zero_division=0))

    print(f"Saving model to {OUT_DIR}/")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    if args.push:
        print(f"Pushing to HF Hub: {args.push}")
        model.push_to_hub(args.push)
        tokenizer.push_to_hub(args.push)
        print(f"  live at https://huggingface.co/{args.push}")

    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--push", type=str, default=None,
                    help="HF Hub repo id to push to, e.g. tt26586/rant-distilbert")
    main(ap.parse_args())
