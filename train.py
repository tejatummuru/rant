"""Train the stacked-ensemble emotion classifier and save artifacts.

Expects three data files under ./data/:
  - sentiment-emotion-labelled_Dell_tweets.csv
  - go_emotions_dataset.csv
  - val.txt

Outputs:
  - model.joblib         (fitted sklearn Pipeline)
  - label_encoder.joblib (fitted LabelEncoder)

Run once locally, then commit the artifacts for the Gradio app to load.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from preprocess import EMOTION_MAPPING, batch_preprocess, preprocess_text

DATA_DIR = Path("data")
MODEL_PATH = Path("model.joblib")
ENCODER_PATH = Path("label_encoder.joblib")

GO_EMOTION_COLS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


def load_dell_tweets() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "sentiment-emotion-labelled_Dell_tweets.csv")
    df = df.drop(columns=["Unnamed: 0", "Datetime", "Tweet Id", "Username"], errors="ignore")
    df = df.dropna(subset=["Text", "emotion"])
    df["processed_text"] = batch_preprocess(df["Text"].astype(str).tolist())
    df["mapped_emotion"] = df["emotion"].map(EMOTION_MAPPING)
    return df.loc[df["mapped_emotion"].notna(), ["processed_text", "mapped_emotion"]]


def load_go_emotions() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "go_emotions_dataset.csv")
    df["emotion"] = df[GO_EMOTION_COLS].idxmax(axis=1)
    df["processed_text"] = batch_preprocess(df["text"].astype(str).tolist())
    df["mapped_emotion"] = df["emotion"].map(EMOTION_MAPPING)
    return df.loc[df["mapped_emotion"].notna(), ["processed_text", "mapped_emotion"]]


def load_val_txt() -> pd.DataFrame:
    rows = []
    with open(DATA_DIR / "val.txt") as f:
        for line in f:
            text, emotion = line.strip().split(";")
            mapped = EMOTION_MAPPING.get(emotion)
            if mapped is None:
                continue
            rows.append({"processed_text": preprocess_text(text), "mapped_emotion": mapped})
    return pd.DataFrame(rows)


def build_pipeline() -> Pipeline:
    base_models = [
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1)),
        ("lr", LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)),
    ]
    final_estimator = LogisticRegression(max_iter=500, random_state=42)
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=final_estimator,
        cv=3,
        n_jobs=-1,
    )
    return Pipeline([
        ("vectorizer", TfidfVectorizer(max_df=0.6, ngram_range=(1, 2))),
        ("classifier", stacking),
    ])


def main() -> None:
    print("Loading Dell tweets...")
    dell = load_dell_tweets()
    print(f"  {len(dell)} rows")

    print("Loading GoEmotions...")
    go = load_go_emotions()
    print(f"  {len(go)} rows")

    print("Loading val.txt...")
    val = load_val_txt()
    print(f"  {len(val)} rows")

    combined = pd.concat([dell, go, val], ignore_index=True).dropna()
    print(f"Combined: {len(combined)} rows, classes: {sorted(combined['mapped_emotion'].unique())}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(combined["mapped_emotion"])
    X_train, X_test, y_train, y_test = train_test_split(
        combined["processed_text"], y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    print("Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on held-out test set...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print(f"Saving model to {MODEL_PATH} and encoder to {ENCODER_PATH}...")
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    joblib.dump(label_encoder, ENCODER_PATH, compress=3)
    print(f"  model: {os.path.getsize(MODEL_PATH)/1e6:.1f} MB")
    print(f"  encoder: {os.path.getsize(ENCODER_PATH)/1e3:.1f} KB")
    print("Done.")


if __name__ == "__main__":
    main()
