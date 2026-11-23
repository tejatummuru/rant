---
title: Rant Emotion Classifier
emoji: "\U0001F621"
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Rant! — Emotion Classifier

An NLP classifier that reads a user's rant and names the emotion behind it. Returns a validating response (e.g., *"Your feelings of anger are valid."*).

Live demo: _after you deploy to Hugging Face Spaces (see below), paste the Space URL here._

## Problem

Classify a short-to-medium piece of free-form text (a review, a vent, a tweet) into one of 7 core emotions: `anger`, `fear`, `joy`, `love`, `neutral`, `sad`, `surprise`.

## Data

Trained on ~243K labeled rows merged from three public corpora, with their labels unified onto a 7-emotion schema.

| Source | Original labels | Rows |
| --- | --- | --- |
| [Sentiment-Emotion-labelled Dell tweets](https://www.kaggle.com/datasets/) | 8 emotions | ~2K |
| [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (Reddit comments) | 28 emotions | ~210K |
| [Emotion text corpus](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) (`val.txt`) | 6 emotions | ~30K |

Label mapping (28 → 7) lives in `preprocess.py` (`EMOTION_MAPPING`).

## Model

- Preprocessing: lowercase → strip handles/hashtags/URLs → spaCy tokenize + lemmatize → remove stopwords/punctuation. Batched with `nlp.pipe(n_process=-1)`.
- Features: TF-IDF (unigrams + bigrams, `max_df=0.6`).
- Classifier: `StackingClassifier` with `RandomForestClassifier` (class-balanced, `n_estimators=100`, `max_depth=15`) + `LogisticRegression` base learners, `LogisticRegression` meta-learner.
- Imbalance handled via `class_weight='balanced'`.

Target accuracy: weighted F1 around 0.55–0.60 on the 7-class split. `surprise` and `love` underperform — likely class imbalance and label overlap in GoEmotions. Future work: fine-tune DistilBERT for a projected ~0.75–0.80 F1.

## Repo layout

```
preprocess.py                # text cleaning + the shared 28->7 label map
train.py                     # trains the stacked ensemble and saves artifacts
app.py                       # Gradio UI, loads artifacts and runs inference
rant-emotion-classifier.ipynb  # original EDA / training notebook
requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Place the three data files under `./data/`:

```
data/
├── sentiment-emotion-labelled_Dell_tweets.csv
├── go_emotions_dataset.csv
└── val.txt
```

Train once (produces `model.joblib` and `label_encoder.joblib`):

```bash
python train.py
```

Launch the Gradio app:

```bash
python app.py
```

Then open the URL it prints (usually `http://127.0.0.1:7860`).

## Deploy to Hugging Face Spaces

1. Create a free account at https://huggingface.co.
2. Create a new Space: **New → Space**. Pick the Gradio SDK.
3. Clone the Space locally: `git clone https://huggingface.co/spaces/<your-username>/<space-name>`.
4. Copy `app.py`, `preprocess.py`, `requirements.txt`, `README.md`, `model.joblib`, and `label_encoder.joblib` into the Space clone.
5. Push: `git add -A && git commit -m "Initial deploy" && git push`.
6. The Space auto-builds; demo lives at `https://huggingface.co/spaces/<your-username>/<space-name>`.

If `model.joblib` is > 10 MB, HF will auto-track it with Git LFS. If it's > 100 MB, you may need to either compress harder (`joblib.dump(..., compress=9)`), reduce `n_estimators` in `train.py`, or host the artifact on HF Hub and load it from there at startup.

## Known limitations

- TF-IDF + stacking caps around 0.55 weighted F1. Fine-tuning `distilbert-base-uncased` should land ~0.75–0.80.
- No validation split beyond train/test (80/20).
- `surprise` class is near-unusable (F1 ~0.20) — needs targeted augmentation or merging into `joy`.
