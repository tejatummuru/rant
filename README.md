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

**Live demo:** https://huggingface.co/spaces/tt26586/rant-emotion-classifier

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

## DistilBERT upgrade (in progress)

A `train_distilbert.py` script is included that fine-tunes `distilbert-base-uncased` on the same data + 28→7 mapping, so results are directly comparable to the sklearn baseline above.

Install the extra deps:

```bash
pip install -r requirements-distilbert.txt
```

Train (auto-detects CUDA / Apple MPS / CPU):

```bash
python train_distilbert.py                              # 3 epochs, local save
python train_distilbert.py --push tt26586/rant-distilbert  # also push to HF Hub
```

Expected runtime:
- Colab free T4: ~45–60 min
- Apple MPS (M1/M2/M3): 2–4 hours
- CPU-only: not recommended (overnight)

Expected accuracy: weighted F1 ~0.72–0.78 (vs. 0.55 for the sklearn baseline).

Once trained, swap to the DistilBERT app:

```bash
python app_distilbert.py
```

For the live Space, push the fine-tuned model to HF Hub and replace `app.py` with `app_distilbert.py` in the Space repo.

## Known limitations

- `surprise` class is near-unusable in the baseline (F1 ~0.20) — fine-tuning should help, but may still underperform due to class imbalance.
- No validation split beyond train/val/test (80/10/10 for DistilBERT, 80/20 for sklearn).
- Both models treat text as context-free; a better approach would prefix the input with placement/source metadata.
