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
**Model on HF Hub:** https://huggingface.co/tt26586/rant-distilbert

## Problem

Classify a short-to-medium piece of free-form text (a review, a vent, a tweet) into one of 7 core emotions: `anger`, `fear`, `joy`, `love`, `neutral`, `sad`, `surprise`.

## Data

Trained on ~243K labeled rows merged from three public corpora, with their labels unified onto a 7-emotion schema.

| Source | Original labels | Rows |
| --- | --- | --- |
| [Sentiment-Emotion-labelled Dell tweets](https://www.kaggle.com/datasets/) | 8 emotions | ~20K |
| [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (Reddit comments) | 28 emotions | ~210K |
| [Emotion text corpus](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) (`val.txt`) | 6 emotions | ~20K |

Label mapping (28 → 7) lives in `preprocess.py` (`EMOTION_MAPPING`). Same mapping is applied across all three sources, including `val.txt` (an earlier bug let `val.txt` labels bypass the mapping — fixed).

## Models

Two models, same data, directly comparable. DistilBERT is what the live Space serves; the sklearn stack is kept as the baseline.

### Baseline — TF-IDF + stacked ensemble

- Preprocessing: lowercase → strip handles/hashtags/URLs → spaCy tokenize + lemmatize → remove stopwords/punctuation. Batched with `nlp.pipe(n_process=-1)`.
- Features: TF-IDF (unigrams + bigrams, `max_df=0.6`).
- Classifier: `StackingClassifier` with `RandomForestClassifier` (class-balanced, `n_estimators=100`, `max_depth=15`) + `LogisticRegression` base learners, `LogisticRegression` meta-learner.
- Imbalance handled via `class_weight='balanced'`.

Train with `python train.py` (produces `model.joblib` + `label_encoder.joblib`). Serve with `python app.py`.

### Shipped — fine-tuned DistilBERT

- Base model: `distilbert-base-uncased`.
- 3 epochs, batch 16, lr 2e-5, max_length 128, AdamW, cross-entropy loss.
- 80/10/10 train/val/test split, stratified by label. Best checkpoint selected by weighted F1 on the val split.
- No spaCy preprocessing — the tokenizer handles raw text.

Train with `python train_distilbert.py --push tt26586/rant-distilbert`. Serve with `python app_distilbert.py`.

### Results (held-out test set, 25,103 rows)

| Class | Baseline F1 | DistilBERT F1 | Δ |
| --- | --- | --- | --- |
| anger | 0.59 | **0.68** | +9 |
| fear | 0.54 | **0.61** | +7 |
| joy | 0.62 | **0.65** | +3 |
| love | 0.52 | **0.59** | +7 |
| neutral | 0.54 | 0.54 | 0 |
| sad | 0.52 | **0.58** | +6 |
| surprise | 0.20 | **0.29** | +9 |
| **weighted F1** | 0.55 | **0.59** | +4 |
| **macro F1** | 0.50 | **0.56** | +6 |
| **accuracy** | 0.55 | **0.59** | +4 |

DistilBERT beats the baseline on every class and improves weighted F1 by 4 points. The gap is smaller than I initially projected — likely because of GoEmotions label noise (many posts have multiple valid emotions, but we force a single top-1 target) and 128-token truncation. Longer `max_length`, a larger base model (`roberta-base`, `deberta-v3-base`), or dropping/merging the near-unusable `surprise` class would plausibly push it further.

## Repo layout

```
preprocess.py                    # text cleaning + shared 28->7 label map
train.py                         # baseline stacked ensemble -> model.joblib
app.py                           # Gradio UI for the baseline (loads .joblib)
train_distilbert.py              # fine-tune distilbert-base-uncased
app_distilbert.py                # Gradio UI for DistilBERT (loads from HF Hub)
rant-emotion-classifier.ipynb    # original EDA / training notebook
requirements.txt                 # baseline deps
requirements-distilbert.txt      # extra deps for DistilBERT path
```

## Run locally

Baseline (sklearn):

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# put the 3 data files in ./data/ (see Data section above)
python train.py          # trains and saves model.joblib
python app.py            # launches Gradio UI at http://127.0.0.1:7860
```

DistilBERT:

```bash
pip install -r requirements.txt -r requirements-distilbert.txt
python train_distilbert.py                                 # trains, saves locally
# OR, train and push to your HF Hub:
python train_distilbert.py --push <your-username>/rant-distilbert
python app_distilbert.py                                   # launches Gradio UI
```

Expected training time for DistilBERT:
- Colab free T4 GPU: ~30 min
- Apple MPS (M1/M2/M3): 2–4 hours
- CPU only: not recommended

## Deploy to Hugging Face Spaces

The live Space serves DistilBERT. The deploy flow:

1. Push the fine-tuned model to HF Hub (`train_distilbert.py --push ...` does this).
2. Create a Gradio Space in the HF web UI.
3. Clone the Space, copy `app_distilbert.py` into it as `app.py`, copy `preprocess.py`, `requirements.txt`, and this `README.md`. Append the DistilBERT deps.
4. `git push` — the Space builds, pulls the model from HF Hub at startup, serves inference.

No model file in the Space repo — much smaller, and the model can be versioned independently on HF Hub.

## Known limitations

- `surprise` is still the worst-performing class even for DistilBERT (F1 0.29). It's small and overlaps with `joy` in GoEmotions annotations. Realistic next step: drop it or merge into `joy`.
- Both models are context-free. A production version would prefix the input with source/placement metadata (e.g., `source=reddit`, `length=short`).
- 128-token max length truncates ~20% of inputs. Bumping to 256 would help marginally.
