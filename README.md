# Rant! — Emotion Classifier

An NLP classifier that reads a user's rant and names the emotion behind it. Returns a validating response (e.g., "Your feelings of anger are valid.").

## Problem

Given a short-to-medium piece of free-form text (a review, a vent, a tweet), classify it into one of 7 core emotions: `anger`, `fear`, `joy`, `love`, `neutral`, `sad`, `surprise`.

## Data

Trained on **~243K labeled rows** merged from three public corpora, with their labels unified onto a 7-emotion schema.

| Source | Original labels | Rows |
| --- | --- | --- |
| [Sentiment-Emotion-labelled Dell tweets](https://www.kaggle.com/datasets/) | 8 emotions | ~2K |
| [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (Reddit comments) | 28 emotions | ~210K |
| [Emotion text corpus](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) (`val.txt`) | 6 emotions | ~30K |

### Label mapping (28 → 7)

GoEmotions' fine-grained labels are collapsed onto the shared 7-class schema. See `updated_emotion_mapping` in the notebook.

## Model

- Preprocessing: lowercase → strip handles/hashtags/URLs → spaCy tokenize + lemmatize → remove stopwords/punctuation. Batched with `nlp.pipe(n_process=-1)` for throughput.
- Features: TF-IDF (unigrams + bigrams, `max_df=0.6`)
- Classifier: `StackingClassifier` with `RandomForestClassifier` (class-balanced, `n_estimators=100`, `max_depth=15`) + `LogisticRegression` base learners, `LogisticRegression` meta-learner
- Hyperparameter search: `RandomizedSearchCV` over vectorizer + RF params on a 2K subset
- Imbalance handled via `class_weight='balanced'`; `SMOTE` tried but not in the final pipeline

## Results

Weighted F1 ~0.54 on the 8-class split in the notebook. Known issue: `sadness` and `sad` appear as separate classes due to a mapping bug in the `val.txt` ingestion path (fixed in the latest commit; re-run to see the corrected 7-class numbers).

| Class | F1 (notebook) |
| --- | --- |
| joy | 0.62 |
| sadness → sad | 0.47–0.80 (two leaks) |
| neutral | 0.53 |
| anger | 0.58 |
| fear | 0.66 |
| love | 0.43 |
| surprise | 0.20 |

`surprise` and `love` underperform — likely from class imbalance (surprise is small) and overlap with `joy` in GoEmotions.

## Run it

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook rant-emotion-classifier.ipynb
```

Place the three data files in a `data/` directory:

```
data/
├── sentiment-emotion-labelled_Dell_tweets.csv
├── go_emotions_dataset.csv
└── val.txt
```

## Known limitations / next steps

- TF-IDF + stacking caps out around 0.55 weighted F1. Fine-tuning `distilbert-base-uncased` on the same data should land ~0.75–0.80.
- No held-out validation split; current numbers are train/test only.
- `surprise` is near-unusable at 0.20 F1 — needs targeted augmentation or dropping the class.
- Demo cell originally skipped preprocessing on inference input (training/inference mismatch); fixed.
