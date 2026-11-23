"""Text preprocessing shared between training and inference.

CRITICAL: inference must apply the exact same preprocessing as training,
otherwise the model sees a different feature distribution and predictions
are garbage. (Previously the demo cell skipped this and got nonsense.)
"""
from __future__ import annotations

import re

import spacy


_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _NLP


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    doc = _get_nlp()(text)
    return " ".join(
        tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_punct
    )


def batch_preprocess(texts, batch_size: int = 500):
    """Parallelized version for training-time corpus prep."""
    nlp = _get_nlp()
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=-1):
        out.append(
            " ".join(
                tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_punct
            )
        )
    return out


# Shared 28 -> 7 emotion mapping. Used for GoEmotions and val.txt.
EMOTION_MAPPING = {
    "anger": "anger",
    "surprise": "surprise",
    "disgust": "sad",
    "joy": "joy",
    "sadness": "sad",
    "admiration": "love",
    "amusement": "joy",
    "annoyance": "sad",
    "approval": "joy",
    "caring": "joy",
    "confusion": "neutral",
    "curiosity": "joy",
    "desire": "love",
    "disappointment": "sad",
    "disapproval": "sad",
    "embarrassment": "sad",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sad",
    "love": "love",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sad",
    "neutral": "neutral",
}
