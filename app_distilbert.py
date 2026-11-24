"""Gradio app backed by the fine-tuned DistilBERT model.

Loads from either:
  - a local directory ./distilbert-rant/ (produced by train_distilbert.py), or
  - the Hugging Face Hub (set MODEL_REPO env var).

No spaCy preprocessing — the DistilBERT tokenizer handles raw text.
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LOCAL_DIR = Path("distilbert-rant")
MODEL_REPO = os.environ.get("MODEL_REPO", "tt26586/rant-distilbert")

if LOCAL_DIR.exists():
    MODEL_SOURCE = str(LOCAL_DIR)
else:
    MODEL_SOURCE = MODEL_REPO

print(f"Loading model from: {MODEL_SOURCE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

id2label = model.config.id2label


@torch.inference_mode()
def classify(rant: str):
    if not rant or not rant.strip():
        return "— enter a rant to start —", {}
    enc = tokenizer(rant, truncation=True, max_length=256, return_tensors="pt").to(device)
    logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    top_idx = int(probs.argmax())
    label = id2label[top_idx]
    response = f"Your feelings of {label} are valid."
    prob_dict = {id2label[i]: float(p) for i, p in enumerate(probs)}
    return response, prob_dict


EXAMPLES = [
    "I ordered my coffee like I have 5 times before. When it arrived, grounds fell out of the bag and every box inside was crushed. A good portion of this delivery is destroyed.",
    "We bought this for our niece's birthday. She loved it! Thank you so much. Fast shipping, just as described.",
    "I can't believe this happened again. I waited an hour and they messed up my order.",
    "Honestly, I was skeptical at first but this turned out to be exactly what I needed.",
    "I just found out my grandfather passed away last night. Everything feels numb.",
]

with gr.Blocks(title="Rant! (DistilBERT)") as demo:
    gr.Markdown(
        "# Rant! — Emotion Classifier (DistilBERT)\n"
        "Paste a rant. Get the emotion behind it + a validating response.\n"
        "Fine-tuned `distilbert-base-uncased` on ~243K labeled posts from GoEmotions, "
        "labeled tweets, and an emotion text corpus. "
        "Classes: anger, fear, joy, love, neutral, sad, surprise."
    )
    with gr.Row():
        with gr.Column():
            rant = gr.Textbox(
                label="Your rant",
                placeholder="Tell me what happened...",
                lines=6,
            )
            btn = gr.Button("Classify", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=rant)
        with gr.Column():
            response = gr.Textbox(label="Response", interactive=False)
            probs = gr.Label(label="Class probabilities", num_top_classes=4)

    btn.click(classify, inputs=rant, outputs=[response, probs])
    rant.submit(classify, inputs=rant, outputs=[response, probs])


if __name__ == "__main__":
    demo.launch()
