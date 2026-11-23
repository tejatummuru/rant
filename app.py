"""Gradio app: paste a rant, get its emotion + a validating response.

Loads model artifacts produced by train.py. Preprocessing is identical to
training (see preprocess.py) so the model sees the same feature distribution
it was trained on.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import joblib

from preprocess import preprocess_text

MODEL_PATH = Path("model.joblib")
ENCODER_PATH = Path("label_encoder.joblib")

if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
    raise SystemExit(
        f"Missing {MODEL_PATH} or {ENCODER_PATH}. Run `python train.py` first "
        "(requires the three data files described in README.md)."
    )

pipeline = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


def classify(rant: str):
    if not rant or not rant.strip():
        return "— enter a rant to start —", ""
    processed = preprocess_text(rant)
    pred = pipeline.predict([processed])[0]
    label = label_encoder.inverse_transform([pred])[0]
    probs = {}
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba([processed])[0]
            probs = {
                label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(proba)
            }
        except Exception:
            probs = {}
    response = f"Your feelings of {label} are valid."
    return response, probs


EXAMPLES = [
    "I ordered my coffee like I have 5 times before. When it arrived, grounds fell out of the bag and every box inside was crushed. A good portion of this delivery is destroyed.",
    "We bought this for our niece's birthday. She loved it! Thank you so much. Fast shipping, just as described.",
    "I can't believe this happened again. I waited an hour and they messed up my order.",
    "Honestly, I was skeptical at first but this turned out to be exactly what I needed.",
    "I just found out my grandfather passed away last night. Everything feels numb.",
]

with gr.Blocks(title="Rant!") as demo:
    gr.Markdown(
        "# Rant! — Emotion Classifier\n"
        "Paste a rant. Get the emotion behind it + a validating response.\n"
        "Trained on ~240K labeled posts from GoEmotions, labeled tweets, and an emotion text corpus. "
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
