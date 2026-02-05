"""
Deployment utilities for
Twitter Post Sentiment Analysis

- load_model: lazy-load trained DL model
- load_tokenizer: lazy-load tokenizer
- load_label_encoder: lazy-load label encoder
- preprocess_text: tokenize + pad input
- predict_single: predict sentiment for one text
- predict_batch: batch sentiment predictions
"""

from typing import Any, Dict, List
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import (
    BEST_MODEL_PATH,
    TOKENIZER_PATH,
    LABEL_ENCODER_PATH,
    MAX_SEQUENCE_LENGTH
)

# -------------------------------------------------
# CACHES
# -------------------------------------------------

_model = None
_tokenizer = None
_label_encoder = None


# -------------------------------------------------
# LOADERS (lazy)
# -------------------------------------------------

def load_model():
    global _model
    if _model is None:
        if not Path(BEST_MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model not found at {BEST_MODEL_PATH}"
            )
        _model = tf.keras.models.load_model(BEST_MODEL_PATH)
    return _model


def load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        if not Path(TOKENIZER_PATH).exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {TOKENIZER_PATH}"
            )
        _tokenizer = joblib.load(TOKENIZER_PATH)
    return _tokenizer


def load_label_encoder():
    global _label_encoder
    if _label_encoder is None:
        if not Path(LABEL_ENCODER_PATH).exists():
            raise FileNotFoundError(
                f"LabelEncoder not found at {LABEL_ENCODER_PATH}"
            )
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return _label_encoder


# -------------------------------------------------
# TEXT PREPROCESSING
# -------------------------------------------------

def preprocess_text(texts: List[str]) -> np.ndarray:
    """
    Converts raw text to padded sequences
    """

    tokenizer = load_tokenizer()

    # Safety: enforce string
    texts = ["" if t is None else str(t) for t in texts]

    sequences = tokenizer.texts_to_sequences(texts)

    padded = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    return padded


# -------------------------------------------------
# SINGLE PREDICTION
# -------------------------------------------------

def predict_single(text: str) -> Dict[str, Any]:
    """
    Predict sentiment for a single text
    """

    model = load_model()
    label_encoder = load_label_encoder()

    X = preprocess_text([text])

    probs = model.predict(X, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    sentiment = label_encoder.inverse_transform([pred_idx])[0]

    # CONFIDENCE THRESHOLD FIX 
    if confidence < 0.55:
        sentiment = "neutral"

    return {
        "text": text,
        "predicted_sentiment": sentiment,
        "confidence": confidence,
        "probabilities": {
            label: float(prob)
            for label, prob in zip(label_encoder.classes_, probs)
        }
    }


# -------------------------------------------------
# BATCH PREDICTION
# -------------------------------------------------

def predict_batch(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Predict sentiment for a batch of texts
    """

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found")

    model = load_model()
    label_encoder = load_label_encoder()

    texts = df[text_column].fillna("").astype(str).tolist()
    X = preprocess_text(texts)

    probs = model.predict(X, verbose=0)
    preds = np.argmax(probs, axis=1)
    confidences = probs.max(axis=1)

    sentiments = label_encoder.inverse_transform(preds)

    out = df.copy()
    out["predicted_sentiment"] = sentiments
    out["confidence"] = confidences


    return out
