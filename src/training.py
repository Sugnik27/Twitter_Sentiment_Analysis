"""
Training module (Deep Learning).

- loads preprocessed data
- trains RNN / LSTM / GRU models
- performs hyperparameter tuning
- saves per-model best
- saves overall best model
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import shutil

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.preprocessing import prepare_data_for_training
from src.config import (
    MAX_VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    NUM_CLASSES,
    EPOCHS,
    BATCH_SIZE,
    MODEL_DIR,
)

print("training file loaded")

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)


# -------------------------------------------------
# Model registry
# -------------------------------------------------
def get_model_builders():
    return {
        "rnn": SimpleRNN,
        "lstm": LSTM,
        "gru": GRU
    }


# -------------------------------------------------
# Model builder
# -------------------------------------------------
def build_model(hp, rnn_layer):

    model = Sequential()
    model.add(
        Embedding(
            input_dim=MAX_VOCAB_SIZE,
            output_dim=hp.Choice("embed_dim", [64, 128]),
        )
    )

    model.add(
        rnn_layer(
            units=hp.Int("units", 64, 256, step=64)
        )
    )

    model.add(
        Dropout(
            hp.Float("dropout", 0.2, 0.5, step=0.1)
        )
    )

    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("lr", [1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -------------------------------------------------
# Training loop
# -------------------------------------------------
def train_and_select_model():

    # 🔧 Clear old tuner logs
    tuner_root = MODEL_DIR / "tuner_logs"
    if tuner_root.exists():
        shutil.rmtree(tuner_root)

    print("STARTING DL MODEL TRAINING")

    X_train, X_val, y_train, y_val = prepare_data_for_training()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    best_overall = None
    best_score = -1.0
    best_name = None

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    )

    for name, rnn_layer in get_model_builders().items():
        print(f"\n--- Training {name.upper()} ---")

        tuner = kt.RandomSearch(
            lambda hp: build_model(hp, rnn_layer),
            objective="val_accuracy",
            max_trials=5,
            executions_per_trial=1,
            directory=MODEL_DIR / "tuner_logs",
            project_name=name
        )

        tuner.search(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1
        )

        # Best model
        best_model = tuner.get_best_models(1)[0]

        # SAFE val_accuracy extraction
        best_trial = tuner.oracle.get_best_trials(1)[0]
        history = best_trial.metrics.get_history("val_accuracy")

        if not history:
            print(f"No val_accuracy history for {name}, skipping")
            continue

        val_acc = max(
            float(v[0] if isinstance(v, list) else v)
            for v in (m.value for m in history)
        )

        # Save per-model best
        model_path = MODEL_DIR / f"{name}_best_model.keras"
        best_model.save(model_path)
        print(f"Saved {name} -> {model_path}")

        results.append({
            "model": name,
            "val_accuracy": val_acc
        })

        # Track overall best (SAFE)
        if val_acc > best_score:
            best_score = val_acc
            best_overall = best_model
            best_name = name

    # Save overall best model
    if best_overall is not None:
        overall_path = MODEL_DIR / "best_overall_model.keras"
        best_overall.save(overall_path)
        print(
            f"\nOverall best model: {best_name} "
            f"(val_accuracy={best_score:.4f}) -> {overall_path}"
        )

    return pd.DataFrame(results)


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    df_results = train_and_select_model()
    print("\nSummary:\n", df_results)
