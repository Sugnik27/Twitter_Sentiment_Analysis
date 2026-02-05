"""
Preprocessing pipeline for Deep Learning sentiment analysis.
Handles:
- Label encoding
- Tokenization
- Padding
"""


import pandas as pd
import numpy as np
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import (
    CLEANED_DATA_PATH,
    TEXT_COLUMN,
    TARGET_COLUMN,
    MAX_VOCAB_SIZE,
    MAX_SEQUENCE_LENGTH,
    OOV_TOKEN,
    TEST_SIZE,
    RANDOM_STATE,
    TOKENIZER_PATH,
    LABEL_ENCODER_PATH,
)


# LOADING CLEAN DATA

def load_clean_data() -> pd.DataFrame:

    """
    loads the cleaned dataset.
    """

    return pd.read_csv(CLEANED_DATA_PATH)



# ENCODE TARGET LABELS

def encode_labels(y: pd.Series) -> np.ndarray:

    """
    Encodes sentiment labels into integers.
    Saves the label encoder.
    """

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    return y_encoded




# TOKENIZER AND PADDING

def tokenizer_and_pad(texts: pd.Series) -> np.ndarray:

    """
    Tokenizes and pads text sequences.
    Saves the tokenizer.
    """
    texts = texts.fillna("").astype(str)

    tokenizer = Tokenizer(
        num_words = MAX_VOCAB_SIZE,
        oov_token = OOV_TOKEN
    )

    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    padded_sequences = pad_sequences(
        sequences,
        maxlen = MAX_SEQUENCE_LENGTH,
        padding = "post",
        truncating = "post"
    )

    joblib.dump(tokenizer, TOKENIZER_PATH)

    return padded_sequences


# FULL PREPROCESSING PIPELINE

def prepare_data_for_training():

    """
     Full preprocessing pipeline:
    - Load clean data
    - Encode labels
    - Tokenize & pad text
    - Train-test split

    Returns:
        X_train, X_val, y_train, y_val
    """

    df = load_clean_data()

    X = df[TEXT_COLUMN]
    y = df[TARGET_COLUMN]

    y_encoded = encode_labels(y)
    X_padded = tokenizer_and_pad(X)

    X_train, X_val, y_train, y_val = train_test_split(X_padded, y_encoded, test_size= TEST_SIZE, random_state= RANDOM_STATE, stratify= y_encoded)

    return X_train, X_val, y_train, y_val

print ("Preprocessing created successfully")