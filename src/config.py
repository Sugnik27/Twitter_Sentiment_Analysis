"""
Basic configuration and paths for the project.
"""

from pathlib import Path

# ROOT DIRECTORY

BASE_DIR: Path = Path(__file__).resolve().parents[1]


# DATA PATHS

DATA_DIR: Path = Path(BASE_DIR / "data")

RAW_DATA_PATH: Path = Path(DATA_DIR / "Twitter_Data.csv")
CLEANED_DATA_PATH: Path = Path(DATA_DIR / "clean_tweets.csv")


# MODEL AND ARTIFACT PATHS

MODEL_DIR: Path = Path(BASE_DIR / "models")
BEST_MODEL_PATH:Path = Path(MODEL_DIR / "best_overall_model.keras")


TOKENIZER_DIR: Path = Path(BASE_DIR / "tokenizer")
TOKENIZER_PATH: Path = Path(TOKENIZER_DIR / "tokenizer.pkl")

LABEL_ENCODER_PATH: Path = Path(MODEL_DIR / "label_encoder.pkl")


# TARGET AND TEXT CONFIG>

TEXT_COLUMN: str = "text"
TARGET_COLUMN: str = "sentiment"

NUM_CLASSES: int = 3


# TOKENIZATION AND SEQUENCE CONFIG.

MAX_VOCAB_SIZE: int = 20000
MAX_SEQUENCE_LENGTH: int = 100
OOV_TOKEN: str = "<OOV>"


# TRAIN/ VALIDATION CONFIG

TEST_SIZE: int = 0.2
RANDOM_STATE: int = 42

BATCH_SIZE: int = 32
EPOCHS: int = 10


# HYPERPARAMETER TUNING (KERAS TUNER)

TUNER_PROJECT_NAME: str = "sentiment_tuning"
TUNER_MAX_TRIALS: int = 10
TUNER_EXECUTION_PER_TRIAL: int = 1

MONITOR_MTRIC: str = "val_accuracy"


# CREATE REQUIRED DIRECTORIES

for path in [
    DATA_DIR,
    MODEL_DIR,
    TOKENIZER_DIR
]:
    path.mkdir(parents = True, exist_ok= True)


print ("Configuration loaded successfully.")