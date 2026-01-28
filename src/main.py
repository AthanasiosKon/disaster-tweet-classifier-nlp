import random
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

from config import *
from preprocess import clean_text
from tokenizer_utils import create_tokenizer, tokenize_and_pad
from model import build_model
from train import train_model
from evaluate import evaluate_model
from baseline import run_baseline
from plots import plot_training


# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load data
df = pd.read_csv("data/train.csv")
df = clean_text(df)

X = df["text"].values
y = df["target"].values

# Train / Val / Test split
# Test set: 10% of total
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)

# Train / Validation split: 20% of the remaining
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VAL_SIZE, random_state=SEED
)

run_baseline(X_train, X_val, y_train, y_val)

# Tokenization
tokenizer = create_tokenizer(X_train, VOCAB_SIZE)

X_train_pad, _ = tokenize_and_pad(tokenizer, X_train, MAX_LENGTH)
X_val_pad, _ = tokenize_and_pad(tokenizer, X_val, MAX_LENGTH)
X_test_pad, _ = tokenize_and_pad(tokenizer, X_test, MAX_LENGTH)

# Model
model = build_model(
    VOCAB_SIZE,
    MAX_LENGTH,
    EMBEDDING_DIM,
    LSTM_UNITS
)

# Train
history = train_model(
    model,
    X_train_pad,
    y_train,
    X_val_pad,
    y_val,
    EPOCHS,
    BATCH_SIZE
)

plot_training(history)


# Evaluate
evaluate_model(model, X_test_pad, y_test)

# Optional: Save model
# model.save("models/lstm_disaster_classifier")
