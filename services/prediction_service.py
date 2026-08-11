import os
import streamlit as st

# Must be set before importing TensorFlow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import gdown

from transformers import (
    BertConfig,
    TFBertForSequenceClassification,
    BertTokenizer,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/tf_model.h5"

# Google Drive file ID for the trained BERT model
MODEL_FILE_ID = "1JvmGJPC0Xr7qTK1eZxkXP0zjrD-aD64P"

LABELS = [
    "Negative",
    "Neutral",
    "Positive",
]

MAX_SEQUENCE_LENGTH = 128


# --------------------------------------------------
# Download model if it does not exist
# --------------------------------------------------

def _download_model():
    """Download the trained BERT model from Google Drive."""

    os.makedirs("models", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        return

    print("Downloading BERT model...")
    print("This may take a few minutes.")

    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False,
    )

    print("BERT model downloaded successfully.")


# --------------------------------------------------
# Load and cache model + tokenizer
# --------------------------------------------------

@st.cache_resource
def load_model_and_tokenizer():
    """Load and cache the BERT model and tokenizer."""

    print("Loading BERT sentiment model...")

    _download_model()

    config = BertConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=3,
    )

    model = TFBertForSequenceClassification(config)

    # Build the model before loading weights

    dummy_input = {
        "input_ids": tf.zeros(
            (1, MAX_SEQUENCE_LENGTH),
            dtype=tf.int32,
        ),
        "attention_mask": tf.ones(
            (1, MAX_SEQUENCE_LENGTH),
            dtype=tf.int32,
        ),
        "token_type_ids": tf.zeros(
            (1, MAX_SEQUENCE_LENGTH),
            dtype=tf.int32,
        ),
    }

    model(dummy_input)

    # Load trained weights

    model.load_weights(MODEL_PATH)

    # Load tokenizer

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    print("BERT sentiment model loaded successfully.")

    return model, tokenizer


# Load model and tokenizer once and reuse them
model, tokenizer = load_model_and_tokenizer()


# --------------------------------------------------
# Single prediction
# --------------------------------------------------

def predict_sentiment(review):
    """
    Predict the sentiment of a single review.

    Returns:
        Negative, Neutral, or Positive
    """

    if not review or not str(review).strip():
        return None

    review = str(review).strip()

    inputs = tokenizer(
        review,
        return_tensors="tf",
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    outputs = model(
        inputs,
        training=False,
    )

    probabilities = tf.nn.softmax(
        outputs.logits,
        axis=-1,
    ).numpy()[0]

    predicted_index = int(probabilities.argmax())

    return LABELS[predicted_index]
