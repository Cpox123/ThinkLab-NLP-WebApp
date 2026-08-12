
import os

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

# Reviews forwarded to the model in a single tensor pass
BATCH_SIZE = 32


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
# Load model and tokenizer
# --------------------------------------------------

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


# --------------------------------------------------
# Batch prediction with probabilities
# --------------------------------------------------

def predict_batch_with_probabilities(reviews, progress_callback=None):
    """
    Predict sentiments for a list of reviews in batched tensor passes.

    Args:
        reviews: list of review texts (any type; empty/None treated as invalid).
        progress_callback: optional callable(done_count, total_count) invoked
            after every processed batch (valid reviews only).

    Returns:
        List with one element per input review:
            (label, probability_dict)  -> ("Positive", {"Negative": 0.01,
                                                         "Neutral": 0.05,
                                                         "Positive": 0.94})
        Empty or whitespace-only reviews get None.
    """

    texts = [str(r).strip() if r is not None else "" for r in reviews]
    results = [None] * len(texts)

    valid_indices = [i for i, text in enumerate(texts) if text]
    valid_texts = [texts[i] for i in valid_indices]

    total = len(valid_texts)

    for start in range(0, total, BATCH_SIZE):

        chunk_texts = valid_texts[start:start + BATCH_SIZE]
        chunk_indices = valid_indices[start:start + BATCH_SIZE]

        inputs = tokenizer(
            chunk_texts,
            return_tensors="tf",
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )

        outputs = model(inputs)

        batch_probabilities = tf.nn.softmax(
            outputs.logits,
            axis=-1,
        ).numpy()

        for position, probs in enumerate(batch_probabilities):

            index = chunk_indices[position]
            predicted_index = int(probs.argmax())

            results[index] = (
                LABELS[predicted_index],
                {
                    label: float(prob)
                    for label, prob in zip(LABELS, probs)
                },
            )

        if progress_callback is not None:
            progress_callback(
                min(start + BATCH_SIZE, total),
                total,
            )

    return results


# --------------------------------------------------
# Single prediction with probabilities
# --------------------------------------------------

def predict_with_probabilities(review):
    """
    Predict the sentiment of a single review.

    Returns:
        (label, probability_dict) or None for an empty/invalid review.
    """

    if not review or not str(review).strip():
        return None

    return predict_batch_with_probabilities(
        [str(review).strip()]
    )[0]


# --------------------------------------------------
# Single prediction (label only)
# --------------------------------------------------

def predict_sentiment(review):
    """
    Predict the sentiment of a single review.

    Returns:
        Negative, Neutral, or Positive
    """

    prediction = predict_with_probabilities(review)

    if prediction is None:
        return None

    label, _ = prediction

    return label
