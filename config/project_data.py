PROJECT_TITLE = "NLP-based Product Review Sentiment Classification System"

# --------------------------------------------------
# Branding
# --------------------------------------------------

APP_NAME = "NLP Sentiment Analyzer"
GROUP_NAME = "NLP Group 05"
COURSE = "CCS3356 Natural Language Processing"
COPYRIGHT_YEAR = "2026"

# --------------------------------------------------
# Dataset
# --------------------------------------------------

DATASET_NAME = "Women's E-Commerce Clothing Reviews"
DATASET_SOURCE = "Kaggle"
DATASET_REVIEW_COUNT = "23,486"

# Approximate label distribution of the dataset, derived from star ratings
# (Positive = 4-5 stars, Neutral = 3 stars, Negative = 1-2 stars)
DATASET_SENTIMENT_SPLIT = {
    "Positive": 77.5,
    "Neutral": 12.2,
    "Negative": 10.3,
}

# --------------------------------------------------
# Model configuration
# --------------------------------------------------

LABEL_ORDER = ["Negative", "Neutral", "Positive"]

MAX_SEQUENCE_LENGTH = 128

BULK_ROW_LIMIT = 500

# Maximum characters accepted for a single review
MAX_REVIEW_CHARS = 1000

# Column names accepted by the bulk CSV page (matched case-insensitively)
REVIEW_COLUMN_CANDIDATES = [
    "Review Text",
    "review",
    "Review",
    "text",
    "Text",
    "sentence",
    "content",
]

MODEL_RESULTS = {
    "Logistic Regression": {
        "accuracy": 0.7674,
        "macro_f1": 0.6010,
    },
    "LSTM": {
        "accuracy": 0.7022,
        "macro_f1": 0.4408,
    },
    "SVM": {
        "accuracy": 0.8206,
        "macro_f1": 0.5900,
    },
    "BERT": {
        "accuracy": 0.8182,
        "macro_f1": 0.6600,
    },
    "Naive Bayes": {
        "accuracy": 0.7813,
        "macro_f1": 0.5943,
    },
    "CNN": {
        "accuracy": 0.7941,
        "macro_f1": 0.6227,
    },
}

FINAL_MODEL_NAME = "BERT"

MODEL_TYPES = {
    "Logistic Regression": "ML",
    "LSTM": "DL",
    "SVM": "ML",
    "BERT": "DL",
    "Naive Bayes": "ML",
    "CNN": "DL",
}

MODEL_INFO = {
    "Model": "BERT (base-uncased)",
    "Type": "Transformer (DL)",
    "Max Sequence Length": str(MAX_SEQUENCE_LENGTH),
    "Trained On": DATASET_NAME,
}

# --------------------------------------------------
# NLP pipeline steps (About page)
# --------------------------------------------------

PIPELINE_STEPS = [
    ("📥", "Data Collection"),
    ("🧹", "Preprocessing"),
    ("🔍", "EDA"),
    ("⚙️", "Feature Engineering"),
    ("🤖", "Model Development"),
    ("📏", "Evaluation"),
    ("🚀", "Deployment"),
]

# --------------------------------------------------
# Team (edit with your real details)
# --------------------------------------------------

TEAM_MEMBERS = [
    {
        "name": "Member 1",
        "sid": "2023/CS/105",
        "role": "Data Collection, Preprocessing & ML Models",
    },
    {
        "name": "Member 2",
        "sid": "2023/CS/112",
        "role": "DL Models, Evaluation & Results",
    },
    {
        "name": "Member 3",
        "sid": "2023/CS/118",
        "role": "Integration, App Development & Deployment",
    },
]
