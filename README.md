# ThinkLab NLP — Product Review Sentiment Analyzer

A Streamlit web app that classifies product reviews (Women's E-Commerce Clothing
Reviews dataset) into **🔴 Negative**, **🟡 Neutral**, and **🟢 Positive**
sentiments using a fine-tuned BERT model.

## Troubleshooting

**Console full of warnings? Safe to ignore — these are normal:**

- `oneDNN custom operations are on...` / `SSE3 SSE4.1 AVX2 ...` — TensorFlow CPU performance notices
- `The name tf.losses... is deprecated` / `tf.get_default_graph is deprecated` — deprecation notices from the legacy Keras compatibility layer (`tf-keras`), needed by the model
- `FutureWarning: resume_download is deprecated...` — from huggingface_hub internals
- `Could not find cuda drivers...` — expected on machines without a GPU; the app intentionally runs on CPU

The only line that matters is:

```
BERT sentiment model loaded successfully.
```

**App won't start / nothing shows?** Remember: `streamlit run app.py`
(or `python -m streamlit run app.py`) — never plain `python app.py`.

## Features

- **Single Prediction** — classify one review with a color-coded result card
- **Bulk Prediction** — upload a CSV (must contain a `Review Text` column),
  classify up to 500 reviews at once, view the sentiment distribution, and
  download the results
- **Dashboard** — performance comparison of all six evaluated models
  (Logistic Regression, LSTM, SVM, BERT, Naive Bayes, CNN)

## Run locally

### Prerequisites

- Python 3.10 – 3.13
- ~3 GB free disk space (TensorFlow + model weights)
- Internet connection (first run downloads the 438 MB trained model)

### Steps

```bash
# 1. Enter the project folder
cd ThinkLab-NLP-WebApp

# 2. (Recommended) create a virtual environment
python -m venv .venv
#    activate it:
#      Windows:  .venv\Scripts\activate
#      Mac/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app — it opens at http://localhost:8501
streamlit run app.py
```

> ⚠️ **Do NOT run `python app.py`.** Streamlit apps must be started with
> `streamlit run ...`, otherwise the script just executes once (you will see
> many `missing ScriptRunContext` and
> `Session state does not function when running a script without streamlit run`
> warnings) and no web server starts.
>
> If Windows says `streamlit is not recognized`, use
> `python -m streamlit run app.py` instead.

> **First run:** the trained BERT weights (`models/tf_model.h5`, ~438 MB) are
> downloaded automatically from Google Drive. This only happens once.
>
> **Tip for Linux:** if the full `tensorflow` install is too heavy, you can
> replace it with `tensorflow-cpu==2.21.0` (same API, no CUDA drivers).

## Project structure

```
ThinkLab-NLP-WebApp/
├── app.py                        # Home page
├── pages/
│   ├── 1_Single_Prediction.py    # Single review prediction
│   ├── 2_Bulk_Prediction.py      # Bulk CSV prediction
│   ├── 3_Dashboard.py            # Model comparison
│   └── 4_About.py                # About / ethics / limitations
├── services/
│   ├── prediction_service.py     # BERT model loading + inference
│   ├── bulk_service.py           # Batch prediction helpers
│   └── ui_service.py             # Shared colors & display components
├── config/
│   └── project_data.py           # Project constants & model results
└── requirements.txt
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select the repo, branch `main`, main file `app.py` → **Deploy**

The first cloud boot also downloads the model from Google Drive
automatically.
