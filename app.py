import streamlit as st
from config.project_data import PROJECT_TITLE, DATASET_NAME

st.set_page_config(
    page_title="ThinkLab Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
)

st.title("ThinkLab NLP Sentiment Analyzer")

st.write(
    "Product Review Sentiment Classification System"
)

st.subheader("Project Overview")

st.write(
    f"This application classifies product reviews into "
    f"Negative, Neutral, and Positive sentiments."
)

st.write(f"Dataset: {DATASET_NAME}")

st.info(
    "The final application will use the selected best-performing "
    "model for sentiment prediction."
)
