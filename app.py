
import streamlit as st

from config.project_data import PROJECT_TITLE, DATASET_NAME
from services.prediction_service import predict_sentiment


st.set_page_config(
    page_title="ThinkLab Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
)

st.title("ThinkLab NLP Sentiment Analyzer")

st.write("Product Review Sentiment Classification System")

st.subheader("Project Overview")

st.write(
    f"This application classifies product reviews into "
    f"Negative, Neutral, and Positive sentiments."
)

st.write(f"Dataset: {DATASET_NAME}")


st.subheader("Sentiment Prediction")

review = st.text_area(
    "Enter a product review:",
    placeholder="Example: The dress is beautiful and I really love it.",
    height=150,
)

if st.button("Predict Sentiment"):

    if not review.strip():
        st.warning("Please enter a review.")

    else:
        with st.spinner("Analyzing review..."):
            prediction = predict_sentiment(review)

        if prediction is None:
            st.warning("Please enter a valid review.")

        else:
            st.success(f"Prediction: {prediction}")

