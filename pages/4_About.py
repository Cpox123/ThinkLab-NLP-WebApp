import streamlit as st

st.title("About the Project")

st.subheader("Project Objective")

st.write(
    "This project develops an NLP-based system for classifying "
    "product reviews into Negative, Neutral, and Positive sentiments."
)

st.subheader("Dataset")

st.write(
    "The project uses the Women's E-Commerce Clothing Reviews dataset."
)

st.subheader("Models")

st.write(
    "Six models were developed and evaluated: Logistic Regression, "
    "LSTM, SVM, BERT, Naive Bayes, and CNN."
)

st.subheader("Ethics and Bias")

st.write(
    "The dataset contains customer reviews and may contain class "
    "imbalance. This can affect the model's ability to identify "
    "minority sentiment classes, especially Neutral reviews."
)

st.write(
    "Model predictions should be treated as automated classifications "
    "and should not be considered perfect representations of customer opinions."
)

st.subheader("Limitations")

st.write(
    "The system is trained on a specific product review dataset, "
    "so performance may differ when it is used on reviews from other "
    "products, platforms, or writing styles."
)
