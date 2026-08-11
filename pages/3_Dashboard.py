import streamlit as st
import pandas as pd
from config.project_data import MODEL_RESULTS, FINAL_MODEL_NAME

st.title("Model Comparison Dashboard")

st.write("Performance comparison of all six sentiment classification models.")

rows = []

for model, results in MODEL_RESULTS.items():
    rows.append({
        "Model": model,
        "Accuracy": results["accuracy"],
        "Macro F1": results["macro_f1"]
    })

df = pd.DataFrame(rows)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

st.subheader("Best Performing Model")

st.success(
    f"Selected final model: {FINAL_MODEL_NAME}"
)

st.write(
    "Macro F1-score is used as the main comparison metric "
    "because the sentiment classes are imbalanced."
)
