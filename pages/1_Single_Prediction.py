import streamlit as st

from services.bulk_service import predict_bulk
from services import ui_service as ui


# Apply shared app styling
ui.apply_theme()
ui.render_brand()


# Page header
ui.render_page_header(
    "Single Review Prediction",
    "Enter a product review to classify it as Positive, Neutral, or Negative.",
)


# Main layout
input_col, result_col = st.columns(
    [1.05, 1.0],
    gap="large",
)


# --------------------------------------------------
# Review input
# --------------------------------------------------

with input_col:

    st.markdown("### ✍️ Enter Review")

    review = st.text_area(
        "Product Review",
        placeholder="Example: The dress is beautiful and comfortable.",
        height=180,
        label_visibility="collapsed",
    )

    if st.button(
        "🔍 Analyze Review",
        type="primary",
        width="stretch",
    ):

        if not review.strip():

            st.warning(
                "Please enter a review."
            )

        else:

            prediction = predict_bulk([review])[0]

            st.session_state["single_result"] = {
                "label": prediction["prediction"],
                "confidence": prediction["confidence"],
            }

    if st.button(
        "🗑️ Clear Result",
        width="stretch",
    ):
        st.session_state.pop(
            "single_result",
            None,
        )
        st.rerun()


# --------------------------------------------------
# Prediction result
# --------------------------------------------------

with result_col:

    result = st.session_state.get(
        "single_result"
    )

    if isinstance(result, dict):

        st.markdown(
            ui.result_card_html(result),
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            ui.result_card_html(None),
            unsafe_allow_html=True,
        )


    # Model information
    model_info = {
        "Model": "BERT (base-uncased)",
        "Type": "Transformer (DL)",
        "Accuracy": "81.82%",
        "Macro F1": "0.6600",
        "Max Sequence Length": "128",
    }

    st.markdown(
        ui.model_info_html(model_info),
        unsafe_allow_html=True,
    )


# Footer
ui.render_footer()
