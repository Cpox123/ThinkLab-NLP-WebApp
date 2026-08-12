import streamlit as st

from config.project_data import (
    MAX_REVIEW_CHARS,
    MODEL_INFO,
)
from services import ui_service as ui
from services.prediction_service import predict_with_probabilities

ui.apply_theme()
ui.render_brand()

ui.render_page_header(
    "Single Review Prediction",
    "Enter a product review below to predict its sentiment using our "
    "BERT model.",
)

RESULT_KEY = "single_result"
INPUT_KEY = "single_review_input"
INPUT_VERSION_KEY = "single_input_version"

# Text widgets only reset when their key changes, so we version the key
st.session_state.setdefault(INPUT_VERSION_KEY, 0)

input_key = f"{INPUT_KEY}_{st.session_state[INPUT_VERSION_KEY]}"

left, right = st.columns([1.05, 0.95], gap="large")

# --------------------------------------------------
# Left: input card
# --------------------------------------------------

with left:

    with st.container(border=True):

        st.markdown(
            f'<div style="font-size:1.05rem; font-weight:700; '
            f'color:{ui.TEXT_DARK}; margin-bottom:6px;">✍️ Enter Review</div>',
            unsafe_allow_html=True,
        )

        review = st.text_area(
            "Review",
            placeholder="This dress fits perfectly and the material is "
                        "amazing. I really love the quality and the color is "
                        "exactly what I expected. Totally worth the price!",
            height=190,
            label_visibility="collapsed",
            key=input_key,
        )

        st.caption(f"{len(review)} / {MAX_REVIEW_CHARS}")

        predict = st.button(
            "✨ Predict Sentiment",
            type="primary",
            width="stretch",
        )

        # Handle the prediction immediately so the result and the Clear
        # button below both reflect this same run.
        if predict:

            if not review.strip():
                st.warning("Please enter a review.")

            else:
                review_text = review.strip()[:MAX_REVIEW_CHARS]

                with st.spinner("Analyzing review..."):
                    prediction = predict_with_probabilities(review_text)

                if prediction is None:
                    st.warning("Please enter a valid review.")
                    st.session_state.pop(RESULT_KEY, None)

                else:
                    label, probabilities = prediction

                    st.session_state[RESULT_KEY] = {
                        "label": label,
                        "confidence": probabilities[label],
                        "probabilities": probabilities,
                    }

        clear_clicked = False

        if st.session_state.get(RESULT_KEY) is not None:
            clear_clicked = st.button(
                "🧹 Clear",
                width="stretch",
            )

    if clear_clicked:
        st.session_state.pop(RESULT_KEY, None)
        # Bumping the version remounts the text area empty
        st.session_state[INPUT_VERSION_KEY] += 1
        st.rerun()

# --------------------------------------------------
# Right: result panels
# --------------------------------------------------

with right:

    result = st.session_state.get(RESULT_KEY)

    st.markdown(
        ui.result_card_html(result),
        unsafe_allow_html=True,
    )

    st.markdown(
        ui.prob_panel_html(result),
        unsafe_allow_html=True,
    )

    st.markdown(
        ui.model_info_html(MODEL_INFO),
        unsafe_allow_html=True,
    )

ui.render_footer()
