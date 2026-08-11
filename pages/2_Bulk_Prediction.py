import streamlit as st
import pandas as pd

st.title("Bulk CSV Prediction")

st.write("Upload a CSV file containing product reviews.")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Read the CSV file
    try:
        df = pd.read_csv(uploaded_file)

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    except pd.errors.ParserError:
        st.error("The uploaded file is not a valid CSV.")
        st.stop()

    st.subheader("CSV Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Check required column
    if "Review Text" not in df.columns:

        st.error(
            "The CSV must contain a column named 'Review Text'."
        )

    else:

        # Remove completely empty reviews
        df["Review Text"] = df["Review Text"].fillna("")

        valid_reviews = df["Review Text"].astype(str).str.strip()

        empty_count = (valid_reviews == "").sum()

        st.success(
            f"{len(df)} reviews loaded successfully."
        )

        if empty_count > 0:
            st.warning(
                f"{empty_count} empty review(s) found."
            )

        else:
            st.info("No empty reviews found.")

        st.subheader("Dataset Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Reviews", len(df))

        with col2:
            st.metric(
                "Valid Reviews",
                len(df) - empty_count
            )

        if st.button("Predict All"):

            if empty_count == len(df):

                st.error(
                    "There are no valid reviews to predict."
                )

            else:

                # Temporary results until the final model is connected
                result_df = df.copy()

                result_df["Predicted Sentiment"] = (
                    "Pending Model"
                )

                st.subheader("Prediction Results")

                st.dataframe(
                    result_df,
                    use_container_width=True
             )

                csv_data = result_df.to_csv(index=False)

            st.download_button(
                    "Download Results CSV",
                    data=csv_data,
                    file_name="sentiment_predictions.csv",
                    mime="text/csv"
                )