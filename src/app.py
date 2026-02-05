"""
Streamlit app for Twitter Sentiment Analysis.

- Inference-only app
- Uses trained DL model + tokenizer + label encoder
- Supports single and batch predictions
"""

# -------------------------------------------------
# Make project root importable (IMPORTANT)
# -------------------------------------------------

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------
# Imports
# -------------------------------------------------

from typing import Any
import pandas as pd
import streamlit as st

from src.deployment import predict_single, predict_batch


# -------------------------------------------------
# UI helpers
# -------------------------------------------------

def build_text_input() -> str:
    st.subheader("✍️ Enter a Tweet / Text")

    text = st.text_area(
        label="Text",
        height=150,
        placeholder="Type a tweet or sentence here..."
    )

    return text


# -------------------------------------------------
# Main app
# -------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Twitter Sentiment Analysis",
        page_icon="💬",
        layout="centered",
    )

    st.title("💬 Twitter Sentiment Analysis")

    st.markdown(
        """
        This application predicts the **sentiment** of a given tweet or text.

        **Supported sentiments:**
        - Positive 🙂
        - Neutral 😐
        - Negative 🙁

        **Outputs:**
        - Predicted sentiment
        - Confidence score
        - Class-wise probabilities
        """
    )

    tab1, tab2 = st.tabs(["📝 Single Prediction", "📂 Batch Prediction"])

    # -------------------------------
    # Single Prediction
    # -------------------------------

    with tab1:
        text = build_text_input()

        if st.button("🔍 Analyze Sentiment"):
            if not text.strip():
                st.warning("Please enter some text.")
            else:
                try:
                    result = predict_single(text)

                    sentiment = result["predicted_sentiment"]
                    
                    if sentiment == "positive":
                        st.success(f"**Predicted Sentiment:** {sentiment.capitalize()}")
                    elif sentiment == "negative":
                        st.error(f"**Predicted Sentiment:** {sentiment.capitalize()}")
                    else:  # neutral
                        st.warning(f"**Predicted Sentiment:** {sentiment.capitalize()}")
                

                    st.metric(
                        label="Confidence",
                        value=f"{result['confidence'] * 100:.2f}%"
                    )

                    st.subheader("📊 Class Probabilities")
                    prob_df = (
                        pd.DataFrame.from_dict(
                            result["probabilities"],
                            orient="index",
                            columns=["Probability"]
                        )
                        .reset_index()
                        .rename(columns={"index": "Sentiment"})
                    )
                    prob_df["Probability"] *= 100

                    st.dataframe(prob_df, use_container_width=True)

                except Exception as exc:
                    st.error("Prediction failed.")
                    st.write(str(exc))

    # -------------------------------
    # Batch Prediction
    # -------------------------------

    with tab2:
        st.subheader(
            "Upload a CSV for batch sentiment prediction "
            "(must contain a text column)"
        )

        uploaded = st.file_uploader("Choose a CSV", type=["csv"])

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df.head())

            text_column = st.selectbox(
                "Select text column",
                options=df.columns.tolist()
            )

            if st.button("Run Batch Sentiment Analysis"):
                try:
                    result_df = predict_batch(df, text_column=text_column)

                    st.subheader("Predictions Preview")
                    st.dataframe(result_df.head(10))

                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions",
                        data=csv,
                        file_name="sentiment_predictions.csv",
                        mime="text/csv",
                    )

                except Exception as exc:
                    st.error("Batch prediction failed.")
                    st.write(str(exc))


# -------------------------------------------------
# Entry point
# -------------------------------------------------

if __name__ == "__main__":
    main()