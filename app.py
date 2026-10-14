import os

import streamlit as st
from transformers import pipeline

NAME = "Ibrahim Hasan Zubairi"
STUDENT_ID = "B040050iz"  # <-- put your full Student ID here
MODEL_ID = "google-t5/t5-small"
# use the local copy if present (slow connection), otherwise download from the Hub (e.g. on HF Spaces)
MODEL = "models/t5-small" if os.path.isdir("models/t5-small") else MODEL_ID

st.set_page_config(page_title=f"Summarizer | {NAME} | {STUDENT_ID}")
st.title(f"AI Text Summarizer — {NAME} ({STUDENT_ID})")
st.caption(f"CS4106 Quiz 01 · pipeline: summarization · model: {MODEL_ID}")


@st.cache_resource
def load_model():
    return pipeline("summarization", model=MODEL)


summarizer = load_model()

text = st.text_area("Paste the text to summarize:", height=220)
max_len = st.slider("Maximum summary length (tokens)", min_value=20, max_value=200, value=60, step=10)

if st.button("Summarize"):
    in_words = len(text.split())
    if not text.strip():
        st.warning("Please enter some text first.")
    elif in_words < 20:
        st.warning("Text is too short to summarize (need at least 20 words).")
    else:
        with st.spinner("Summarizing..."):
            result = summarizer(
                text,
                max_new_tokens=max_len,  # max_length is overridden by the model's generation config
                min_new_tokens=min(10, max_len - 5),
                do_sample=False,
                truncation=True,  # long inputs are cut to the model's 512-token limit
            )
        summary = result[0]["summary_text"].strip()
        out_words = len(summary.split())

        st.subheader("Summary")
        st.write(summary)

        # Personal feature (d = 0): compression ratio = output words / input words
        ratio = out_words / in_words
        c1, c2, c3 = st.columns(3)
        c1.metric("Input words", in_words)
        c2.metric("Summary words", out_words)
        c3.metric("Compression ratio", f"{ratio:.2f}", f"{(1 - ratio) * 100:.0f}% shorter")
