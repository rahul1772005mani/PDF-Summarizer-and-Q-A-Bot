import streamlit as st
from io import BytesIO
from utils import (
    extract_text_from_pdf,
    summarize_text,
    generate_questions,
    QAHelper,
    ensure_nltk_resources
)
# at the top of app.py
import re

def squish(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

st.set_page_config(page_title="PDF Summarizer & Q/A Bot", layout="wide")
ensure_nltk_resources()

st.title("📄➡️🧠 PDF Summarizer & Q/A Bot")
st.caption("Upload a *text-based* PDF, get a clean summary, auto-generated quiz questions, and ask questions like a chatbot.")

# Session state
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "qa" not in st.session_state:
    st.session_state.qa = None
if "questions" not in st.session_state:
    st.session_state.questions = []

with st.sidebar:
    st.header("Steps")
    st.markdown("1) **Upload PDF**\n2) **Summarize**\n3) **Generate Questions**\n4) **Chat/Q&A**")
    st.divider()
    st.markdown("**Tips**\n- Use text-based PDFs (not scanned images)\n- For scanned PDFs, run OCR first (e.g., Adobe, Tesseract)")
    st.divider()
    st.markdown("Made with ❤️ in Streamlit.")

st.subheader("1) Upload PDF")
uploaded = st.file_uploader("Choose a text-based PDF", type=["pdf"])

col1, col2 = st.columns([1,1])
with col1:
    summarize_btn = st.button("🔎 Extract & Summarize")
with col2:
    gen_q_btn = st.button("📝 Generate Questions (from summary if available, else full text)")

if summarize_btn:
    if not uploaded:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Extracting text from PDF..."):
            st.session_state.full_text = extract_text_from_pdf(uploaded)
        if not st.session_state.full_text.strip():
            st.error("Couldn't extract any text. Make sure the PDF is text-based or run OCR.")
        else:
            with st.spinner("Summarizing..."):
                st.session_state.summary = summarize_text(st.session_state.full_text, max_sentences=7)
            st.success("Summary generated! See below.")
            st.session_state.qa = QAHelper(st.session_state.full_text)

if gen_q_btn:
    if st.session_state.summary.strip():
        base_text = st.session_state.summary
    elif st.session_state.full_text.strip():
        base_text = st.session_state.full_text
    else:
        base_text = ""
    if not base_text:
        st.warning("Upload and/or summarize first.")
    else:
        with st.spinner("Generating questions..."):
            st.session_state.questions = generate_questions(base_text, n=10)
        st.success("Questions generated! See the 'Quiz Yourself' section.")

if st.session_state.summary:
    st.subheader("2) Summary")
    st.text_area("Auto Summary", value=st.session_state.summary, height=250)

if st.session_state.questions:
    st.subheader("3) Quiz Yourself")
    for i, q in enumerate(st.session_state.questions, 1):
        st.markdown(f"**Q{i}. {q}**")

st.subheader("4) Ask the Document (Chat/Q&A)")
user_q = st.text_input("Type your question about the PDF")
if st.button("Ask") and user_q.strip():
    if st.session_state.qa is None:
        if not st.session_state.full_text.strip():
            st.warning("Upload a PDF and extract text first.")
        else:
            st.session_state.qa = QAHelper(st.session_state.full_text)
    if st.session_state.qa:
        with st.spinner("Thinking..."):
            answer, context = st.session_state.qa.answer(user_q)
        st.markdown("**Answer**")
        st.write(squish(answer) if answer else "Sorry, I couldn't find a good answer in the document.")
        with st.expander("Show supporting context"):
            st.write(squish(context))

