# PDF Summarizer & Q/A Bot (Streamlit)

A lightweight, local Python app that lets you:
1. **Upload a text-based PDF**
2. **Summarize** the document (unsupervised LexRank – fast & no GPU required)
3. **Generate quiz questions** automatically (heuristics-based, no big ML downloads)
4. **Ask questions** like a chatbot using simple retrieval over the document

> **Note**: Works best for text-based PDFs. For scanned PDFs, run OCR first (e.g., Tesseract or Adobe).

---

## 🧰 1) Install

Create a virtual environment (recommended) and install dependencies:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If NLTK downloads its small tokenizers on first run, allow it (the app tries to auto-download if missing).

---

## ▶️ 2) Run the App

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

---

## 🗂 3) Project Structure

```
.
├─ app.py          # Streamlit UI
├─ utils.py        # Text extraction, summarization, QG, simple QA
├─ requirements.txt
└─ README.md
```

---

## 💡 Tips / Common Issues

- **Tkinter errors?** This app uses Streamlit instead, so you don't need Tkinter.
- **Torch/Transformers too heavy?** Not used here. The summarizer is LexRank (Sumy) and the QA is TF‑IDF based.
- **Scanned PDFs** need OCR first.
- If you see an NLTK resource error, run:
  ```python
  import nltk
  nltk.download("punkt"); nltk.download("stopwords"); nltk.download("averaged_perceptron_tagger")
  ```

---

## 🧪 Extending (Optional)

- Swap LexRank with modern summarizers using `transformers` (e.g., `facebook/bart-large-cnn`) if you have PyTorch installed.
- Replace heuristic QG with a T5-based question generator model.
- Add document chunking + embeddings (e.g., `sentence-transformers`) for better answers on long PDFs.
