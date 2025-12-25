from typing import List, Tuple
from io import BytesIO
import re

# PDF text extraction
from pypdf import PdfReader

# Summarization (lexrank via sumy) + NLTK sentence tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords

# Lightweight retrieval for Q/A
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import nltk

def safe_sent_tokenize(text: str):
    """Try NLTK; if data missing, use a simple regex split."""
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]


def ensure_nltk_resources():
    # Attempt to download required NLTK data if missing
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    # Newer NLTK uses punkt_tab
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except:
            pass
    for pkg in ["stopwords", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg)
            except:
                pass

def extract_text_from_pdf(file) -> str:
    # `file` is a BytesIO from Streamlit uploader
    reader = PdfReader(file)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        texts.append(t)
    # Clean up excessive whitespace
    text = "\n".join(texts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def summarize_text(text: str, max_sentences: int = 7) -> str:
    """Summarize using LexRank (unsupervised, no GPU)."""
    # sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    sentences = [s.strip() for s in safe_sent_tokenize(text) if s.strip()]

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sents = summarizer(parser.document, sentences_count=max_sentences)
    return " ".join(str(s) for s in summary_sents)

def _top_n_nouns(text: str, n: int = 10) -> List[str]:
    tokens = word_tokenize(text)
    try:
        stops = set(stopwords.words("english"))
    except LookupError:
        stops = set()
    words = [w for w in tokens if w.isalpha() and w.lower() not in stops]
    try:
        tagged = pos_tag(words)
    except LookupError:
        # If tagger missing, fallback: treat all words as nouns
        tagged = [(w, "NN") for w in words]
    nouns = [w for w, t in tagged if t.startswith("NN")]
    # frequency
    freq = {}
    for w in nouns:
        wl = w.lower()
        freq[wl] = freq.get(wl, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:n]]

def generate_questions(text: str, n: int = 10) -> List[str]:
    """Very lightweight heuristic QG without heavy ML models."""
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    nouns = _top_n_nouns(text, n=20)

    questions = []
    # 1) Definition-style
    for kw in nouns[: max(1, n//2)]:
        questions.append(f"What is {kw}?")

    # 2) Factoid from 'is/are/was/were' sentences -> 'What ...?'
    copula_pat = re.compile(r"(.+?)\s+(is|are|was|were)\s+(.+?)\.", re.IGNORECASE)
    for s in sentences:
        m = copula_pat.search(s)
        if m and len(questions) < n:
            subj = re.sub(r"^[Tt]he\s+|^[Aa]n?\s+", "", m.group(1)).strip()
            if subj:
                questions.append(f"What {m.group(2).lower()} {subj}?")

    # 3) When/Why prompts for sentences with dates/causality
    when_pat = re.compile(r"\b(19|20)\d{2}\b")
    for s in sentences:
        if when_pat.search(s) and len(questions) < n:
            questions.append("When did that happen?")
    for s in sentences:
        if ("because" in s.lower() or "due to" in s.lower()) and len(questions) < n:
            questions.append("Why did that happen?")

    # Deduplicate and trim
    seen = set()
    out = []
    for q in questions:
        qq = q.strip()
        if qq and qq not in seen:
            seen.add(qq)
            out.append(qq)
        if len(out) >= n:
            break
    return out

class QAHelper:
    """Very simple retrieval-based QA over document sentences using TF-IDF."""
    def __init__(self, text: str):
        sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        self.sentences = sentences if sentences else [text]
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english").fit(self.sentences)
        self.matrix = self.vectorizer.transform(self.sentences)

    def answer(self, question: str) -> Tuple[str, str]:
        qv = self.vectorizer.transform([question])
        sims = cosine_similarity(qv, self.matrix)[0]
        if sims.max() < 0.05:
            return ("I'm not confident based on the document.", "")
        idx = int(sims.argmax())
        best = self.sentences[idx]
        # Provide a small window of context
        left = max(0, idx-2)
        right = min(len(self.sentences), idx+3)
        context = " ".join(self.sentences[left:right])
        return (best, context)
