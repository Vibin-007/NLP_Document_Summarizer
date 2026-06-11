import streamlit as st
import nltk
import string

from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from pypdf import PdfReader
from docx import Document

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(
    page_title="NLP Document Summarizer",
    page_icon="📄",
    layout="wide"
)

st.title("NLP Document Summarizer")

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "txt"]
)

summary_percent = st.slider(
    "Summary Percentage",
    10,
    90,
    50
)

def extract_text(uploaded_file):

    if uploaded_file.type == "application/pdf":

        pdf = PdfReader(uploaded_file)

        text = ""

        for page in pdf.pages:
            text += page.extract_text()

        return text

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":

        doc = Document(uploaded_file)

        text = ""

        for para in doc.paragraphs:
            text += para.text + "\n"

        return text

    else:

        return str(
            uploaded_file.read(),
            "utf-8"
        )

if uploaded_file:

    text = extract_text(uploaded_file)

    st.subheader("Document Statistics")

    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Words",
        len(words)
    )

    col2.metric(
        "Sentences",
        len(sentences)
    )

    col3.metric(
        "Avg Sentence Length",
        round(len(words)/len(sentences),2)
    )

    stop_words = set(
        stopwords.words("english")
    )

    clean_words = []

    for word in words:

        word = word.lower()

        if word.isalpha():

            if word not in stop_words:

                clean_words.append(word)

    frequency = Counter(clean_words)

    max_frequency = max(
        frequency.values()
    )

    for word in frequency:

        frequency[word] /= max_frequency

    sentence_scores = {}

    for sentence in sentences:

        for word in word_tokenize(
            sentence.lower()
        ):

            if word in frequency:

                sentence_scores[sentence] = (
                    sentence_scores.get(
                        sentence,
                        0
                    )
                    + frequency[word]
                )

    ranked_sentences = sorted(
        sentence_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    summary_length = int(
        len(sentences)
        * summary_percent
        / 100
    )

    top_sentences = ranked_sentences[
        :summary_length
    ]

    selected = [
        sentence
        for sentence, score
        in top_sentences
    ]

    summary = " ".join(
        [
            sentence
            for sentence in sentences
            if sentence in selected
        ]
    )

    st.subheader("Generated Summary")

    st.write(summary)

    original_words = len(
        word_tokenize(text)
    )

    summary_words = len(
        word_tokenize(summary)
    )

    compression_ratio = (
        summary_words
        /
        original_words
    ) * 100

    st.subheader("Evaluation")

    st.metric(
        "Compression Ratio",
        f"{compression_ratio:.2f}%"
    )

    st.download_button(
        "Download Summary",
        summary,
        file_name="summary.txt"
    )