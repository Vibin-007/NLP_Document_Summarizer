# NLP Document Summarizer

A Streamlit-based web application that serves as a Research Assistant Interface. It extracts, summarizes, and analyzes text from uploaded documents (PDF, DOCX, TXT) using Natural Language Processing (NLP).

## Features

* **Document Parsing:** Extracts text from `.pdf`, `.docx`, and `.txt` files.
* **Extractive Summarization:** Uses NLTK to generate adjustable text summaries based on word frequency.
* **Keyword Extraction:** Identifies key insights and phrases using YAKE.
* **Data Visualization:** Generates Word Clouds and Plotly relevance distribution charts.
* **Audio Synthesis:** Converts the generated summary into playable audio using Google Text-to-Speech (gTTS).
* **Export Options:** Download summaries as Plain Text (`.txt`) or Word Documents (`.docx`).
* **Custom UI:** Includes a dark/light theme toggle and responsive glass-card styling.

## Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install streamlit nltk pypdf python-docx yake wordcloud plotly matplotlib gtts
