# Semantic News Search (Embeddings + Local LLM)

A production-ready semantic search application for a news corpus. It retrieves relevant articles using dense embeddings + cosine similarity, and enhances results with a local LLM for summarization or question answering.

## What this project does
- Loads a dataset of news articles from JSON
- Cleans and prepares text
- Generates **SentenceTransformer** embeddings for each article
- Performs **cosine similarity search** to return Top-K results
- Adds an **LLM enhancement**:
  - Summarize Top-K results
  - Answer questions using retrieved context (RAG-style)

## Tech Stack
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Search: cosine similarity (normalized embeddings + dot product)
- Local LLM:
  - Summarization: `facebook/bart-large-cnn`
  - Q&A: `google/flan-t5-base`
- Interfaces:
  - CLI (deployment requirement)
  - Streamlit (bonus/demo)

## Project Structure
```text
news-semantic-search/
  data/
    news_articles_filtered.json
    processed/                # generated locally (ignored by git)
      news_embeddings.npy
      news_metadata.csv
  src/
    pipeline.py               # preprocess + embeddings build
    search.py                 # semantic search
    llm_local.py              # summarization + Q&A (local HF models)
  interface/
    cli.py                    # command-line interface
    app.py                    # streamlit UI
  requirements.txt
  README.md
  ARCHITECTURE.md
  TEAM_CONTRIBUTIONS.md
  .gitignore
```

## Setup

### 1. Create & activate virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1

```
### 2. Install dependencies
``` bash
pip install -r requirements.txt

```
### 3. Build embeddings (run once)
``` bash
python .\src\pipeline.py

```
This creates:

data/processed/news_embeddings.npy

data/processed/news_metadata.csv

## Run: Streamlit UI
```bash
streamlit run .\interface\app.py
```
