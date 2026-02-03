
---

## `ARCHITECTURE.md`

```md
# Architecture

## 1. Overview
This system implements a lightweight Retrieval-Augmented pipeline for semantic search over a news corpus:

1) **Index-time pipeline (offline)**
- Load articles from JSON
- Clean and normalize text
- Embed each article using a SentenceTransformer model
- Save embeddings + metadata locally for fast search

2) **Query-time pipeline (online)**
- Embed user query
- Compute cosine similarity against the stored embeddings
- Return Top-K matching articles

3) **LLM enhancement (optional)**
- Summarize Top-K results OR
- Answer user questions using only retrieved context (grounded output)

---

## 2. Data Layer
### Input
- File: `data/news_articles_filtered.json`
- Schema fields typically include:
  - `id`, `title`, `source`, `date`, `content`, `url`, `category`

### Preprocessing
Implemented in `src/pipeline.py`:
- Removes HTML artifacts via BeautifulSoup
- Removes truncation markers like `[+9921 chars]`
- Collapses extra whitespace
- Creates embedding text: `title + ". " + cleaned_content`
- Truncates long article text to a safe maximum length for speed and stability

### Output Artifacts
Generated locally (not committed):
- `data/processed/news_embeddings.npy` (N x d embedding matrix)
- `data/processed/news_metadata.csv` (article metadata + cleaned content)

---

## 3. Embedding + Retrieval
### Embedding Model
- `sentence-transformers/all-MiniLM-L6-v2`

### Similarity Search
Implemented in `src/search.py`:
- Query is embedded and normalized
- Document embeddings are normalized at build time
- Cosine similarity computed using dot product:
  - `scores = embeddings @ query_embedding`
  - because normalized vectors → dot product ≈ cosine similarity

### Why this approach?
- Simple and fast for a few hundred documents
- Easy to understand and explain in a demo
- Can scale later with FAISS if needed

---

## 4. LLM Enhancement (Local HF Models)
Implemented in `src/llm_local.py`.

### Summarization
- Model: `facebook/bart-large-cnn`
- Input: context built from top-K retrieved documents
- Output: concise summary of retrieved results

### Q&A
- Model: `google/flan-t5-base`
- Prompt instructs the model:
  - answer using only the retrieved articles
  - return "Not enough information" if context is insufficient

### Grounding Strategy
- Only the retrieved snippets are passed into the model
- This reduces hallucinations and keeps outputs aligned with the dataset

### Device Handling
- Automatically chooses:
  - GPU (`cuda`) if available
  - else CPU (`cpu`) for portability

---

## 5. Interfaces / Deployment
### CLI (required)
- `interface/cli.py`
- Supports:
  - `--query` semantic search
  - `--summarize` summarize retrieved articles
  - `--ask` Q&A over retrieved articles
  - `--top_k` number of results

### Streamlit 
- `interface/app.py`
- Provides interactive search + expanders + optional summarization/Q&A 

---

## 6. Code Quality & Production Readiness
- Modular structure:
  - `pipeline.py` = index build
  - `search.py` = retrieval
  - `llm_local.py` = generation
  - `cli.py/app.py` = interfaces
- Cached embeddings avoid recomputation
- Safe file paths with `pathlib`
- Clear commands for reproducibility

---

## 7. Possible Extensions
- Replace in-memory similarity with FAISS
- Add citation/highlighting in UI
- Add evaluation metrics (precision@k, qualitative checks)
- Add scheduling to refresh the corpus and rebuild embeddings
