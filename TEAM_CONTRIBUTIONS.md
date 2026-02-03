# Team Contribution

## 1. Tamarakare Edwin-Biayeibo
- Collected 292 articles in a json file
- Input dataset: `data/news_articles_filtered.json`
- Each record contains: `id`, `title`, `source`, `date`, `content`, `url`, `category`

## 2. Gema Zhu
-Implemented in `src/pipeline.py`:
- Clean HTML artifacts, normalize whitespace
- Remove truncation markers like `[+9921 chars]`
- Create embedding text: `title + ". " + cleaned_content`
- Truncate long text to a safe length (prevents extremely long sequences)

Outputs:
- `data/processed/news_embeddings.npy`
- `data/processed/news_metadata.csv`

### Embeddings & Index
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- We generate embeddings for each document and use `normalize_embeddings=True`

Encode user query → normalized embedding
- Similarity scores computed using dot product:
  - Since embeddings are normalized, dot product ≈ cosine similarity
- Return Top-K results with scores and metadata


## 3. Ruofan (Kate) Yang 

LLM Enhancement (Local)
Implemented in `src/llm_local.py`.
Summarization
- Model: `facebook/bart-large-cnn`
- Input: concatenated snippets from Top-K results
- Output: short summary

### Q&A
- Model: `google/flan-t5-base`
- Prompt: instructs the model to answer ONLY from the provided retrieved context
- If information is missing, it responds with “Not enough information.”

### CPU/GPU Handling
- Automatically selects device:
  - `cuda` if available
  - otherwise `cpu`
This improves portability across student laptops and demo environments.


## 4. Vishal Srivastava
### Interfaces
### CLI (required deployment)
`interface/cli.py` supports:
- `--query` for semantic search
- `--summarize` for summarization
- `--ask` for Q&A

### Streamlit 
`interface/app.py` provides:
- Search input
- Top-K results display
- Summarize/Q&A mode for demo
  
## 5. Qihua (Kiara) Liu
### Error Handling & Production Readiness
- Deterministic file paths via `pathlib`
- Embeddings cached to disk; avoids recomputation on each run
- Clear separation of concerns:
  - `pipeline.py` (index build)
  - `search.py` (retrieval)
  - `llm_local.py` (generation)
  - `cli.py` / `app.py` (interfaces)
