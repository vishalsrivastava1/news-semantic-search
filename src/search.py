from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

EMB_PATH = PROCESSED_DIR / "news_embeddings.npy"
META_PATH = PROCESSED_DIR / "news_metadata.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
embeddings = np.load(EMB_PATH)
df = pd.read_csv(META_PATH)


def search(query: str, top_k: int = 5) -> pd.DataFrame:
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    # dot product ≈ cosine because embeddings are normalized
    scores = embeddings @ q_emb
    top_idx = np.argsort(-scores)[:top_k]

    cols = ["id", "title", "source", "date", "url", "category", "content_clean"]
    out = df.iloc[top_idx][cols].copy()
    out["score"] = scores[top_idx]
    return out.sort_values("score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print(search("AI regulation in the US", top_k=5)[["title", "source", "score"]])
