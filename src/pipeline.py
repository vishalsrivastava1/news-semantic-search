import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_JSON = DATA_DIR / "news_articles_filtered.json"
EMB_PATH = PROCESSED_DIR / "news_embeddings.npy"
META_PATH = PROCESSED_DIR / "news_metadata.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS = 2000


def clean_text(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = BeautifulSoup(x, "html.parser").get_text(" ")
    x = re.sub(r"\[\+\d+\s*chars\]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def load_articles():
    with open(RAW_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def build_df(articles):
    df = pd.DataFrame(articles)
    df["title_clean"] = df["title"].apply(clean_text)
    df["content_clean"] = df["content"].apply(clean_text)
    df["text"] = (df["title_clean"] + ". " + df["content_clean"]).str.strip()
    df["text"] = df["text"].str.slice(0, MAX_CHARS)
    return df


def build_or_load_embeddings(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if EMB_PATH.exists():
        return np.load(EMB_PATH)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # matches notebook
    )
    np.save(EMB_PATH, embeddings)
    return embeddings


def save_metadata(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["id", "title", "source", "date", "url", "category", "content_clean", "text"]
    df[keep].to_csv(META_PATH, index=False)


def main():
    articles = load_articles()
    df = build_df(articles)
    _ = build_or_load_embeddings(df)
    save_metadata(df)
    print("✅ Done. Saved:")
    print(" -", EMB_PATH)
    print(" -", META_PATH)


if __name__ == "__main__":
    main()
