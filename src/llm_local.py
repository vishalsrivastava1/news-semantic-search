import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Matches notebook models
SUM_MODEL = "facebook/bart-large-cnn"
QA_MODEL = "google/flan-t5-base"

_sum_tok = None
_sum_mdl = None
_qa_tok = None
_qa_mdl = None


def _load_summarizer():
    global _sum_tok, _sum_mdl
    if _sum_mdl is None:
        _sum_tok = AutoTokenizer.from_pretrained(SUM_MODEL)
        _sum_mdl = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL).to(DEVICE)
    return _sum_tok, _sum_mdl


def _load_qa():
    global _qa_tok, _qa_mdl
    if _qa_mdl is None:
        _qa_tok = AutoTokenizer.from_pretrained(QA_MODEL)
        _qa_mdl = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL).to(DEVICE)
    return _qa_tok, _qa_mdl


def build_context(docs_df, per_doc_chars=800, max_chars=4500) -> str:
    parts = []
    for i, row in docs_df.iterrows():
        parts.append(
            f"(Doc {i+1}) {row['title']}\n"
            f"Source: {row['source']} | Date: {row['date']}\n"
            f"{str(row['content_clean'])[:per_doc_chars]}"
        )
    ctx = "\n\n".join(parts)
    return ctx[:max_chars]


def summarize_with_bart(text: str, max_new_tokens: int = 180) -> str:
    tok, mdl = _load_summarizer()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)


def answer_with_flan(question: str, context: str, max_new_tokens: int = 220) -> str:
    tok, mdl = _load_qa()
    prompt = (
        "Answer the question using ONLY the articles below. "
        'If the answer is not present, say "Not enough information."\n\n'
        f"Question: {question}\n\nArticles:\n{context}"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)
