import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.search import search
from src.llm_local import build_context, summarize_with_bart, answer_with_flan

st.set_page_config(page_title="Semantic News Search", layout="wide")

st.title("📰 Semantic News Search (Embeddings + Local LLM)")
st.caption("Search news articles using embeddings. Summarize or ask questions using local Hugging Face models.")

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top K", 1, 10, 5)
mode = st.sidebar.radio("Mode", ["Search only", "Summarize top-K", "Q&A over top-K"])

query = st.text_input("Search query", "AI regulation in the US")

if st.button("Search"):
    results = search(query, top_k=top_k)

    st.subheader("Results")
    for i, row in results.iterrows():
        with st.expander(f"{i+1}. {row['title']}  |  score={row['score']:.3f}", expanded=(i == 0)):
            st.write(f"**Source:** {row['source']}")
            st.write(f"**Date:** {row['date']}")
            st.write(f"**URL:** {row['url']}")
            st.write("---")
            st.write(str(row["content_clean"])[:1200] + ("..." if len(str(row["content_clean"])) > 1200 else ""))

    if mode == "Summarize top-K":
        st.subheader("🧾 Summary")
        with st.spinner("Summarizing..."):
            ctx = build_context(results)
            st.write(summarize_with_bart(ctx))

    if mode == "Q&A over top-K":
        st.subheader("❓ Ask a question")
        question = st.text_input("Question", "What is the main theme across these results?")
        if st.button("Answer"):
            with st.spinner("Answering..."):
                ctx = build_context(results)
                st.write(answer_with_flan(question, ctx))
