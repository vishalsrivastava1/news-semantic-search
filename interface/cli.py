import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.search import search
from src.llm_local import build_context, summarize_with_bart, answer_with_flan



def main():
    parser = argparse.ArgumentParser(description="Semantic News Search (CLI)")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--summarize", action="store_true", help="Summarize top-k results")
    parser.add_argument("--ask", type=str, help="Ask a question based on retrieved docs")
    args = parser.parse_args()

    if not args.query and not args.ask:
        print("❌ Provide --query or --ask")
        return

    user_text = args.ask if args.ask else args.query
    results = search(user_text, top_k=args.top_k)

    print("\nTop results:")
    for i, row in results.iterrows():
        print(f"{i+1}. {row['title']} ({row['source']} | {row['date']})  score={row['score']:.3f}")

    if args.summarize:
        ctx = build_context(results)
        print("\n--- Summary ---")
        print(summarize_with_bart(ctx))

    if args.ask:
        ctx = build_context(results)
        print("\n--- Answer ---")
        print(answer_with_flan(args.ask, ctx))


if __name__ == "__main__":
    main()
