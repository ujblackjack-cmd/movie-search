"""질문 -> 파싱 -> 하이브리드 검색 -> 근거 기반 추천 답변.

    export ANTHROPIC_API_KEY=...
    python ask.py "2010년대 평점 높은 반전 심리 스릴러 추천해줘"
    python ask.py "인셉션 같은데 덜 유명한 영화" --k 5 --mode vector
"""
import argparse

import anthropic

import config as C
from search import HybridSearcher, parse_query

ANSWER_SYSTEM = """You are a movie recommendation assistant.
Recommend ONLY movies from the provided search results. Never mention a movie that is not in them.
For each recommendation give the title (year) and 1-2 sentences on why it fits the request,
grounded in the overview/genres/director shown. If the results do not fit well, say so plainly.
Answer in the same language as the user's question."""


def format_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        rating = f"{r['rating']:.1f}" if r["rating"] else "n/a"
        cast = ", ".join(str(r["cast"] or "").split(", ")[:4])
        blocks.append(
            f"[{i}] {r['title']} ({r['year']}) | rating {rating} | {r['genres']}\n"
            f"Director: {r['director']} | Cast: {cast}\n"
            f"Overview: {str(r['overview'])[:600]}"
        )
    return "\n\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    args = ap.parse_args()

    client = anthropic.Anthropic()
    query, filters = parse_query(args.question, client)
    print(f"[검색어] {query}\n[필터] {filters}\n")

    searcher = HybridSearcher()
    results = searcher.search(query, filters, k=args.k, mode=args.mode)
    if not results:
        print("조건에 맞는 영화를 찾지 못했어요. 필터(연도, 평점, 장르)를 완화해 보세요.")
        return

    for r in results:
        print(f"- {r['title']} ({r['year']})  vec#{r['vec_rank']} bm25#{r['bm25_rank']}")
    print()

    resp = client.messages.create(
        model=C.LLM_MODEL, max_tokens=900, system=ANSWER_SYSTEM,
        messages=[{"role": "user",
                   "content": f"Question: {args.question}\n\nSearch results:\n{format_context(results)}"}],
    )
    print(resp.content[0].text)


if __name__ == "__main__":
    main()
