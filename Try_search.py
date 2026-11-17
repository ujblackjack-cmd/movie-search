"""API 키 없이 검색기만 확인: python try_search.py"""
from search import HybridSearcher, Filters

s = HybridSearcher()

TESTS = [
    ("1) 의미 검색", "a man loses his memory and tries to find his wife's killer", Filters(), "hybrid"),
    ("2) 이름 검색(BM25 강점)", "Christopher Nolan", Filters(), "hybrid"),
    ("3) 필터 + 의미", "twist ending psychological", Filters(year_min=2010, year_max=2019, min_rating=7.0, genres=["Thriller"]), "hybrid"),
    # ("4) 한국어 직접 질의(bge-m3)", "우주에서 살아남으려는 사람들 이야기", Filters(), "vector"),
    ("4b) 영어 동일 질의", "people trying to survive in space", Filters(), "vector"),
]

for title, query, f, mode in TESTS:
    print(f"\n=== {title} | '{query}' | {mode}")
    for r in s.search(query, f, k=5, mode=mode):
        rating = f"{r['rating']:.1f}" if r["rating"] else "n/a"
        print(f"  {r['title']} ({r['year']})  ★{rating}  vec#{r['vec_rank']} bm25#{r['bm25_rank']}  {r['genres']}")