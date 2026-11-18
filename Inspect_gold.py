"""정답이 왜 안 잡히는지 확인 (오류 분석).

    python inspect_gold.py ko-03 --category ko_en
    python inspect_gold.py fil-02

정답 영화의 줄거리와, 각 검색기에서의 순위(상위 200위까지)를 보여줍니다.
줄거리가 질문과 아예 다른 표현이면 '검색 실패'가 아니라 '질문-라벨 불일치'입니다.
"""
import argparse
import sys

import config as C
from evaluate import build_title_index, load_queries, relevant_ids
from search import HybridSearcher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("qid")
ap.add_argument("--category", default=None, help="ko 질문: ko_direct 또는 ko_en")
ap.add_argument("--queries", default="eval/queries.json")
args = ap.parse_args()

qs = [q for q in load_queries(args.queries)
      if q["id"] == args.qid and (args.category is None or q["category"] == args.category)]
if not qs:
    sys.exit(f"질문 id 를 찾지 못함: {args.qid}")
q = qs[0]

s = HybridSearcher()
tidx, vc = build_title_index(s.df)
f, rel, notes, _ = relevant_ids(q, s, tidx, vc)
vec, bm = s.retrieve(q["text"], f, pool=200)
hyb = [i for i, _ in s.fuse(vec, bm)]


def rank(lst, i):
    return lst.index(i) + 1 if i in lst else ">200"


print(f"[{q['id']} / {q['category']}] {q['text']}\n필터: {f}\n정답 수: {len(rel)}  {notes}\n")
by = s.by_id
for i in list(rel)[:5]:
    r = by.loc[i]
    print(f"- 정답 {r['title']} ({r['year']}) ★{r['rating']}  votes={int(r['vote_count'])}")
    print(f"  순위  vector={rank(vec, i)}  bm25={rank(bm, i)}  hybrid={rank(hyb, i)}")
    print(f"  줄거리: {str(r['overview'])[:300]}\n")
print("[hybrid 상위 3]")
for i in hyb[:3]:
    r = by.loc[i]
    print(f"- {r['title']} ({r['year']}): {str(r['overview'])[:160]}")