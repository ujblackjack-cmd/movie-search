"""검색 방식 비교 평가 (LLM/API 키 불필요).

    python evaluate.py
    python evaluate.py --k 10 --misses hybrid
    python evaluate.py --methods vector,hybrid,hybrid_q

질문은 eval/queries.json 에서 읽습니다. 정답은 두 종류입니다.
  - gold     : 사람이 지정한 [제목, 연도]. 코퍼스에서 (제목, 연도±1)로 찾아 id 로 바꿉니다.
  - auto_rel : 감독/배우/장르 조건으로 코퍼스에서 자동 계산 (이름 검색용)
코퍼스에 없거나 질문의 필터를 위반하는 정답은 제외하고, 그 목록을 맨 처음에 보여줍니다.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from common import clean
from search import Filters, HybridSearcher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 비교할 방식. 필요하면 여기에서 추가/수정하세요.
METHODS = {
    "vector":      dict(mode="vector"),
    "bm25":        dict(mode="bm25"),
    "hybrid":      dict(mode="hybrid"),                          # 현재 방식(동일 가중 RRF)
    "hybrid_w":    dict(mode="hybrid", w_bm25=0.5),              # BM25 비중 절반
    "hybrid_q15":  dict(mode="hybrid", quality=0.15),            # 평점/투표수 가산 (약)
    "hybrid_q":    dict(mode="hybrid", quality=0.3),             # 평점/투표수 가산 (중)
    "hybrid_q50":  dict(mode="hybrid", quality=0.5),             # 평점/투표수 가산 (강)
    "hybrid_w_q":  dict(mode="hybrid", w_bm25=0.5, quality=0.3),
}
METRICS = ["hit", "recall", "mrr", "ndcg"]


# ---------------------------------------------------------------- 정답 만들기
def norm(t: str) -> str:
    return re.sub(r"[^\w]+", " ", str(t).lower()).strip()


def build_title_index(df: pd.DataFrame):
    idx: dict[str, list[tuple[int, int]]] = {}
    for i, t, y in zip(df["id"], df["title"], df["year"]):
        idx.setdefault(norm(t), []).append((int(y), int(i)))
    vc = dict(zip(df["id"].astype(int), df["vote_count"]))
    return idx, vc


def resolve(title: str, year: int, idx, vc) -> int | None:
    cands = [(y, i) for y, i in idx.get(norm(title), []) if abs(y - year) <= 1]
    if not cands:
        return None
    # 연도가 정확히 맞는 것 우선, 그다음 투표수 많은 것
    cands.sort(key=lambda c: (abs(c[0] - year), -vc.get(c[1], 0)))
    return cands[0][1]


def auto_relevant(df: pd.DataFrame, genre_sets, spec: dict) -> set[int]:
    m = np.ones(len(df), dtype=bool)
    if "director" in spec:
        key = spec["director"].lower()
        m &= df["director"].map(lambda x: key in clean(x).lower()).to_numpy()
    if "cast" in spec:
        key = spec["cast"].lower()

        def in_top(x):  # 문서에 들어간 상위 TOP_CAST 명만 기준 (검색이 볼 수 있는 범위)
            names = [n.strip().lower() for n in clean(x).split(",")][: C.TOP_CAST]
            return any(key in n for n in names)

        m &= df["cast"].map(in_top).to_numpy()
    if "genre" in spec:
        m &= np.array([spec["genre"] in g for g in genre_sets])
    return set(df.loc[m, "id"].astype(int))


def load_queries(path: str) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for q in raw:
        if q["category"] == "ko":  # 한국어 직접 vs (번역된) 영어, 같은 정답으로 두 번 평가
            out.append({**q, "category": "ko_direct", "text": q["query"]})
            out.append({**q, "category": "ko_en", "text": q["query_en"]})
        else:
            out.append({**q, "text": q["query"]})
    return out


def relevant_ids(q, s, tidx, vc):
    f = Filters(**q.get("filters", {}))
    allowed = set(s.df.loc[s.build_mask(f), "id"].astype(int))
    rel, notes, n_gold = set(), [], 0
    for title, year in q.get("gold", []):
        n_gold += 1
        i = resolve(title, year, tidx, vc)
        if i is None:
            notes.append(f"코퍼스에 없음: {title} ({year})")
        elif i not in allowed:
            notes.append(f"필터 위반(검색 불가): {title} ({year})")
        else:
            rel.add(i)
    if "auto_rel" in q:
        rel |= auto_relevant(s.df, s.genre_sets, q["auto_rel"]) & allowed
    return f, rel, notes, n_gold


# ---------------------------------------------------------------- 지표
def metrics(ranked: list[int], rel: set[int], k: int) -> dict:
    hits = [1 if i in rel else 0 for i in ranked[:k]]
    first = next((r + 1 for r, h in enumerate(hits) if h), None)
    dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
    idcg = sum(1 / np.log2(r + 2) for r in range(min(len(rel), k)))
    return {
        "hit": float(any(hits)),
        "recall": sum(hits) / min(len(rel), k),   # 정답이 k개보다 많으면 k 기준(최대 1.0)
        "mrr": 1 / first if first else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
        "first_rank": first,
    }


# ---------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/queries.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--misses", default="hybrid", help="정답을 top-k 에서 못 찾은 질문을 보여줄 방식")
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args()
    methods = {m: METHODS[m] for m in args.methods.split(",")}

    s = HybridSearcher()
    tidx, vc = build_title_index(s.df)
    label = dict(zip(s.df["id"].astype(int), s.df["title"] + " (" + s.df["year"].astype(str) + ")"))
    queries = load_queries(args.queries)
    print(f"코퍼스 {len(s.df):,}편 / 질문 {len(queries)}건 / k={args.k}\n")

    rows, skipped, all_notes = [], [], []
    gold_total = gold_ok = 0
    for q in queries:
        f, rel, notes, n_gold = relevant_ids(q, s, tidx, vc)
        if q["category"] != "ko_en":   # ko_en 은 ko_direct 와 같은 정답이라 점검 통계에서 중복 제외
            gold_total += n_gold
            gold_ok += n_gold - len(notes)
            all_notes += [f"[{q['id']}] {n}" for n in notes]
        if not rel:
            skipped.append(f"{q['id']} ({q['category']})")
            continue
        vec, bm = s.retrieve(q["text"], f, pool=C.POOL_SIZE)
        for name, cfg in methods.items():
            ids = [int(i) for i, _ in s.fuse(vec, bm, **cfg)]
            rows.append({
                "id": q["id"], "category": q["category"], "method": name,
                "n_rel": len(rel), "query": q["text"],
                "top3": " | ".join(label[i] for i in ids[:3]),
                **metrics(ids, rel, args.k),
            })

    print(f"[정답 라벨 점검] 사람 지정 정답 {gold_ok}/{gold_total}개가 코퍼스에서 확인됨")
    for n in all_notes:
        print("  -", n)
    if skipped:
        print(f"[평가 제외] 정답이 하나도 없는 질문 {len(skipped)}건: {', '.join(skipped)}")
    print()
    if not rows:
        print("평가할 질문이 없습니다. eval/queries.json 의 정답 제목/연도를 코퍼스에 맞게 고쳐 주세요.")
        return

    df = pd.DataFrame(rows)
    order = list(methods)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{args.out}/per_query.csv", index=False, encoding="utf-8-sig")

    overall = df[df["category"] != "ko_en"]      # ko_en 은 ko_direct 와 같은 질문이라 중복 제외
    n_q = overall["id"].nunique()
    summary = overall.groupby("method")[METRICS].mean().reindex(order)
    summary.to_csv(f"{args.out}/summary.csv", encoding="utf-8-sig")
    print(f"=== 전체 평균 (질문 {n_q}건, k={args.k}) ===")
    print(summary.round(3).to_string(), "\n")

    counts = df[df["method"] == order[0]].groupby("category")["id"].count()
    print("카테고리별 질문 수:", ", ".join(f"{c}={n}" for c, n in counts.items()), "\n")
    for metric in ("mrr", "recall"):
        pv = df.pivot_table(index="method", columns="category", values=metric, aggfunc="mean").reindex(order)
        print(f"=== 카테고리별 {metric.upper()}@{args.k} ===")
        print(pv.round(3).to_string(), "\n")

    if args.misses in methods:
        miss = df[(df["method"] == args.misses) & (df["hit"] == 0)]
        print(f"=== '{args.misses}' 가 top-{args.k} 에서 정답을 못 찾은 질문 {len(miss)}건 ===")
        for _, r in miss.head(15).iterrows():
            print(f"  [{r['id']}] {r['query'][:70]}\n      상위3: {r['top3']}")
    print(f"\n저장: {args.out}/per_query.csv, {args.out}/summary.csv")


if __name__ == "__main__":
    main()