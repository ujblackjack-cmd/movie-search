"""1단계: 96만 편 CSV -> 검색 대상 코퍼스(parquet) 생성.

    python build_corpus.py                # 기본 기준
    python build_corpus.py --min-votes 500
"""
import argparse

import duckdb
import pandas as pd

import config as C
from common import clean


def make_document(r: dict) -> str:
    """영화 1편 = 문서 1개. 청킹 불필요."""
    head = f"{clean(r['title'])} ({int(r['year'])})"
    parts = [head]
    if clean(r["tagline"]):
        parts.append(clean(r["tagline"]))
    parts.append(clean(r["overview"]))
    if clean(r["genres"]):
        parts.append("Genres: " + clean(r["genres"]))
    if clean(r["keywords"]):
        parts.append("Keywords: " + ", ".join(clean(r["keywords"]).split("|")))
    if clean(r["director"]):
        parts.append("Director: " + clean(r["director"]))
    cast = [c.strip() for c in clean(r["cast"]).split(",") if c.strip()][: C.TOP_CAST]
    if cast:
        parts.append("Cast: " + ", ".join(cast))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(C.DATA_CSV))
    ap.add_argument("--min-votes", type=int, default=C.MIN_VOTE_COUNT)
    ap.add_argument("--min-overview", type=int, default=C.MIN_OVERVIEW_LEN)
    ap.add_argument("--limit", type=int, default=None, help="디버깅용: 상위 N편만")
    args = ap.parse_args()

    con = duckdb.connect()
    src = f"read_csv_auto('{args.csv}')"
    base_where = f"""
        status = 'Released'
        AND release_date IS NOT NULL
        AND overview IS NOT NULL AND length(overview) >= {args.min_overview}
    """

    # 임계값별 편수 -> 기준 잡을 때 참고
    print("[vote_count 기준별 남는 편수]")
    for v in (0, 50, 100, 200, 500, 1000, 3000):
        n = con.execute(f"SELECT count(*) FROM {src} WHERE {base_where} AND vote_count >= {v}").fetchone()[0]
        print(f"  vote_count >= {v:>5}: {n:,}")

    limit = f"LIMIT {args.limit}" if args.limit else ""
    df = con.execute(f"""
        SELECT id, title, CAST(EXTRACT(year FROM release_date) AS INTEGER) AS year,
               COALESCE(NULLIF(imdb_rating, 0), NULLIF(vote_average, 0)) AS rating,
               vote_count, runtime, original_language AS language,
               certification_us AS certification,
               genres, director, "cast", overview, tagline, keywords,
               poster_path, imdb_id,
               NULLIF(revenue, 0) AS revenue, NULLIF(budget, 0) AS budget
        FROM {src}
        WHERE {base_where} AND vote_count >= {args.min_votes}
        ORDER BY vote_count DESC {limit}
    """).df()

    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    df["doc"] = [make_document(r) for r in df.to_dict("records")]

    C.OUT_DIR.mkdir(exist_ok=True)
    df.to_parquet(C.CORPUS_PATH, index=False)

    print(f"\n저장: {C.CORPUS_PATH}  ({len(df):,}편)")
    print("[연대 분포]\n", (df.year // 10 * 10).value_counts().sort_index().to_string())
    print("[언어 상위]\n", df.language.value_counts().head(8).to_string())
    print("[문서 길이(자)]\n", df.doc.str.len().describe().round(0).to_string())


if __name__ == "__main__":
    main()
