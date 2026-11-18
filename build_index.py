"""2단계: 코퍼스 -> 임베딩 -> Chroma(벡터) + BM25(키워드) 인덱스.

    python build_index.py              # 전체
    python build_index.py --limit 500  # 먼저 소량으로 동작 확인
"""
import argparse
import pickle

import chromadb
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config as C
from common import clean, genre_key, split_genres, tokenize


def build_metadata(r: dict) -> dict:
    """Chroma 는 None 값을 허용하지 않으므로 있는 값만 넣는다."""
    m = {"year": int(r["year"]), "vote_count": int(r["vote_count"])}
    if r["rating"] == r["rating"] and r["rating"] is not None:
        m["rating"] = float(r["rating"])
    if r["runtime"] == r["runtime"] and r["runtime"] and r["runtime"] > 0:
        m["runtime"] = int(r["runtime"])
    if clean(r["language"]):
        m["language"] = clean(r["language"])
    if clean(r["certification"]):
        m["certification"] = clean(r["certification"])
    for g in split_genres(r["genres"]):
        m[genre_key(g)] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bm25-only", action="store_true",
                    help="임베딩/Chroma 는 그대로 두고 BM25 인덱스만 다시 만든다 (토크나이저 변경 시)")
    args = ap.parse_args()

    df = pd.read_parquet(C.CORPUS_PATH)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    rows = df.to_dict("records")
    docs = [r["doc"] for r in rows]
    if args.bm25_only:
        bm25 = BM25Okapi([tokenize(d) for d in docs])
        with open(C.BM25_PATH, "wb") as f:
            pickle.dump({"bm25": bm25, "ids": [int(r["id"]) for r in rows]}, f)
        print(f"BM25 만 다시 생성 ({len(rows):,}편). Chroma 는 그대로입니다.")
        return
    print(f"{len(rows):,}편 인덱싱 / 임베딩 모델: {C.EMBED_MODEL}")

    # ---- 벡터 인덱스 ----
    model = SentenceTransformer(C.EMBED_MODEL)
    model.max_seq_length = C.MAX_SEQ_LEN

    client = chromadb.PersistentClient(path=str(C.CHROMA_DIR))
    try:
        client.delete_collection(C.COLLECTION)
    except Exception:
        pass
    col = client.create_collection(C.COLLECTION, metadata={"hnsw:space": "cosine"})

    CHUNK = 2000
    for i in range(0, len(rows), CHUNK):
        part = rows[i : i + CHUNK]
        emb = model.encode(
            [r["doc"] for r in part],
            batch_size=C.EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        col.add(
            ids=[str(r["id"]) for r in part],
            embeddings=emb.tolist(),
            metadatas=[build_metadata(r) for r in part],
            documents=[r["doc"] for r in part],
        )
        print(f"  임베딩 {min(i + CHUNK, len(rows)):,}/{len(rows):,}")

    # ---- BM25 인덱스 ----
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    with open(C.BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [int(r["id"]) for r in rows]}, f)

    print("완료.")


if __name__ == "__main__":
    main()
