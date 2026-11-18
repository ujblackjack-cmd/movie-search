"""하이브리드 검색: (LLM 질문 파싱) -> 메타데이터 필터 + 벡터 검색 + BM25 -> RRF 결합."""
import json
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as C
from common import genre_key, split_genres, tokenize

TMDB_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
    "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance",
    "Science Fiction", "TV Movie", "Thriller", "War", "Western",
]


@dataclass
class Filters:
    year_min: int | None = None
    year_max: int | None = None
    min_rating: float | None = None
    genres: list[str] = field(default_factory=list)   # AND 조건
    language: str | None = None                       # 원어 ISO 코드 (ko, en, ...)
    min_runtime: int | None = None
    max_runtime: int | None = None
    certification: list[str] = field(default_factory=list)  # OR 조건

    def is_empty(self) -> bool:
        return not any(
            [self.year_min, self.year_max, self.min_rating, self.genres, self.language,
             self.min_runtime, self.max_runtime, self.certification]
        )


# ----------------------------------------------------------------------------
# 1) 질문 파싱 (LLM) : 자연어 -> 영어 시맨틱 쿼리 + 구조화 필터
# ----------------------------------------------------------------------------
PARSE_SYSTEM = f"""You convert a movie-search request into JSON. Output ONLY the JSON object.

Schema:
{{"query_en": str,
  "filters": {{"year_min": int|null, "year_max": int|null, "min_rating": float|null,
              "genres": [str], "language": str|null,
              "min_runtime": int|null, "max_runtime": int|null, "certification": [str]}}}}

Rules:
- query_en: English description of what the user wants (themes, mood, plot, "similar to X").
  Keep person and film names in their original spelling. Do not repeat constraints that are
  already captured as filters unless they also describe the desired feel.
- genres: only from {TMDB_GENRES}.
- language: ISO 639-1 code of the film's ORIGINAL language, only if explicitly requested
  (e.g. Korean movies -> "ko").
- min_rating: 0-10 scale, only if the user states a quality bar ("highly rated" -> 7.5).
- Decades: "2010s" -> year_min 2010, year_max 2019. "recent" -> year_min 2020.
- certification values: G, PG, PG-13, R, NC-17, NR.
- Use null / [] when not specified. Never invent constraints."""


def parse_query(question: str, generate_fn=None) -> tuple[str, Filters]:
    """실패하면 필터 없이 원문으로 검색하도록 안전하게 폴백.
    generate_fn(system, user, max_tokens) -> str  (기본: llm.generate, 테스트 때 주입 가능)"""
    try:
        if generate_fn is None:
            from llm import generate as generate_fn
        text = generate_fn(PARSE_SYSTEM, question, 1024)
        data = json.loads(text[text.index("{") : text.rindex("}") + 1])
        f = data.get("filters") or {}
        genres = [g for g in (f.get("genres") or []) if g in TMDB_GENRES]
        filters = Filters(
            year_min=f.get("year_min"), year_max=f.get("year_max"),
            min_rating=f.get("min_rating"), genres=genres,
            language=f.get("language"), min_runtime=f.get("min_runtime"),
            max_runtime=f.get("max_runtime"),
            certification=f.get("certification") or [],
        )
        query = (data.get("query_en") or "").strip() or " ".join(genres) or question
        return query, filters
    except Exception as e:  # noqa: BLE001
        print(f"[parse_query 폴백: {e}]")
        return question, Filters()


# ----------------------------------------------------------------------------
# 2) 검색기
# ----------------------------------------------------------------------------
def rrf(rank_lists: list[list[int]], weights: list[float] | None = None,
        k: int = C.RRF_K) -> list[tuple[int, float]]:
    """가중 RRF: score(d) = sum_i w_i / (k + rank_i(d))"""
    weights = weights or [1.0] * len(rank_lists)
    scores: dict[int, float] = {}
    for lst, w in zip(rank_lists, weights):
        for rank, _id in enumerate(lst):
            scores[_id] = scores.get(_id, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridSearcher:
    def __init__(self, model=None):
        import chromadb

        with open(C.BM25_PATH, "rb") as f:
            payload = pickle.load(f)
        self.bm25, ids = payload["bm25"], payload["ids"]

        # 인덱싱된 id 순서에 코퍼스를 맞춘다 (build_index --limit 으로 만들어도 안전)
        corpus = pd.read_parquet(C.CORPUS_PATH).set_index("id", drop=False)
        self.df = corpus.loc[ids].reset_index(drop=True)
        self.genre_sets = [set(split_genres(g)) for g in self.df["genres"]]
        self.by_id = self.df.set_index("id", drop=False)

        # 품질 사전점수(0~1): 평점/투표수의 코퍼스 내 백분위 평균
        rating = self.df["rating"].fillna(self.df["rating"].median())
        pct = (rating.rank(pct=True) + self.df["vote_count"].rank(pct=True)) / 2
        self.prior_by_id = dict(zip(self.df["id"].astype(int), pct.to_numpy()))

        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(C.EMBED_MODEL)
            model.max_seq_length = C.MAX_SEQ_LEN
        self.model = model

        client = chromadb.PersistentClient(path=str(C.CHROMA_DIR))
        self.col = client.get_collection(C.COLLECTION)

    # ---- 필터: Chroma where 절 / pandas 마스크 (두 구현은 같은 결과여야 함) ----
    @staticmethod
    def build_where(f: Filters) -> dict | None:
        conds = []
        if f.year_min is not None:
            conds.append({"year": {"$gte": int(f.year_min)}})
        if f.year_max is not None:
            conds.append({"year": {"$lte": int(f.year_max)}})
        if f.min_rating is not None:
            conds.append({"rating": {"$gte": float(f.min_rating)}})
        if f.min_runtime is not None:
            conds.append({"runtime": {"$gte": int(f.min_runtime)}})
        if f.max_runtime is not None:
            conds.append({"runtime": {"$lte": int(f.max_runtime)}})
        if f.language:
            conds.append({"language": {"$eq": f.language}})
        if f.certification:
            conds.append({"certification": {"$in": list(f.certification)}})
        for g in f.genres:
            conds.append({genre_key(g): {"$eq": True}})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def build_mask(self, f: Filters) -> np.ndarray:
        d = self.df
        m = np.ones(len(d), dtype=bool)
        if f.year_min is not None:
            m &= (d["year"] >= f.year_min).to_numpy()
        if f.year_max is not None:
            m &= (d["year"] <= f.year_max).to_numpy()
        if f.min_rating is not None:
            m &= (d["rating"] >= f.min_rating).to_numpy()
        if f.min_runtime is not None:
            m &= ((d["runtime"] > 0) & (d["runtime"] >= f.min_runtime)).to_numpy()
        if f.max_runtime is not None:
            m &= ((d["runtime"] > 0) & (d["runtime"] <= f.max_runtime)).to_numpy()
        if f.language:
            m &= (d["language"] == f.language).to_numpy()
        if f.certification:
            m &= d["certification"].isin(f.certification).to_numpy()
        for g in f.genres:
            m &= np.array([g in s for s in self.genre_sets])
        return m

    # ---- 개별 검색기 ----
    def _vector(self, query: str, where, n: int) -> list[int]:
        emb = self.model.encode([query], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=emb, n_results=n, where=where)
        return [int(i) for i in res["ids"][0]]

    def _bm25(self, query: str, mask: np.ndarray, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        scores = np.where(mask, scores, -1.0)
        order = np.argsort(-scores)[:n]
        return [int(self.df["id"].iat[i]) for i in order if scores[i] > 0]

    # ---- 후보 검색 / 결합 / 메인 ----
    def retrieve(self, query: str, filters: Filters | None = None,
                 pool: int = C.POOL_SIZE, use_vec: bool = True,
                 use_bm25: bool = True) -> tuple[list[int], list[int]]:
        """각 검색기의 후보 id 목록(순위순). 평가 때 한 번만 돌리고 결합만 바꿔 보기 위해 분리."""
        filters = filters or Filters()
        where, mask = self.build_where(filters), self.build_mask(filters)
        vec = self._vector(query, where, pool) if use_vec else []
        bm = self._bm25(query, mask, pool) if use_bm25 else []
        return vec, bm

    def fuse(self, vec: list[int], bm: list[int], mode: str = "hybrid",
             w_vec: float = 1.0, w_bm25: float = 1.0,
             quality: float = 0.0) -> list[tuple[int, float]]:
        """mode: hybrid | vector | bm25.
        quality > 0 이면 최종점수 = (RRF점수/최고점수) + quality * 품질사전점수"""
        if mode == "vector":
            lists, ws = [vec], [w_vec]
        elif mode == "bm25":
            lists, ws = [bm], [w_bm25]
        else:
            lists, ws = [vec, bm], [w_vec, w_bm25]
        ranked = rrf(lists, ws)
        if quality and ranked:
            top = ranked[0][1]
            ranked = sorted(
                ((i, sc / top + quality * self.prior_by_id[i]) for i, sc in ranked),
                key=lambda x: -x[1],
            )
        return ranked

    def search(self, query: str, filters: Filters | None = None, k: int = 10,
               mode: str = "hybrid", pool: int = C.POOL_SIZE,
               w_vec: float = 1.0, w_bm25: float = 1.0,
               quality: float = 0.0) -> list[dict]:
        """mode: 'hybrid' | 'vector' | 'bm25'"""
        vec, bm = self.retrieve(query, filters, pool,
                                use_vec=mode in ("hybrid", "vector"),
                                use_bm25=mode in ("hybrid", "bm25"))
        ranked = self.fuse(vec, bm, mode, w_vec, w_bm25, quality)

        out = []
        for _id, score in ranked[:k]:
            r = self.by_id.loc[_id]
            out.append({
                "id": int(_id), "title": r["title"], "year": int(r["year"]),
                "rating": None if pd.isna(r["rating"]) else float(r["rating"]),
                "genres": r["genres"], "director": r["director"], "cast": r["cast"],
                "overview": r["overview"], "poster_path": r["poster_path"],
                "imdb_id": r["imdb_id"], "score": score,
                "vec_rank": vec.index(_id) + 1 if _id in vec else None,
                "bm25_rank": bm.index(_id) + 1 if _id in bm else None,
            })
        return out