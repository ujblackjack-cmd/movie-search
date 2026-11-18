import os
from pathlib import Path

# ---- 경로 ----
DATA_CSV = Path("data/TMDB_all_movies.csv")
OUT_DIR = Path("artifacts")
CORPUS_PATH = OUT_DIR / "corpus.parquet"
CHROMA_DIR = OUT_DIR / "chroma"
BM25_PATH = OUT_DIR / "bm25.pkl"
COLLECTION = "movies"

# ---- 코퍼스 필터 (build_corpus.py 가 편수를 보여주니 보고 조절) ----
MIN_VOTE_COUNT = 200
MIN_OVERVIEW_LEN = 50
TOP_CAST = 5

# ---- 임베딩 ----
# 다국어(한국어 질문 가능): "BAAI/bge-m3"  (CPU 에선 느림, GPU 권장)
# 가볍게 영어만: "BAAI/bge-small-en-v1.5"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_BATCH = 32
MAX_SEQ_LEN = 512

# ---- LLM (질문 파싱 + 답변 생성) ----
# gemini(기본): GEMINI_API_KEY 환경변수 / anthropic: ANTHROPIC_API_KEY 환경변수
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")   # python check_gemini.py --list 로 확인
LLM_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")        # provider=anthropic 일 때만 사용

# ---- 검색 ----
POOL_SIZE = 50     # 각 검색기에서 가져올 후보 수
RRF_K = 60
