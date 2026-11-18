import re


def clean(x) -> str:
    """None / NaN -> '' , 그 외 문자열로."""
    if x is None:
        return ""
    if isinstance(x, float) and x != x:
        return ""
    return str(x).strip()


def split_genres(s) -> list[str]:
    return [g.strip() for g in clean(s).split(",") if g.strip()]


def genre_key(g: str) -> str:
    """Chroma 메타데이터용 장르 플래그 키. 'Science Fiction' -> 'g_Science_Fiction'"""
    return "g_" + re.sub(r"\W+", "_", g.strip())

# BM25 노이즈 방지용 최소 불용어 목록 (her/up/out 처럼 영화 제목이 될 수 있는 단어는 뺐음)
STOPWORDS = set("""a an the and or but nor of in on at to for with by from as into onto over
is are was were be been being am do does did have has had having it its this that these those
he she they his him them their i you we me my your our us who whom whose which what when where
why how not no so if then than too very can could will would shall should may might must just
s t about after before again all any both each more most other some such only own same""".split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())
