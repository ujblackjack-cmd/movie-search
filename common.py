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


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())
