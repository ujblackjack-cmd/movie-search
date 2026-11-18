"""LLM 호출을 한 곳에 모은 래퍼.

    LLM_PROVIDER=gemini(기본) | anthropic
    Gemini:    GEMINI_API_KEY 환경변수, 모델은 config.GEMINI_MODEL (환경변수 GEMINI_MODEL 로 변경)
    Anthropic: ANTHROPIC_API_KEY 환경변수

참고: Gemini 3.5 Flash-Lite / 3.6 Flash 이후 모델은 temperature, top_p, top_k 를 받지 않으므로
의도적으로 넘기지 않습니다.
"""
import os
import time

import config as C

RETRY_CODES = {429, 500, 503}   # 한도 초과 / 일시적 서버 오류


def _gemini(system: str, user: str, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise RuntimeError("GEMINI_API_KEY 환경변수가 없습니다 (Google AI Studio 에서 발급).")
    client = genai.Client()
    resp = client.models.generate_content(
        model=C.GEMINI_MODEL,
        contents=user,
        # max_output_tokens 에는 생각(thinking) 토큰도 포함되므로 넉넉하게 줍니다.
        config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens),
    )
    return resp.text or ""


def _anthropic(system: str, user: str, max_tokens: int) -> str:
    import anthropic

    resp = anthropic.Anthropic().messages.create(
        model=C.LLM_MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def generate(system: str, user: str, max_tokens: int = 1024, retries: int = 3) -> str:
    fn = _anthropic if C.LLM_PROVIDER == "anthropic" else _gemini
    for attempt in range(retries + 1):
        try:
            return fn(system, user, max_tokens)
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if code in RETRY_CODES and attempt < retries:
                wait = 5 * 2**attempt   # 5, 10, 20초 (무료 티어는 분당 한도가 작음)
                print(f"[LLM 재시도 {attempt + 1}/{retries}: HTTP {code}, {wait}초 대기]")
                time.sleep(wait)
                continue
            raise