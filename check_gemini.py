"""Gemini 연결 점검.

    $env:GEMINI_API_KEY="발급받은_키"
    python check_gemini.py            # 설정된 모델로 아주 짧은 호출
    python check_gemini.py --list     # 이 키로 쓸 수 있는 모델 목록
    $env:GEMINI_MODEL="다른-모델-ID"; python check_gemini.py
"""
import argparse
import os
import sys

import config as C
from llm import generate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true")
args = ap.parse_args()

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    sys.exit("GEMINI_API_KEY 환경변수가 없습니다. 현재 터미널에서 먼저 설정하세요.")

if args.list:
    from google import genai
    for m in genai.Client().models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "gemini" in m.name and (not actions or "generateContent" in actions):
            print(m.name.replace("models/", ""))
    sys.exit(0)

print(f"모델: {C.GEMINI_MODEL}")
try:
    out = generate("Reply with JSON only.", 'Return exactly {"ok": true}', max_tokens=256, retries=1)
    print("응답:", out.strip()[:200] or "(빈 응답)")
    print("성공" if out.strip() else "호출은 됐지만 응답이 비어 있음 -> max_tokens/모델을 확인")
except Exception as e:  # noqa: BLE001
    code = getattr(e, "code", None)
    hint = {
        400: "API 키가 잘못되었거나, 요청 형식/지역 제한 문제일 수 있습니다.",
        403: "키 권한 문제입니다. AI Studio 에서 키를 새로 만들어 보세요.",
        404: "모델 ID 가 없습니다. python check_gemini.py --list 로 쓸 수 있는 모델을 확인하세요.",
        429: "무료 한도 초과입니다. 잠시 뒤 다시 하거나 다른 모델로 바꿔 보세요.",
    }.get(code, "")
    print(f"실패: {type(e).__name__} (code={code})\n{str(e)[:400]}\n{hint}")