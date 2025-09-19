# server.py
from __future__ import annotations
from flask import Flask, request, jsonify
from flask_cors import CORS
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from youtube_api import search_youtube  # 기존 모듈 그대로 사용

# ---------------------------
# 앱 & CORS 설정
# ---------------------------
app = Flask(__name__)
CORS(app)  # 프론트에서 호출 쉽게

# ---------------------------
# 설정 (빠른 모드 / EasyOCR) + 파일 저장
# ---------------------------
SETTINGS_FILE = Path(__file__).with_name("settings.json")

@dataclass
class Settings:
    fast_mode: bool = True
    use_easyocr: bool = False

def load_settings() -> Settings:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            return Settings(**data)
        except Exception:
            pass
    return Settings()

def save_settings(s: Settings) -> None:
    SETTINGS_FILE.write_text(json.dumps(asdict(s), indent=2, ensure_ascii=False), encoding="utf-8")

settings = load_settings()

@app.route("/settings", methods=["GET"])
def get_settings():
    return jsonify(asdict(settings))

@app.route("/settings", methods=["POST"])
def set_settings():
    global settings
    data = request.get_json(silent=True) or {}
    fast = bool(data.get("fast_mode", settings.fast_mode))
    use_ocr = bool(data.get("use_easyocr", settings.use_easyocr))
    # 정책: 빠른 모드에서도 사용자가 원하면 EasyOCR 사용 허용(경고만 로그)
    if fast and use_ocr:
        print("[INFO] Fast mode ON with EasyOCR -> 속도 저하 가능")
    settings = Settings(fast_mode=fast, use_easyocr=use_ocr)
    save_settings(settings)
    return jsonify(asdict(settings))

# ---------------------------
# 유틸
# ---------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ---------------------------
# 핵심 기능: 문제 분석 + 유튜브 검색
# ---------------------------
@app.route("/analyze", methods=["POST"])
def analyze_problem():
    """
    요청 JSON 예)
    {
      "text": "x + 3 = 7",
      // 향후 확장: "image_path": "C:\\path\\img.jpg" (use_easyocr=True일 때 OCR 시도)
    }
    """
    data = request.get_json(silent=True) or {}
    problem_text = (data.get("text") or "").strip()

    # (선택) EasyOCR 시도: text가 비어있고, 설정에서 use_easyocr가 켜져 있을 때
    # 설치가 안 되어 있어도 서버가 죽지 않도록 try/except 처리
    if not problem_text and settings.use_easyocr:
        img_path = data.get("image_path") or data.get("image") or ""
        if img_path:
            try:
                import easyocr  # heavy dependency, 미설치면 except로 내려감
                reader = easyocr.Reader(['ko', 'en'])
                results = reader.readtext(img_path, detail=0)
                problem_text = " ".join(results).strip()
                print("[INFO] OCR extracted:", problem_text)
            except Exception as e:
                print("[WARN] EasyOCR 사용 실패:", e)

    # 아주 간단한 규칙 (AI 대신 임시 로직)
    txt = problem_text
    if any(sym in txt for sym in ["x", "y", "=", "+", "-", "*", "/"]):
        problem_type = "방정식"
    elif ("삼각형" in txt) or ("사각형" in txt) or ("각" in txt) or ("넓이" in txt):
        problem_type = "도형"
    else:
        problem_type = "일반 문제"

    # 난이도 (임시값)
    difficulty = "초등 수준"

    # 유튜브 검색 (빠른 모드라면 검색어를 더 짧게/단순화 하는 식으로 '가벼운 경로' 예시)
    base_query = f"{problem_type} 풀이"
    query = base_query if settings.fast_mode else f"{base_query} 자세한 설명"
    videos = search_youtube(query)

    return jsonify({
        "ocr_text": problem_text,
        "type": problem_type,
        "difficulty": difficulty,
        "fast_mode": settings.fast_mode,
        "use_easyocr": settings.use_easyocr,
        "query_used": query,
        "videos": videos
    })

# ---------------------------
# 실행
# ---------------------------
if __name__ == "__main__":
    # 기본 포트 5000 유지 (프론트에서 http://127.0.0.1:5000 호출)
    app.run(host="0.0.0.0", port=5000, debug=True)
