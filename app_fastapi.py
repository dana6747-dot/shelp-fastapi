# app_fastapi.py (고친판: NameError 픽스 + 추가검색어 + 유튜브 HTML 파싱 백업)
# - NameError: use_easyocr → use_easyocr_flag 로 통일
# - "추가 검색어(선택)" 폼 지원(extra)
# - YouTube API → Piped → Invidious → YouTube HTML 파싱 → 최종 검색링크

import os, io, re, json, traceback, urllib.parse
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ─────────────── .env ───────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ─────────────── OCR 기본값(Windows) ───────────────
os.environ.setdefault("TESSDATA_PREFIX", r"C:\Program Files\Tesseract-OCR\tessdata")
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import requests
import httpx
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from urllib.parse import urlparse, parse_qs

# ─────────────── fallback 필터 ───────────────
try:
    from skimage.filters import threshold_sauvola
except Exception:
    def threshold_sauvola(arr, window_size=25, k=0.2, r=None):
        arr = arr.astype(np.float32)
        ksize = int(max(3, window_size//2*2+1))
        mean = cv2.boxFilter(arr, ddepth=-1, ksize=(ksize, ksize))
        mean_sq = cv2.boxFilter(arr*arr, ddepth=-1, ksize=(ksize, ksize))
        var = np.clip(mean_sq - mean*mean, 0, None)
        std = np.sqrt(var)
        R = r if r is not None else (arr.max() - arr.min() + 1e-6)
        thresh = mean * (1 + k*((std/R)-1))
        return thresh

# ─────────────── EasyOCR(있으면 사용) ───────────────
try:
    import easyocr
    EASY_AVAILABLE = True
    EASY_READER = easyocr.Reader(['ko','en'], gpu=False, verbose=False)
except Exception:
    EASY_AVAILABLE = False
    EASY_READER = None

SESSION = requests.Session()

# ─────────────── FastAPI ───────────────
app = FastAPI(title="Study OCR → Video Search (FastAPI)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ─────────────── 설정 저장 ───────────────
SETTINGS_FILE = Path(__file__).with_name("settings.json")

class Settings(BaseModel):
    fast_mode: bool = False
    use_easyocr: bool = True

def load_settings() -> Settings:
    if SETTINGS_FILE.exists():
        try:
            return Settings(**json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return Settings()

def save_settings(s: Settings) -> None:
    SETTINGS_FILE.write_text(s.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")

RUNTIME = load_settings()

@app.get("/settings", response_model=Settings)
def get_settings(): return RUNTIME

@app.post("/settings", response_model=Settings)
def set_settings(s: Settings):
    global RUNTIME
    RUNTIME = s
    save_settings(RUNTIME)
    return RUNTIME

# ─────────────── OCR 유틸 ───────────────
def _tess_lang() -> str:
    base = os.environ.get("TESSDATA_PREFIX", r"C:\Program Files\Tesseract-OCR\tessdata")
    lang = "kor+eng"
    try:
        if os.path.exists(os.path.join(base, "equ.traineddata")):
            lang += "+equ"
    except Exception:
        pass
    return lang

def _is_hangul(c: str) -> bool:
    o = ord(c); return 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF

def _score_text_ko(s: str) -> int:
    s = s or ""
    hangul = sum(_is_hangul(c) for c in s)
    digits = sum(c.isdigit() for c in s)
    noise  = s.count("?") + s.count(" ")
    bonus  = 0
    for kw in ["자연수","분수","혼합수","약분","통분","문제","해설","구하시오","빈칸","되는 수"]:
        if kw in s: bonus += 3
    length_penalty = 5 if len(s.strip()) < 12 else 0
    return hangul*3 + digits*1 + bonus - noise*2 - length_penalty

FIXES = {
    "가연수":"자연수","기산":"다음","결파":"결과","구하 세요":"구하시오","구하세요":"구하시오","우름":"수","TH":"수",
    "긴파기":"빈칸","빈카기":"빈칸","빈강":"빈칸","긴칸":"빈칸",
    "후":"수","후에":"수에","후가":"수가","후로":"수로",
    "인의":"칸에","인어":"칸에","인에":"칸에",
    "tal":"값","지인수":"자연수",
    "자 연 수":"자연수","문 제":"문제","빈 칸":"빈칸","들 어 갈":"들어갈","모 두":"모두","계 산":"계산","해 설":"해설",
    " {":" (","{":"(", "}":")", " [":" (","[":"(", "]":")",
}
def _clean_text(s: str) -> str:
    s = s or ""
    for k,v in FIXES.items(): s = s.replace(k,v)
    s = re.sub(r"[\u0000-\u001F]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith(("(", "[", "{", "“", "\"", "'")): s = s[:-1].strip()
    return s

def rotate_img(img_np: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:   return img_np
    if angle == 90:  return cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180: return cv2.rotate(img_np, cv2.ROTATE_180)
    if angle == 270: return cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_np

def rotate_pil(pil: Image.Image, angle_cw: int) -> Image.Image:
    angle_cw %= 360
    mapping = {0:0, 90:-90, 180:180, 270:-270}
    deg = mapping.get(angle_cw, 0)
    return pil if deg == 0 else pil.rotate(deg, expand=True)

def unsharp(gray: np.ndarray, amount: float = 1.1, sigma: float = 0.9) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=sigma)
    sharp = cv2.addWeighted(gray, 1+amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def detect_osd_angle(pil: Image.Image) -> int:
    try:
        osd = pytesseract.image_to_osd(pil)
        m = re.search(r"Rotate:\s+(\d+)", osd)
        if m:
            ang = int(m.group(1)) % 360
            if ang in (0,90,180,270): return ang
    except Exception: pass
    return 0

def auto_warp_document(bgr: np.ndarray) -> np.ndarray:
    try:
        h,w = bgr.shape[:2]
        img = bgr.copy()
        scale = 800.0 / max(h,w)
        small = cv2.resize(img, (int(w*scale), int(h*scale))) if scale < 1.0 else img.copy()
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        doc = None; max_area = 0
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area: max_area, doc = area, approx
        if doc is None: return bgr
        pts = doc.reshape(4,2).astype(np.float32) / (scale if scale < 1.0 else 1.0)
        rect = np.zeros((4,2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1); rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
        (tl,tr,br,bl) = rect
        maxW = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl))); maxW = max(maxW, 700)
        maxH = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl))); maxH = max(maxH, 700)
        dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (maxW, maxH))
    except Exception:
        return bgr

def remove_grid_lines(bw: np.ndarray) -> np.ndarray:
    h, w = bw.shape[:2]
    hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w//25), 1))
    ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h//25)))
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hor_k, iterations=1)
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_k, iterations=1)
    lines = cv2.bitwise_or(hor, ver)
    return cv2.bitwise_and(bw, cv2.bitwise_not(lines))

def preprocess_variant(pil_img: Image.Image, mode: str, do_warp: bool, target: int = 2200) -> np.ndarray:
    bgr = np.array(pil_img.convert("RGB"))[:, :, ::-1]
    ang0 = detect_osd_angle(pil_img)
    bgr = rotate_img(bgr, ang0)
    if do_warp: bgr = auto_warp_document(bgr)
    h,w = bgr.shape[:2]
    if max(h,w) < target:
        s = target / max(h,w)
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    bgr = cv2.bilateralFilter(bgr, d=7, sigmaColor=35, sigmaSpace=35)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = unsharp(gray, amount=1.1, sigma=0.9)

    if mode == "adaptive":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        g2 = clahe.apply(gray)
        bg = cv2.medianBlur(g2, 21)
        norm = cv2.divide(g2, bg, scale=255)
        bw = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 12)
    elif mode == "otsu":
        g2 = cv2.GaussianBlur(gray, (5,5), 0)
        _, bw = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "sauvola":
        window = 25
        th = threshold_sauvola(gray, window_size=window, k=0.2)
        bw = (gray > th).astype(np.uint8) * 255
    elif mode == "degrid":
        g2 = cv2.GaussianBlur(gray, (3,3), 0)
        _, bw0 = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(bw0) < 127: bw0 = cv2.bitwise_not(bw0)
        bw = remove_grid_lines(bw0)
    else:
        bw = gray

    if np.mean(bw) < 110: bw = cv2.bitwise_not(bw)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker, iterations=1)
    bw = cv2.medianBlur(bw, 3)
    return bw

def fast_ocr(pil: Image.Image, angle: int) -> Tuple[str, str]:
    degrid = preprocess_variant(pil, "degrid", True, target=2000)
    if angle in (0,90,180,270): degrid = rotate_img(degrid, angle)
    cfg = (rf"--oem 1 --psm 6 -c user_defined_dpi=300 -c preserve_interword_spaces=1")
    t1 = _clean_text(pytesseract.image_to_string(degrid, lang=_tess_lang(), config=cfg))
    s1 = _score_text_ko(t1)
    if s1 < 8:
        adap = preprocess_variant(pil, "adaptive", False, target=2000)
        if angle in (0,90,180,270): adap = rotate_img(adap, angle)
        t2 = _clean_text(pytesseract.image_to_string(adap, lang=_tess_lang(), config=cfg))
        if _score_text_ko(t2) > s1: return t2, "tesseract|fast:adaptive"
    return t1, "tesseract|fast:degrid"

def best_ocr_from_variants(pil: Image.Image) -> Tuple[str, str]:
    variants: List[Tuple[int, str, str]] = []
    for mode in ("adaptive", "degrid", "sauvola", "otsu"):
        for warp in (True, False):
            pre = preprocess_variant(pil, mode, warp)
            for angle in (0,90,180,270):
                img = rotate_img(pre, angle)
                for psm in (4,6,11):
                    cfg = (rf"--oem 1 --psm {psm} -c user_defined_dpi=330 -c preserve_interword_spaces=1")
                    text = pytesseract.image_to_string(img, lang=_tess_lang(), config=cfg)
                    text = _clean_text(text)
                    score = _score_text_ko(text)
                    meta = f"tesseract|mode:{mode}, warp:{warp}, ang:{angle}, psm:{psm}"
                    variants.append((score, meta, text))
    variants.sort(key=lambda x: x[0], reverse=True)
    return (variants[0][2], variants[0][1]) if variants else ("", "none")

def easyocr_best(pil: Image.Image):
    if not EASY_AVAILABLE: return ("", -999, 0, "")
    best_text, best_score, best_hangul, best_meta = "", -999, 0, ""
    rgb_raw0 = np.array(pil.convert("RGB"))
    def _score_text(t: str): return _score_text_ko(t), sum(_is_hangul(c) for c in t)
    for src_name, base in (("raw", rgb_raw0),):
        for ang in (0,90,180,270):
            rgb = rotate_img(base, ang)
            try:
                lines = EASY_READER.readtext(rgb, detail=True, paragraph=True)
                texts = []
                for _, txt, conf in lines:
                    if conf is None: continue
                    if conf >= 0.15 and (any(_is_hangul(c) for c in txt) or any(ch.isdigit() for c in txt)):
                        texts.append(txt)
                t = _clean_text(" ".join(texts))
                sc, hn = _score_text(t)
                if sc > best_score or (sc == best_score and hn > best_hangul):
                    best_text, best_score, best_hangul = t, sc, hn
                    best_meta = f"easyocr|src:{src_name}, ang:{ang}, para:1"
            except Exception:
                continue
    return best_text, best_score, best_hangul, best_meta

def best_ocr_with_easyocr(pil: Image.Image, allow_easy: bool) -> Tuple[str, str]:
    if allow_easy and EASY_AVAILABLE:
        e_text, e_score, e_hangul, e_meta = easyocr_best(pil)
        if len(e_text.strip()) >= 3: return e_text, e_meta
    t_text, t_meta = best_ocr_from_variants(pil)
    if allow_easy and EASY_AVAILABLE:
        easy_text, easy_score, easy_hangul, easy_meta = easyocr_best(pil)
        tess_score = _score_text_ko(t_text)
        tess_hangul = sum(_is_hangul(c) for c in t_text)
        if easy_score >= tess_score + 1: return easy_text, easy_meta
        if easy_hangul >= tess_hangul + 3 and len(easy_text) >= len(t_text) * 0.9: return easy_text, easy_meta
        if abs(easy_score - tess_score) <= 3 and easy_hangul >= tess_hangul: return easy_text, easy_meta
        if len(easy_text) > len(t_text) * 1.25: return easy_text, easy_meta
    return t_text, t_meta

# ─────────────── 검색 유틸 ───────────────
def _yt_id_from_url(u: str) -> str:
    try:
        p = urlparse(u); host = (p.hostname or "").lower()
        if "youtube.com" in host or "youtu.be" in host:
            qs = parse_qs(p.query)
            if "v" in qs and qs["v"]: return qs["v"][0]
            if host == "youtu.be" and p.path.strip("/"): return p.path.strip("/")
            if "/shorts/" in p.path: return p.path.split("/shorts/")[1].split("/")[0]
            if "/embed/" in p.path: return p.path.split("/embed/")[1].split("/")[0]
    except Exception: pass
    return ""

def piped_search_sync(query: str, region: str = "KR", max_results: int = 20) -> List[Dict]:
    instances = ["https://piped.video","https://piped.kavin.rocks","https://piped.tokhmi.xyz"]
    headers = {"Accept":"application/json"}
    for base in instances:
        for filt in ({"filter":"videos"},{"searchFilters":"videos"}):
            try:
                with httpx.Client(verify=False, timeout=10, follow_redirects=True, trust_env=True) as client:
                    r = client.get(f"{base}/api/v1/search", params={"q":query,"region":region,**filt}, headers=headers)
                    r.raise_for_status()
                    data = r.json()
                out = []
                if isinstance(data, list):
                    for e in data:
                        if not isinstance(e, dict): continue
                        typ = str(e.get("type") or "").lower()
                        if typ and typ not in ("video","stream"): continue
                        url_path = str(e.get("url") or "")
                        if url_path.startswith("http"): full = url_path
                        elif url_path: full = "https://www.youtube.com" + url_path
                        else: full = ""
                        vid = _yt_id_from_url(full) or str(e.get("id") or "")
                        if not vid: continue
                        title = str(e.get("title") or f"YouTube 영상 ({vid})")
                        thumb = e.get("thumbnail") or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
                        out.append({
                            "videoId": vid, "title": title, "description": "",
                            "channel": "", "publishedAt": "", "thumb": str(thumb),
                            "url": f"https://www.youtube.com/watch?v={vid}", "query": query,
                        })
                        if len(out) >= max_results: break
                if out: return out
            except Exception:
                continue
    return []

def invidious_search_sync(query: str, region: str = "KR", max_results: int = 20) -> List[Dict]:
    bases = ["https://yewtu.be","https://vid.puffyan.us","https://invidious.protokolla.fi"]
    headers = {"Accept":"application/json"}
    for base in bases:
        try:
            with httpx.Client(verify=False, timeout=10, follow_redirects=True, trust_env=True) as client:
                r = client.get(f"{base}/api/v1/search",
                               params={"q":query,"type":"video","region":region},
                               headers=headers)
                r.raise_for_status()
                data = r.json()
            out = []
            if isinstance(data, list):
                for e in data:
                    vid = str(e.get("videoId") or "")
                    if not vid: continue
                    title = str(e.get("title") or f"YouTube 영상 ({vid})")
                    thumb = (e.get("videoThumbnails") or [{}])[-1].get("url") or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
                    out.append({
                        "videoId": vid, "title": title, "description":"",
                        "channel": str(e.get("author") or ""), "publishedAt":"",
                        "thumb": str(thumb),
                        "url": f"https://www.youtube.com/watch?v={vid}", "query": query,
                    })
                    if len(out) >= max_results: break
            if out: return out
        except Exception:
            continue
    return []

# ─────────────── 유튜브 HTML(PC/모바일) 파싱 백업 ───────────────
def _extract_from_ytinitialdata(html: str, max_results: int) -> List[Dict]:
    try:
        m = re.search(r'ytInitialData"\s*:\s*(\{.*?\})\s*[,<]', html, re.S)
        if not m: return []
        j = json.loads(m.group(1))
        def walk(node):
            if isinstance(node, dict):
                if "videoRenderer" in node:
                    yield node["videoRenderer"]
                for v in node.values():
                    yield from walk(v)
            elif isinstance(node, list):
                for v in node:
                    yield from walk(v)
        out = []
        for vr in walk(j):
            vid = (vr.get("videoId") or "") if isinstance(vr, dict) else ""
            if not vid: continue
            title_runs = (((vr.get("title") or {}).get("runs") or [{}]))
            title = title_runs[0].get("text", f"YouTube 영상 ({vid})")
            thumbs = (((vr.get("thumbnail") or {}).get("thumbnails")) or [])
            thumb = thumbs[-1]["url"] if thumbs else f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
            out.append({"videoId":vid,"title":title,"thumb":thumb,"url":f"https://www.youtube.com/watch?v={vid}"})
            if len(out) >= max_results: break
        return out
    except Exception:
        return []

def _extract_by_regex(html: str, max_results: int) -> List[Dict]:
    ids = []
    seen = set()
    for m in re.finditer(r'"videoId"\s*:\s*"([a-zA-Z0-9_-]{11})"', html):
        vid = m.group(1)
        if vid not in seen:
            ids.append(vid); seen.add(vid)
        if len(ids) >= max_results: break
    if len(ids) < max_results:
        for m in re.finditer(r'watch\?v=([a-zA-Z0-9_-]{11})', html):
            vid = m.group(1)
            if vid not in seen:
                ids.append(vid); seen.add(vid)
            if len(ids) >= max_results: break
    out = [{"videoId":vid,"title":f"YouTube 영상 ({vid})","thumb":f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg","url":f"https://www.youtube.com/watch?v={vid}"} for vid in ids]
    return out

def youtube_html_fallback_search(query: str, max_results: int = 10) -> List[Dict]:
    try:
        variants = [
            ("https://www.youtube.com/results", {"search_query": query},
             {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)","Accept-Language":"ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"}),
            ("https://m.youtube.com/results", {"search_query": query},
             {"User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1","Accept-Language":"ko-KR,ko;q=0.9,en;q=0.8"}),
        ]
        with httpx.Client(verify=False, timeout=10, follow_redirects=True, trust_env=True) as client:
            client.cookies.set("CONSENT", "YES+")
            for url, params, headers in variants:
                try:
                    r = client.get(url, params=params, headers=headers)
                    r.raise_for_status()
                    html = r.text
                    out = _extract_from_ytinitialdata(html, max_results)
                    if not out:
                        out = _extract_by_regex(html, max_results)
                    if out:
                        return out
                except Exception:
                    continue
        return []
    except Exception:
        return []

def youtube_search(query: str, max_results: int = 18, region: str = "KR") -> List[Dict]:
    if YOUTUBE_API_KEY:
        try:
            r = SESSION.get("https://www.googleapis.com/youtube/v3/search",
                params={
                    "part":"snippet","q":query,"type":"video","maxResults":max_results,
                    "order":"relevance","videoEmbeddable":"true","relevanceLanguage":"ko",
                    "regionCode":region,"key":YOUTUBE_API_KEY
                }, timeout=8)
            if r.status_code == 200:
                items = r.json().get("items", [])
                out=[]
                for it in items:
                    vid = it.get("id", {}).get("videoId"); sn = it.get("snippet", {})
                    if not vid: continue
                    thumb = (sn.get("thumbnails", {}).get("medium", {}) or sn.get("thumbnails", {}).get("default", {})).get("url")
                    out.append({"videoId":vid,"title":sn.get("title",""),"description":sn.get("description",""),
                                "channel":sn.get("channelTitle",""),"publishedAt":sn.get("publishedAt",""),
                                "thumb":thumb,"url":f"https://www.youtube.com/watch?v={vid}","query":query})
                if out: return out
        except Exception:
            pass
    out = piped_search_sync(query, region=region, max_results=max_results)
    if out: return out
    out = invidious_search_sync(query, region=region, max_results=max_results)
    if out: return out
    out = youtube_html_fallback_search(query, max_results=max_results)
    if out: return out
    return [{
        "videoId": None,
        "title": f"유튜브 검색: {query}",
        "description": "",
        "channel": "",
        "publishedAt": "",
        "thumb": "",
        "url": f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}",
        "query": query,
    }]

# ─────────────── 결과 정렬 ───────────────
_KO_WORD = re.compile(r"[가-힣]{2,}")
_EN_WORD = re.compile(r"[A-Za-z]{3,}")
_DIGITS  = re.compile(r"\d+")

def _tok(s: str):
    s = (s or "").lower()
    return set([*_KO_WORD.findall(s), *_EN_WORD.findall(s), *_DIGITS.findall(s)])

def _overlap(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    return inter / (len(a) ** 0.5 * len(b) ** 0.5)

def score_video(item: dict, queries: list, ocr_text: str) -> float:
    title = (item.get("title") or "")
    t = _tok(title); oq = _tok(" ".join(queries)); ot = _tok(ocr_text)
    s1 = _overlap(t, oq) * 1.0; s2 = _overlap(t, ot) * 1.2
    bonus = 0.4 if any(k in title for k in ["초등","수학","문제","해설","풀이","분수","혼합수","자연수","약분","통분"]) else 0.0
    if "shorts" in title.lower() or "#shorts" in title.lower(): bonus -= 0.4
    return s1 + s2 + bonus

def youtube_search_multi(queries: List[str], max_total: int = 12, ocr_text: str = "") -> List[Dict]:
    results: List[Dict] = []
    if not queries: return results
    with ThreadPoolExecutor(max_workers=min(4, len(queries))) as ex:
        futs = {ex.submit(youtube_search, q, 18): q for q in queries}
        for f in as_completed(futs):
            try:
                res = f.result()
                if res: results.extend(res)
            except Exception:
                continue
    uniq = {}
    for v in results:
        vid = v.get("videoId")
        if vid and vid not in uniq: uniq[vid] = v
    items = list(uniq.values())
    items.sort(key=lambda it: score_video(it, queries, ocr_text), reverse=True)
    no_vid = [v for v in results if not v.get("videoId")]
    final = items[:max_total]
    if not final and no_vid: final.append(no_vid[0])
    return final

def run_ocr_and_search(
    pil: Image.Image,
    angle: int,
    fast_flag: Optional[bool],
    easy_flag: Optional[bool],
    extra_q: Optional[str] = None,
):
    if angle in (0,90,180,270): pil = rotate_pil(pil, angle)
    eff_fast = (bool(fast_flag) if fast_flag is not None else RUNTIME.fast_mode)
    eff_easy = (bool(easy_flag) if easy_flag is not None else RUNTIME.use_easyocr)

    if eff_fast:
        text, meta = fast_ocr(pil, angle)
        q_all = build_queries(text); queries = q_all[:2]
    else:
        text, meta = best_ocr_with_easyocr(pil, allow_easy=eff_easy)
        queries = build_queries(text)

    # 추가 검색어 맨 앞에 우선 적용
    if extra_q:
        ex = extra_q.strip()
        if ex and ex not in queries:
            queries = [ex] + queries

    videos = youtube_search_multi(queries, max_total=12, ocr_text=text)

    return {
        "ok": True, "ocr_text": text, "queries": queries, "videos": videos, "meta": meta,
        "fast": bool(eff_fast), "use_easyocr": bool(eff_easy)
    }

KOR_STOP = {"다음","계산","구하시오","구하여라","문제","조건","모두","수","풀이","빈칸","보기","설명하라"}
FORMULA_PAT = re.compile(r"[0-9xX\^_+\-*/=(){}\[\]π√∑∫%<>]+")

def build_queries(text: str) -> List[str]:
    text = (text or "").strip()
    tokens = [t for t in re.split(r"[^\w가-힣]+", text) if t]
    tok_kr = [x for x in tokens if re.match(r"[가-힣]", x) and x not in KOR_STOP][:10]
    is_frac = any(k in text for k in ["분수","혼합수"]); is_nat = "자연수" in text
    has_blank = any(k in text for k in ["빈칸","ㅁ","□"])
    grade_hint = " 초등 수학" + (" 5학년" if (is_frac or is_nat) else "")
    formulas = FORMULA_PAT.findall(text.replace(" ", ""))[:2]
    base_words = []
    if is_frac: base_words += ["분수의 곱셈","혼합수 곱셈"]
    if is_nat:  base_words += ["자연수 조건"]
    if has_blank: base_words += ["빈칸에 들어갈 수"]
    if not base_words: base_words = ["분수","혼합수","곱셈","자연수","문제","풀이","해설"]
    base = " ".join(sorted(set(tok_kr), key=tok_kr.index))[:60]
    cand = [
        (" ".join(base_words) + grade_hint + " 해설").strip(),
        (" ".join(base_words) + grade_hint + " 문제 풀이").strip(),
    ]
    if base: cand.append((base + " " + " ".join(base_words[:2]) + grade_hint + " 해설").strip())
    if formulas:
        fstr = " ".join(formulas).rstrip("([{\"'")
        cand.append(("중요 식 " + fstr + grade_hint).strip())
    out = []
    for q in cand:
        q = q.replace("문제 풀이 문제 풀이","문제 풀이")
        q = re.sub(r"\s{2,}"," ",q).strip()
        if q and q not in out: out.append(q)
    if out: out.append(out[0].replace("해설","설명"))
    return out[:4]

# ─────────────── Endpoints ───────────────
@app.get("/")
def root(): return {"ok": True, "message": "alive"}

@app.post("/image-search")
async def image_search(
    file: UploadFile = File(...),
    angle: int = Form(0),
    fast: Optional[int] = Form(None),
    use_easyocr_flag: Optional[int] = Form(None),   # ← 이름 통일
    extra: Optional[str] = Form(None),              # ← 추가 검색어
):
    try:
        img_bytes = await file.read()
        if not img_bytes: raise HTTPException(status_code=400, detail="empty upload")
        try:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="not an image")
        result = run_ocr_and_search(
            pil=pil, angle=angle,
            fast_flag=bool(fast) if fast is not None else None,
            easy_flag=bool(use_easyocr_flag) if use_easyocr_flag is not None else None,
            extra_q=extra,
        )
        return result
    except HTTPException as he:
        return {"ok": False, "error": he.detail}
    except Exception as e:
        traceback.print_exc(); return {"ok": False, "error": str(e)}

@app.post("/search_from_image")
async def search_from_image(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    upload: Optional[UploadFile] = File(None),
    picture: Optional[UploadFile] = File(None),
    photo: Optional[UploadFile] = File(None),
    img: Optional[UploadFile] = File(None),
    media: Optional[UploadFile] = File(None),
    angle: int = Form(0),
    fast: Optional[int] = Form(None),
    use_easyocr_flag: Optional[int] = Form(None),   # ← 이름 통일
    extra: Optional[str] = Form(None),              # ← 추가 검색어
):
    try:
        chosen: Optional[UploadFile] = None
        for cand in (image, file, image_file, upload, picture, photo, img, media):
            if cand is not None: chosen = cand; break
        if chosen is None: raise HTTPException(status_code=400, detail="no file field")
        img_bytes = await chosen.read()
        if not img_bytes: raise HTTPException(status_code=400, detail="empty upload")
        try:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="not an image")
        result = run_ocr_and_search(
            pil=pil, angle=angle,
            fast_flag=bool(fast) if fast is not None else None,
            easy_flag=bool(use_easyocr_flag) if use_easyocr_flag is not None else None,
            extra_q=extra,
        )
        return result
    except HTTPException as he:
        return {"ok": False, "error": he.detail}
    except Exception as e:
        traceback.print_exc(); return {"ok": False, "error": str(e)}

@app.get("/search_by_text")
def search_by_text(q: str = Query(..., min_length=1)):
    videos = youtube_search_multi([q], max_total=12, ocr_text=q)
    return {"ok": True, "query": q, "results": videos, "videos": videos}

@app.get("/search_youtube")
def search_youtube_endpoint(q: str = Query(..., min_length=1), maxResults: int = 5):
    q = (q or "").strip()
    if not q:
        return {"results": []}

    if YOUTUBE_API_KEY:
        try:
            r = SESSION.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet", "q": q, "type": "video", "maxResults": maxResults,
                    "order": "relevance", "videoEmbeddable": "true", "relevanceLanguage": "ko",
                    "regionCode": "KR", "key": YOUTUBE_API_KEY,
                }, timeout=8,
            )
            if r.status_code == 200:
                items = r.json().get("items", [])
                results = []
                for it in items:
                    vid = (it.get("id") or {}).get("videoId")
                    if not vid: continue
                    snip = it.get("snippet") or {}
                    title = snip.get("title") or f"YouTube 영상 ({vid})"
                    thumb = ((snip.get("thumbnails", {}).get("medium", {})) or (snip.get("thumbnails", {}).get("default", {})) or {}).get("url", "")
                    results.append({"title": title, "videoId": vid, "url": f"https://www.youtube.com/watch?v={vid}", "thumbnail": thumb})
                if results: return {"results": results}
        except Exception:
            pass

    items = piped_search_sync(q, region="KR", max_results=maxResults * 3)
    if not items:
        items = invidious_search_sync(q, region="KR", max_results=maxResults * 3)
    if items:
        return {"results":[{"title":it.get("title") or f"YouTube 영상 ({it.get('videoId')})","videoId":it.get("videoId") or "","url":it.get("url") or "","thumbnail":it.get("thumb") or ""} for it in items[:maxResults]]}

    items = youtube_html_fallback_search(q, max_results=maxResults)
    if items:
        return {"results":[{"title":it.get("title"),"videoId":it.get("videoId"),"url":it.get("url"),"thumbnail":it.get("thumb")} for it in items]}

    return {"results":[{"title": f"유튜브 검색: {q}","videoId": "","url": f"https://www.youtube.com/results?search_query={urllib.parse.quote(q)}","thumbnail": ""}]}

# ─────────────── 테스트 UI (/ui) ───────────────
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Study OCR Search</title>
<style>
  body{font-family:sans-serif;margin:24px}
  .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .card{display:flex;gap:12px;align-items:center;border:1px solid #ddd;padding:12px;border-radius:12px;margin-top:12px}
  img{border-radius:8px;max-width:480px}
  pre{background:#f7f7f7;padding:12px;border-radius:8px;overflow:auto}
  button{padding:8px 14px;border:1px solid #ccc;border-radius:8px;background:#fff;cursor:pointer}
  #status{margin-top:10px;color:#555}
  #error{margin-top:10px;color:#c00}
  label{user-select:none}
  #toolbar.disabled *{opacity:0.5;pointer-events:none}
  #cropOverlay{position:fixed; inset:0; background:rgba(0,0,0,.6);
    display:none; align-items:center; justify-content:center; z-index:9999;}
  #cropBox{background:#111; padding:12px; border-radius:12px; color:#fff;
    display:flex; flex-direction:column; gap:8px; max-width:95vw; max-height:95vh;}
  #cropCanvas{background:#222; border-radius:8px; cursor:crosshair; max-width:90vw; max-height:70vh}
  #cropActions{display:flex; gap:8px; justify-content:flex-end}
  .pill{border-radius:999px}
  body.cropping #toolbar { display:none !important; }
</style>
</head>
<body>
  <h2>사진 → 풀이 영상 검색 <small id="ver" style="font-weight:normal;color:#888"></small></h2>

  <div id="toolbar" class="row" style="margin-bottom:10px">
    <input type="file" id="file" accept="image/*">
    <button type="button" id="rotL">↺ 90°</button>
    <button type="button" id="rotR">↻ 90°</button>
    <button type="button" id="cropBtn">자르기</button>
    <input id="extra" placeholder="추가 검색어(선택)" style="min-width:240px;padding:6px 10px;border:1px solid #ccc;border-radius:8px">
    <label><input type="checkbox" id="fast"> 빠른 모드</label>
    <label><input type="checkbox" id="use_easyocr"> EasyOCR 사용</label>
    <button type="button" id="btn" class="pill">검색</button>
    <button type="button" id="save">설정 저장</button>
  </div>

  <div id="status"></div>
  <div id="error"></div>

  <div style="margin-top:12px">
    <img id="preview" style="transform-origin:center center;display:none;" />
  </div>

  <div id="out" style="margin-top:16px"></div>

  <div id="cropOverlay">
    <div id="cropBox">
      <canvas id="cropCanvas" width="1200" height="800"></canvas>
      <div style="font-size:12px;opacity:.8">드래그로 영역 선택 → 완료를 누르면 잘라서 전송합니다.</div>
      <div id="cropActions">
        <button id="cropCancel">취소</button>
        <button id="cropDone" class="pill">완료</button>
      </div>
    </div>
  </div>

<script>
(async function(){
  const verEl = document.getElementById('ver');
  const ver = new URLSearchParams(location.search).get('ver') || '';
  verEl.textContent = ver ? `V: ${ver}` : '(no ver param)';

  const out = document.getElementById('out');
  const status = document.getElementById('status');
  const errorEl = document.getElementById('error');
  const btn = document.getElementById('btn');
  const save = document.getElementById('save');
  const fast = document.getElementById('fast');
  const use_easyocr = document.getElementById('use_easyocr');
  const extra = document.getElementById('extra');
  const fileInput = document.getElementById('file');
  const preview = document.getElementById('preview');
  const rotL = document.getElementById('rotL');
  const rotR = document.getElementById('rotR');
  const cropBtn = document.getElementById('cropBtn');
  const cropOverlay = document.getElementById('cropOverlay');
  const cropCanvas = document.getElementById('cropCanvas');
  const cropCtx = cropCanvas.getContext('2d');
  const cropCancel = document.getElementById('cropCancel');
  const cropDone = document.getElementById('cropDone');

  try{
    const r = await fetch('/settings');
    if (r.ok){
      const s = await r.json();
      fast.checked = !!s.fast_mode; use_easyocr.checked = !!s.use_easyocr;
    } else { fast.checked = false; use_easyocr.checked = true; }
  }catch{ fast.checked = false; use_easyocr.checked = true; }

  let angle = 0, originalBlob = null;
  function setBusy(b){ status.textContent = b ? '업로드/검색 중...' : ''; }

  fileInput.onchange = async () => {
    const f = fileInput.files[0];
    out.innerHTML = ''; errorEl.textContent = '';
    if(!f){ preview.style.display='none'; originalBlob=null; return; }
    const url = URL.createObjectURL(f);
    preview.src = url; preview.style.display = 'block';
    angle = 0; preview.style.transform = `rotate(${angle}deg)`;
    originalBlob = f;
  };

  rotL.onclick = () => { if(!preview.src) return; angle = (angle+270)%360; preview.style.transform = `rotate(${angle}deg)`; };
  rotR.onclick = () => { if(!preview.src) return; angle = (angle+90)%360;  preview.style.transform = `rotate(${angle}deg)`; };

  save.onclick = async () => {
    try{
      await fetch('/settings', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ fast_mode: fast.checked, use_easyocr: use_easyocr.checked })
      });
      alert('설정 저장됨');
    }catch(e){ alert('설정 저장 실패: ' + e.message); }
  };

  let dragStart=null, dragEnd=null, imgBitmap=null, imgW=0, imgH=0, scale=1;

  async function openCropper(){
    if(!originalBlob){ alert('이미지를 선택하세요'); return; }
    document.body.classList.add('cropping');
    cropOverlay.style.display = 'flex';

    imgBitmap = await createImageBitmap(originalBlob);
    imgW = imgBitmap.width; imgH = imgBitmap.height;

    const rotated = (angle % 180 !== 0);
    const finalW = rotated ? imgH : imgW;
    const finalH = rotated ? imgW : imgH;

    const maxW = Math.min(1200, window.innerWidth*0.9);
    const maxH = Math.min(800, window.innerHeight*0.7);
    const s = Math.min(maxW/finalW, maxH/finalH, 1);
    scale = s;

    cropCanvas.width = Math.floor(finalW*s);
    cropCanvas.height= Math.floor(finalH*s);

    drawCropCanvas();
    dragStart = null; dragEnd = null;

    let isDragging = false;
    cropCanvas.onmousedown = (e) => { isDragging = true; dragStart = getPos(e); dragEnd = dragStart; drawCropCanvas(); };
    cropCanvas.onmousemove = (e) => { if (!isDragging) return; dragEnd = getPos(e); drawCropCanvas(); };
    window.addEventListener('mouseup', (e) => { if (!isDragging) return; isDragging = false; dragEnd = getPos(e); drawCropCanvas(); });
  }

  function getPos(e){
    const r = cropCanvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  function drawCropCanvas(){
    cropCtx.fillStyle = '#222';
    cropCtx.fillRect(0,0,cropCanvas.width,cropCanvas.height);

    cropCtx.save();
    if(angle===0){
      cropCtx.drawImage(imgBitmap, 0, 0, imgW, imgH, 0, 0, imgW*scale, imgH*scale);
    }else{
      cropCtx.translate(cropCanvas.width/2, cropCanvas.height/2);
      const rad = angle * Math.PI/180 * -1;
      cropCtx.rotate(rad);
      const drawW = imgW*scale, drawH = imgH*scale;
      cropCtx.drawImage(imgBitmap, -drawW/2, -drawH/2, drawW, drawH);
    }
    cropCtx.restore();

    if(dragStart && dragEnd){
      const a = dragStart, b = dragEnd;
      const x = Math.min(a.x,b.x), y = Math.min(a.y,b.y);
      const w = Math.abs(a.x-b.x), h = Math.abs(a.y-b.y);
      cropCtx.strokeStyle = '#00e0ff'; cropCtx.lineWidth = 2; cropCtx.strokeRect(x,y,w,h);
      cropCtx.fillStyle = 'rgba(0,224,255,0.15)'; cropCtx.fillRect(x,y,w,h);
    }
  }

  cropBtn.onclick = openCropper;
  cropCancel.onclick = () => { cropOverlay.style.display = 'none'; document.body.classList.remove('cropping'); };

  cropDone.onclick = async () => {
    try{
      if(!imgBitmap){ return; }
      const sel = normalizeRect(dragStart, dragEnd, cropCanvas.width, cropCanvas.height);
      if(sel.w < 4 || sel.h < 4){ alert('선택 영역이 너무 작습니다.'); return; }
      const temp = document.createElement('canvas');
      temp.width = sel.w; temp.height = sel.h;
      const tctx = temp.getContext('2d');
      tctx.drawImage(cropCanvas, sel.x, sel.y, sel.w, sel.h, 0, 0, sel.w, sel.h);
      const blob = await new Promise(res => temp.toBlob(res, 'image/png', 0.95));
      if(!blob){ throw new Error('크롭 Blob 생성 실패'); }
      await uploadWithBlob(blob, angle);
    }catch(e){
      errorEl.textContent = '자르기 업로드 오류: ' + (e && e.message ? e.message : e);
    }
  };

  function normalizeRect(a,b,w,h){
    if(!a) return {x:0,y:0,w:0,h:0};
    b = b || a;
    let x = Math.max(0, Math.min(a.x,b.x));
    let y = Math.max(0, Math.min(a.y,b.y));
    let ww = Math.min(w, Math.max(a.x,b.x)) - x;
    let hh = Math.min(h, Math.max(a.y,b.y)) - y;
    return {x:Math.round(x), y:Math.round(y), w:Math.round(ww), h:Math.round(hh)};
  }

  async function uploadWithBlob(blob, usedAngle){
    errorEl.textContent = ''; out.innerHTML = '';
    const fd = new FormData();
    fd.append('file', blob, 'crop.png');
    fd.append('angle', String(usedAngle||0));
    fd.append('fast', fast.checked ? '1' : '0');
    fd.append('use_easyocr_flag', use_easyocr.checked ? '1' : '0');  // ← 이름 통일
    if (extra.value.trim()) fd.append('extra', extra.value.trim());   // ← 추가 검색어

    setBusy(true);
    const res = await fetch('/image-search', { method:'POST', body: fd });
    const j = await res.json().catch(()=>({ok:false,error:'JSON 파싱 실패'}));
    setBusy(false);

    if(!res.ok || !j.ok){
      errorEl.textContent = '오류: ' + (j.error || ('HTTP ' + res.status));
      return;
    }
    renderResult(j);
    cropOverlay.style.display = 'none';
    document.body.classList.remove('cropping');
  }

  function renderResult(j){
    out.insertAdjacentHTML('beforeend',
      `<pre>${JSON.stringify({fast:j.fast, use_easyocr:j.use_easyocr, ocr_text:j.ocr_text, queries:j.queries, meta:j.meta}, null, 2)}</pre>`);
    if (j.videos && j.videos.length){
      j.videos.forEach(v=>{
        out.insertAdjacentHTML('beforeend', 
          `<div class="card">
            <img src="${v.thumb||''}" width="160" height="90" style="object-fit:cover;">
            <div>
              <a href="${v.url}" target="_blank">${v.title||'제목 없음'}</a>
              <div>${v.channel||''}</div>
              <div style="font-size:12px;color:#666">${v.publishedAt||''}</div>
            </div>
          </div>`);
      });
    } else {
      out.insertAdjacentHTML('beforeend','<p>영상 없음(키/쿼터/검색어 확인)</p>');
    }
  }

  btn.onclick = async () => {
    try{
      errorEl.textContent = ''; out.innerHTML = '';
      const f = fileInput.files[0];
      if(!f){ alert('이미지를 선택하세요'); return; }
      const fd = new FormData();
      fd.append('file', f);
      fd.append('angle', angle.toString());
      fd.append('fast', fast.checked ? '1' : '0');
      fd.append('use_easyocr_flag', use_easyocr.checked ? '1' : '0'); // ← 이름 통일
      if (extra.value.trim()) fd.append('extra', extra.value.trim()); // ← 추가 검색어
      setBusy(true);
      const res = await fetch('/image-search', { method:'POST', body: fd });
      const j = await res.json().catch(()=>({ok:false,error:'JSON 파싱 실패'}));
      setBusy(false);
      if(!res.ok || !j.ok){ errorEl.textContent = '오류: ' + (j.error || ('HTTP ' + res.status)); return; }
      renderResult(j);
    }catch(e){
      setBusy(false);
      errorEl.textContent = '네트워크/스크립트 오류: ' + (e && e.message ? e.message : e);
    }
  };
})();
</script>
</body></html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_fastapi:app", host="127.0.0.1", port=8000, reload=True)
