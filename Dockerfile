FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Tesseract(한글) + OpenCV용 libgl
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-kor libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render가 주는 포트로 실행해야 502가 안 남
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV PORT=10000
EXPOSE 10000

CMD ["sh","-c","uvicorn app_fastapi:app --host 0.0.0.0 --port ${PORT}"]
