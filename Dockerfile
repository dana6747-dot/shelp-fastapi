FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Tesseract(한글) + OpenCV 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-kor libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render/Railway 등이 주는 PORT 사용(없으면 8000)
ENV PORT=8000
CMD ["bash","-lc","uvicorn app_fastapi:app --host 0.0.0.0 --port ${PORT}"]
