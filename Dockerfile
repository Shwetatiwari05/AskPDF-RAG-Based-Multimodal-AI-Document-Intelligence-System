FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

# libs needed by opencv / easyocr / pymupdf at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first so Docker caches this layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Cloud Run injects $PORT (default 8080)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}