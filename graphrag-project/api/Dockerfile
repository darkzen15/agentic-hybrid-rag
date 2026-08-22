FROM python:3.11-slim

WORKDIR /app

# curl: healthchecks/debugging; libreoffice-writer: converts legacy .doc ->
# .txt (no pure-Python library reliably parses the old binary format);
# tesseract-ocr + poppler-utils: OCR fallback for scanned/image-only PDF
# pages (pytesseract + pdf2image in requirements.txt need these system
# binaries on PATH - the Python packages alone aren't enough).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    libreoffice-writer \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
