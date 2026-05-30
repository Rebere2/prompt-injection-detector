# ===========================================
# Prompt Injection Detector — Dockerfile
# ===========================================
# Usage:
#   docker build -t prompt-injection-detector .
#   docker run -p 8000:8000 prompt-injection-detector

FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Production stage ----------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY train.py .

# Create models directory
RUN mkdir -p models

# Train the ML model at build time
RUN python train.py

# Create non-root user
RUN adduser --system --no-create-home appuser
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
