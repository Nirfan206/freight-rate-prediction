# Build from the repository root:
#   docker build -f docker/Dockerfile -t freight-rate-api .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src \
    MODEL_REGISTRY_PATH=/app/models/registry \
    MONITORING_PATH=/app/monitoring

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Runtime code + the promoted model registry. Training data is NOT needed to serve.
COPY src ./src
COPY api ./api
COPY scripts ./scripts
COPY models ./models

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/monitoring/predictions \
    && chown -R appuser:appuser /app/monitoring
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
