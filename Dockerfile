# Lean serving image: python:3.11-slim + serving requirements only (no MLflow / pytest / pandas).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PREDICTION_LOG=/app/logs/predictions.jsonl

# libgomp1 is the OpenMP runtime that XGBoost needs on Debian slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app.py features.py ./
COPY model/model.joblib model/model.joblib
COPY static/ static/

RUN mkdir -p /app/logs
VOLUME ["/app/logs"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
