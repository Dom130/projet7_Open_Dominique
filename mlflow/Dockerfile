FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && apt-get clean && rm -rf /var/lib/apt/lists/*

# MLflow minimal (léger pour Render plan gratuit)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mlflow==2.9.2 gunicorn==21.2.0

# Copier les runs MLflow trackés
COPY mlruns/ /app/mlruns/

ENV MLFLOW_TRACKING_URI=/app/mlruns
ENV PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# 1 seul worker pour rester dans les limites mémoire du plan gratuit Render
CMD mlflow server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --backend-store-uri /app/mlruns \
    --workers 1
