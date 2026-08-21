FROM python:3.10-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/api/requirements.txt

# Code source
COPY src/ /app/src/
COPY api/ /app/api/
COPY models/ /app/models/

# Variables d'environnement
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
