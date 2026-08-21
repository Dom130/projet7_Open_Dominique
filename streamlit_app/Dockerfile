FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY streamlit_app/requirements.txt /app/streamlit_app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/streamlit_app/requirements.txt

COPY streamlit_app/ /app/streamlit_app/
COPY reports/ /app/reports/
COPY src/ /app/src/

ENV PYTHONPATH=/app
ENV API_URL=http://localhost:8000
ENV MLFLOW_URL=http://localhost:5002
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
