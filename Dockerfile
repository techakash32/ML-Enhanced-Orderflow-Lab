FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# ML entry gate. All ML state lives under /app/logs so the existing volume persists it.
# With no trained model present the gate automatically falls back to the rule-based gate
# (set USE_ML_GATE=false to force the original rules).
ENV USE_ML_GATE=true \
    ML_MODEL_PATH=/app/logs/ml/models/champion \
    ML_GATE_THRESHOLD=0.55 \
    FEATURE_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT \
    FEATURE_STORE_PATH=/app/logs/feature_store \
    MLFLOW_TRACKING_URI=sqlite:////app/logs/ml/mlruns/mlflow.db

WORKDIR /app

# Install dependencies first so this layer is cached between code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code incl. features/ and ml/ (see .dockerignore)
COPY . .
RUN chmod +x start.sh

# Run as non-root; logs/ holds paper-trading state + the dashboard's data
RUN useradd --create-home appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

# Persist paper-trading state across restarts by mounting a volume here
VOLUME ["/app/logs"]

EXPOSE 8000
# 5000 = optional MLflow UI (MLFLOW_SERVER=true); Render only exposes $PORT, so it is off by default
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/dashboard' % os.environ.get('PORT','8000'), timeout=4)" || exit 1

# Starts paper-live (+ feature recorder) and the dashboard; optional MLflow UI / weekly retrain loop
CMD ["./start.sh"]