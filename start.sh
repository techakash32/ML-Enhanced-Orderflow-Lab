#!/bin/bash
# Runs the paper-live engine (with the order-flow feature recorder + ML gate) and the
# dashboard in one container. If either CORE process dies, the container exits so the
# platform restarts it. Optional helpers (MLflow UI, weekly retrain) never take it down.
#
#   MLFLOW_SERVER=true         start the MLflow UI on $MLFLOW_PORT (default 5000)
#   ENABLE_RETRAIN_LOOP=true   run `python -m ml.retrain --loop` every $RETRAIN_EVERY_DAYS (default 7)

python -m scripts.paper_live &
ENGINE_PID=$!

python -m uvicorn scripts.dashboard:app --host 0.0.0.0 --port "${PORT:-8000}" &
WEB_PID=$!

OPTIONAL_PIDS=""
if [ "${MLFLOW_SERVER:-false}" = "true" ]; then
  mkdir -p logs/ml/mlruns "${MLFLOW_ARTIFACT_ROOT:-logs/ml/mlruns/artifacts}"
  python -m mlflow server \
    --backend-store-uri "${MLFLOW_TRACKING_URI:-sqlite:///logs/ml/mlruns/mlflow.db}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT:-logs/ml/mlruns/artifacts}" \
    --host 0.0.0.0 --port "${MLFLOW_PORT:-5000}" &
  OPTIONAL_PIDS="$OPTIONAL_PIDS $!"
fi
if [ "${ENABLE_RETRAIN_LOOP:-false}" = "true" ]; then
  python -m ml.retrain --loop --every-days "${RETRAIN_EVERY_DAYS:-7}" &
  OPTIONAL_PIDS="$OPTIONAL_PIDS $!"
fi

trap 'kill $ENGINE_PID $WEB_PID $OPTIONAL_PIDS 2>/dev/null' TERM INT

wait -n $ENGINE_PID $WEB_PID
STATUS=$?
kill $ENGINE_PID $WEB_PID $OPTIONAL_PIDS 2>/dev/null
exit $STATUS
