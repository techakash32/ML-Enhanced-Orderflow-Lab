#!/usr/bin/env bash
# Weekly champion/challenger retrain. Safe to run from cron; overlapping runs are skipped.
#
#   crontab -e   (Sundays 03:15 UTC)
#   15 3 * * 0  /path/to/repo/ml/retrain.sh
#
# Extra arguments are passed to `python -m ml.retrain` / `ml.train`, e.g.
#   ml/retrain.sh --lookback-days 60 --threshold 0.58
# Windows: see ML_README.md (Task Scheduler running `python -m ml.retrain`).
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/ml
exec 9>logs/ml/retrain.flock
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
  echo "$(date -u +%FT%TZ) retrain already running, skipping" >> logs/ml/retrain.log
  exit 0
fi
PY="${PYTHON:-python}"
echo "$(date -u +%FT%TZ) retrain start" >> logs/ml/retrain.log
"$PY" -m ml.retrain "$@" >> logs/ml/retrain.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) retrain exit=$rc" >> logs/ml/retrain.log
exit $rc
