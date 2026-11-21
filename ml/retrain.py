# ml/retrain.py
"""
Weekly champion/challenger retrain (Phase 5).

    python -m ml.retrain                       # one run on the last 90 days
    python -m ml.retrain --lookback-days 60 --threshold 0.58   # extra args go to ml.train
    python -m ml.retrain --loop --every-days 7 # long-running scheduler (container use)

Steps: compact the feature store -> rebuild labels -> train LightGBM (purged CV) ->
evaluate on a time holdout -> log to MLflow -> promote to `champion` ONLY if the
holdout metric is within tolerance of the current champion (see ml.train.decide_promotion).
The live Decider hot-reloads the champion; no restart needed.
Exit codes: 0 ok (promoted or kept champion), 2 not enough data, 1 error.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

from config import settings
from features.store import FeatureStore
from ml import train

log = logging.getLogger("ml.retrain")
HISTORY = os.path.join(os.path.dirname(settings.ML_MODEL_PATH.rstrip("/\\")) or "logs/ml", "retrain_history.jsonl")
LOCK = os.path.join(os.path.dirname(HISTORY), "retrain.lock")


class _Lock:
    """Single-run guard (cron overlap / container restart). Stale after 6h."""

    def __enter__(self):
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
        if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < 6 * 3600:
            raise RuntimeError("another retrain appears to be running (lock present)")
        open(LOCK, "w").write(str(os.getpid()))
        return self

    def __exit__(self, *a):
        try:
            os.remove(LOCK)
        except OSError:
            pass


def once(argv: list[str]) -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--lookback-days", type=float, default=float(os.environ.get("RETRAIN_LOOKBACK_DAYS", "90")))
    known, rest = p.parse_known_args(argv)
    extra = os.environ.get("RETRAIN_ARGS", "").split()
    a = train.parse_args(["--lookback-days", str(known.lookback_days), *extra, *rest])
    started = datetime.now(timezone.utc).isoformat()
    try:
        with _Lock():
            removed = FeatureStore(a.store).compact()
            log.info("feature store compacted (%d part files merged)", removed)
            res = train.run(a)
        status, code = "ok", 0
    except RuntimeError as exc:                     # e.g. not enough data yet
        log.error("retrain skipped: %s", exc)
        res, status, code = {"error": str(exc)}, "skipped", 2
    except Exception as exc:
        log.exception("retrain failed")
        res, status, code = {"error": repr(exc)}, "failed", 1
    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps({"started": started, "status": status,
                            **{k: res.get(k) for k in ("run_name", "promoted", "reason", "registry_version",
                                                       "holdout", "champion_holdout", "error")}},
                           default=float) + "\n")
    return code


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--loop" in argv:
        argv.remove("--loop")
        days = 7.0
        if "--every-days" in argv:
            i = argv.index("--every-days")
            days = float(argv[i + 1])
            del argv[i:i + 2]
        log.info("retrain loop: every %.1f days", days)
        while True:
            time.sleep(60)                                  # let the recorder warm up first
            once(argv)
            time.sleep(days * 86400)
    return once(argv)


if __name__ == "__main__":
    sys.exit(main())
