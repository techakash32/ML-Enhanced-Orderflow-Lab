# ml/telemetry.py - tiny file-based telemetry the dashboard/monitor read (separate processes).
import json
import os
import time

from config import settings


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def write_gate_state(symbol: str, decision, bar_ts: int) -> None:
    """Latest gate state per symbol -> settings.GATE_STATE_PATH (read by /health)."""
    try:
        with open(settings.GATE_STATE_PATH, encoding="utf-8") as f:
            cur = json.load(f)
    except (OSError, ValueError):
        cur = {}
    cur[symbol] = {"ts": time.time(), "bar_ts": bar_ts, "gate_state": decision.gate_state,
                   "reason": decision.gate_reason, "feature_age_s": decision.feature_age_s,
                   "p": decision.ml_p, "threshold": decision.ml_threshold,
                   "allowed": bool(decision.allowed), "direction": decision.direction.value
                   if decision.direction is not None else None}
    _atomic_json(settings.GATE_STATE_PATH, cur)


def log_shadow(symbol: str, bar_ts: int, scores: dict, decision) -> None:
    """One line per bar close: model P for BOTH sides (whether or not Bucket A fired).
    This is the stream ml.monitor uses for prediction drift and gate-firing rate."""
    os.makedirs(os.path.dirname(settings.GATE_LOG_PATH) or ".", exist_ok=True)
    with open(settings.GATE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": bar_ts, "symbol": symbol, **scores, "thr": settings.ML_GATE_THRESHOLD,
                            "state": decision.gate_state, "allowed": bool(decision.allowed)}) + "\n")
