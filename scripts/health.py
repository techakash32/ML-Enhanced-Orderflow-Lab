# scripts/health.py - cheap, dependency-free health report (read by the dashboard /health route).
# The dashboard is a separate process from paper_live, so everything is read from files the
# other processes write: recorder heartbeat, trades log, champion manifest, retrain history.
import json
import os
import time

from config import settings

_START = time.time()


def _last_json_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            lines = f.read().decode("utf-8", "ignore").strip().splitlines()
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def health(now=None) -> dict:
    now = now or time.time()
    out = {"uptime_s": int(now - _START), "interval": settings.INTERVAL, "symbols": settings.SYMBOLS,
           "ml_gate_enabled": settings.USE_ML_GATE}
    # recorder
    hb = _read_json(settings.FEATURE_HEARTBEAT_PATH)
    if hb:
        last = [v for v in (hb.get("last_snapshot_ms") or {}).values() if v]
        out["recorder"] = {
            "heartbeat_age_s": round(now - hb["ts"], 1),
            "last_feature_ts_ms": max(last) if last else None,
            "last_feature_age_s": round(now - max(last) / 1000, 1) if last else None,
            "trades_seen": hb.get("stats", {}).get("trades"), "books_seen": hb.get("stats", {}).get("books"),
            "reconnects": hb.get("stats", {}).get("reconnects"),
            "feed_resets": sum((hb.get("resets") or {}).values()),
        }
    else:
        out["recorder"] = None
    # last paper trade
    t = _last_json_line(settings.PAPER_TRADES_LOG)
    out["last_trade"] = ({"symbol": t.get("symbol"), "exit_ts_ms": t.get("exit_ts"),
                          "entry_ts_ms": t.get("entry_ts"), "reason": t.get("exit_reason"),
                          "r": t.get("r_multiple")} if t else None)
    # model
    man = _read_json(os.path.join(settings.ML_MODEL_PATH, "manifest.json"))
    out["model"] = ({"present": True, "champion_version": man.get("model_version"),
                     "trained_utc": man.get("created_utc"), "exit_mode": man.get("exit_mode"),
                     "exit_parity_with_settings": man.get("exit_parity_with_settings"),
                     "holdout_edge": (man.get("holdout", {}).get("economic", {}) or {}).get("edge")}
                    if man else {"present": False, "note": "no champion -> rule-based gate"})
    rt = _last_json_line(os.path.join(os.path.dirname(settings.ML_MODEL_PATH.rstrip("/\\")), "retrain_history.jsonl"))
    out["last_retrain"] = ({"started": rt.get("started"), "status": rt.get("status"),
                            "promoted": rt.get("promoted"), "reason": rt.get("reason")} if rt else None)
    expected = settings.RECORD_FEATURES or settings.USE_ML_GATE
    stale = expected and (not out["recorder"] or out["recorder"]["heartbeat_age_s"] > 180)
    out["status"] = "degraded" if stale else "ok"
    if stale:
        out["problem"] = "feature recorder heartbeat missing/stale"
    return out
