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


def _gate_state(out: dict, now: float) -> dict:
    """Human-readable gate status: ACTIVE (p=..) / STAND-ASIDE (reason) / FALLBACK (no model) / OFF."""
    if not settings.USE_ML_GATE:
        return {"state": "rules", "line": "ML gate: OFF (rule-based gate)"}
    if not out["model"]["present"]:
        return {"state": "fallback", "reason": "missing_model", "line": "ML gate: FALLBACK (no model - rule-based gate)"}
    rec = out.get("recorder")
    if not rec or rec["heartbeat_age_s"] > 180:                       # live, between bars
        age = (rec or {}).get("last_feature_age_s")
        txt = f"stale features, {age:.0f}s old" if age is not None else "stale features, no recorder heartbeat"
        return {"state": "stand_aside", "reason": "stale_features", "line": f"ML gate: STAND-ASIDE ({txt})"}
    per = _read_json(settings.GATE_STATE_PATH) or {}
    if not per:
        return {"state": "active", "line": "ML gate: ACTIVE (waiting for first decision)"}
    sym, g = max(per.items(), key=lambda kv: kv[1].get("ts", 0))
    st, why, p = g.get("gate_state"), (g.get("reason") or "").replace("_", " "), g.get("p")
    if st == "active":
        line = f"ML gate: ACTIVE (p={p:.2f}, {sym} trade)" if p is not None else "ML gate: ACTIVE"
    elif st == "idle":
        line = "ML gate: ACTIVE (no candidate on last bar)"
    elif st == "stand_aside":
        extra = f", {g['feature_age_s']:.0f}s old" if g.get("feature_age_s") is not None else ""
        extra = f", p={p:.2f} < {g.get('threshold') or 0:.2f}" if p is not None else extra
        line = f"ML gate: STAND-ASIDE ({why}{extra})"
    else:
        line = f"ML gate: FALLBACK ({why or 'rule-based'})"
    return {"state": st, "reason": g.get("reason"), "last": g, "symbols": per, "line": line}


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
    out["gate_state"] = _gate_state(out, now)
    mon = _read_json(settings.MONITOR_PATH)
    out["monitor"] = ({"ts": mon.get("ts"), "any_flag": mon.get("any_flag"), "flags": mon.get("flags"),
                       "summary": mon.get("summary")} if mon else None)
    expected = settings.RECORD_FEATURES or settings.USE_ML_GATE
    stale = expected and (not out["recorder"] or out["recorder"]["heartbeat_age_s"] > 180)
    out["status"] = "degraded" if stale else "ok"
    if stale:
        out["problem"] = "feature recorder heartbeat missing/stale"
    return out
