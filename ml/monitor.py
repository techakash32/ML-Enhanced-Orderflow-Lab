# ml/monitor.py
"""
Phase 7.1 - model-drift and gate-health monitoring.

    python -m ml.monitor                        # one check, writes logs/ml/monitor.json
    python -m ml.monitor --loop --every-min 60  # scheduler (start.sh: ENABLE_MONITOR_LOOP)
    python -m ml.monitor --mlflow               # also tag a run in experiment <exp>_monitoring

What is compared with what (all baselines live in the champion's manifest["training_distribution"],
written by ml.train - nothing is recomputed from training data at monitor time):

  feature drift     PSI of each raw order-flow feature: live warm snapshots from the feature store
                    (last --days) vs the training snapshots. Flag PSI > 0.2 (the usual "significant
                    shift" line; 0.1-0.2 = watch).
  prediction drift  mean of live P(favourable) (both sides, every bar close, from gate_log.jsonl)
                    vs the holdout mean. Flag if |z| > 2, z = shift / (train_std / sqrt(n_eff)),
                    n_eff = number of distinct bar closes (rows from different symbols/sides on the
                    same bar are correlated, so they are not counted as independent).
  firing rate       fraction of live scores with P > threshold vs the holdout fraction.
                    Flag if it deviates by more than 50 % relative.
  label/outcome     realised win rate (net R > be band) and expectancy of ML-gated PAPER trades vs
  drift             the holdout's, with a bootstrap CI. Flag if the holdout win rate lies above the
                    CI (live is significantly worse). "Immature" until >= --min-trades trades closed.

Minimum sample sizes guard every check (no flags from 5 data points). Output schema (read by
scripts/health.py and the dashboard): {ts, champion, any_flag, flags[], summary{...}, features{...},
prediction{...}, firing{...}, outcomes{...}}.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd

from config import settings

log = logging.getLogger("ml.monitor")
PSI_FLAG, PSI_WATCH = 0.2, 0.1
EPS = 1e-4


# --------------------------------------------------------------- baseline ---
def _edges(x: np.ndarray, bins: int = 10) -> list[float]:
    q = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    return [float(v) for v in q]


def _props(x: np.ndarray, edges: list[float]) -> np.ndarray:
    idx = np.searchsorted(np.asarray(edges), x, side="left")
    return np.bincount(idx, minlength=len(edges) + 1) / max(len(x), 1)


def build_training_distribution(snaps: pd.DataFrame, p_holdout: np.ndarray, threshold: float,
                                holdout_eco: dict, be_band_r: float = 0.10) -> dict:
    """Baseline stored in the champion manifest. snaps: WARM training-period snapshots."""
    from features.orderflow import FEATURE_COLUMNS
    feats = {}
    for c in FEATURE_COLUMNS:
        v = snaps[c].to_numpy(float)
        fin = v[np.isfinite(v)]
        if len(fin) < 100 or len(np.unique(fin)) < 3:
            continue
        e = _edges(fin)
        feats[c] = {"edges": e, "props": _props(fin, e).tolist(), "nan_frac": float(1 - len(fin) / len(v)),
                    "n": int(len(fin))}
    p = np.asarray(p_holdout, float)
    return {"features": feats,
            "prediction": {"mean": float(p.mean()), "std": float(p.std()), "n": int(len(p))},
            "gate": {"threshold": float(threshold), "fire_rate": float((p > threshold).mean()),
                     "n": int(len(p))},
            "outcomes": {"win_rate": holdout_eco.get("win_rate"), "expectancy_r": holdout_eco.get("expectancy_r"),
                         "n_taken": holdout_eco.get("n_taken"), "be_band_r": be_band_r}}


# ----------------------------------------------------------------- checks ---
def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e, a = np.clip(np.asarray(expected, float), EPS, None), np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_drift(dist: dict, live: pd.DataFrame, min_n: int = 100, flag: float = PSI_FLAG) -> dict:
    out = {}
    for name, b in dist.get("features", {}).items():
        if name not in live:
            continue
        v = live[name].to_numpy(float)
        fin = v[np.isfinite(v)]
        if len(fin) < min_n:
            out[name] = {"psi": None, "n": int(len(fin)), "flag": False, "note": "insufficient data"}
            continue
        p = psi(np.asarray(b["props"]), _props(fin, b["edges"]))
        out[name] = {"psi": round(p, 4), "n": int(len(fin)), "flag": p > flag,
                     "watch": PSI_WATCH < p <= flag, "nan_frac": round(float(1 - len(fin) / len(v)), 3),
                     "train_nan_frac": round(b.get("nan_frac", 0.0), 3)}
    return out


def prediction_drift(dist: dict, shadow: pd.DataFrame, z_flag: float = 2.0, min_bars: int = 30) -> dict:
    """shadow rows: ts, symbol, p_long, p_short."""
    base = dist.get("prediction", {})
    if shadow.empty or not base:
        return {"flag": False, "note": "no live scores yet"}
    p = np.concatenate([shadow["p_long"].to_numpy(float), shadow["p_short"].to_numpy(float)])
    n_eff = int(shadow["ts"].nunique())
    res = {"live_mean": float(p.mean()), "train_mean": base["mean"], "train_std": base["std"],
           "n_scores": int(len(p)), "n_eff": n_eff, "flag": False}
    if n_eff < min_bars or base["std"] <= 0:
        res["note"] = f"insufficient data ({n_eff} bars < {min_bars})"
        return res
    res["shift_in_train_std"] = float((p.mean() - base["mean"]) / base["std"])
    res["z"] = float((p.mean() - base["mean"]) / (base["std"] / np.sqrt(n_eff)))
    res["flag"] = abs(res["z"]) > z_flag
    return res


def firing_rate(dist: dict, shadow: pd.DataFrame, rel_flag: float = 0.5, min_n: int = 100) -> dict:
    g = dist.get("gate", {})
    if shadow.empty or not g:
        return {"flag": False, "note": "no live scores yet"}
    thr = g["threshold"]
    p = np.concatenate([shadow["p_long"].to_numpy(float), shadow["p_short"].to_numpy(float)])
    live = float((p > thr).mean())
    res = {"live_rate": live, "train_rate": g["fire_rate"], "threshold": thr, "n_scores": int(len(p)), "flag": False}
    if len(p) < min_n:
        res["note"] = f"insufficient data ({len(p)} < {min_n})"
        return res
    if g["fire_rate"] > 0:
        res["relative_change"] = live / g["fire_rate"] - 1.0
        res["flag"] = abs(res["relative_change"]) > rel_flag
    else:
        res["flag"] = live > 0.02                      # model fired on ~nothing in training, now fires
        res["note"] = "train fire rate was 0"
    return res


def outcome_drift(dist: dict, trades: list[dict], min_trades: int = 10, n_boot: int = 2000,
                  seed: int = 0) -> dict:
    """Realised results of ML-gated paper trades vs the holdout. Trades come from trades.jsonl;
    ML-gated = entry_reason carries the 'ML P(target first)' tag written by the Decider."""
    base = dist.get("outcomes", {})
    ml = [t for t in trades if "ML P(target first)" in str(t.get("entry_reason", ""))]
    res = {"n_trades": len(ml), "flag": False}
    if len(ml) < min_trades or not base or base.get("win_rate") is None:
        res["note"] = f"immature ({len(ml)} closed ML-gated trades < {min_trades})"
        return res
    band = base.get("be_band_r", 0.10)
    r = np.array([float(t["r_multiple"]) for t in ml])
    win = (r > band).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), (n_boot, len(r)))
    boots = win[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res.update(realized_win_rate=float(win.mean()), ci=[float(lo), float(hi)], holdout_win_rate=base["win_rate"],
               realized_expectancy_r=float(r.mean()), holdout_expectancy_r=base.get("expectancy_r"))
    res["flag"] = bool(hi < base["win_rate"])           # live significantly worse than holdout
    return res


# ------------------------------------------------------------------ driver ---
def _read_jsonl(path: str, since_ts: float | None = None, ts_key: str = "ts", limit: int = 100000) -> list[dict]:
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if since_ts is None or d.get(ts_key, 0) >= since_ts:
                    rows.append(d)
    except OSError:
        pass
    return rows[-limit:]


def _manifest(path: str) -> dict | None:
    try:
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run_once(days: float = 3.0, last_n_decisions: int = 400, now: float | None = None) -> dict:
    now = now or time.time()
    report = {"ts": now, "champion": None, "any_flag": False, "flags": [], "summary": {}}
    man = _manifest(settings.ML_MODEL_PATH)
    if man is None:
        report["summary"] = {"note": "no champion model - nothing to monitor"}
        return _write(report)
    report["champion"] = man.get("model_version")
    dist = man.get("training_distribution")
    if not dist:
        report["summary"] = {"note": "champion manifest has no training_distribution (trained before Phase 7.1) - retrain"}
        return _write(report)

    from features.store import FeatureStore
    live = FeatureStore(settings.FEATURE_STORE_PATH).read(None, int((now - days * 86400) * 1000), None)
    live = live[live["warm"] == 1.0] if len(live) else live
    shadow = pd.DataFrame(_read_jsonl(settings.GATE_LOG_PATH, None)[-last_n_decisions:])
    trades = _read_jsonl(settings.PAPER_TRADES_LOG, None, "entry_ts")

    f = feature_drift(dist, live)
    pr = prediction_drift(dist, shadow)
    fr = firing_rate(dist, shadow)
    oc = outcome_drift(dist, trades)
    flags = [f"feature_psi:{k}" for k, v in f.items() if v.get("flag")]
    flags += ["prediction_shift"] * pr["flag"] + ["gate_firing_rate"] * fr["flag"] + ["realized_win_rate_low"] * oc["flag"]
    psis = {k: v["psi"] for k, v in f.items() if v.get("psi") is not None}
    top = max(psis, key=psis.get) if psis else None
    report.update(any_flag=bool(flags), flags=flags, features=f, prediction=pr, firing=fr, outcomes=oc,
                  summary={"max_psi": psis.get(top), "max_psi_feature": top, "n_live_snapshots": int(len(live)),
                           "pred_z": pr.get("z"), "fire_rate_live": fr.get("live_rate"),
                           "fire_rate_train": fr.get("train_rate"), "realized_n": oc["n_trades"],
                           "realized_win_rate": oc.get("realized_win_rate"),
                           "holdout_win_rate": oc.get("holdout_win_rate")})
    return _write(report)


def _write(report: dict) -> dict:
    path = settings.MONITOR_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(report, f, default=float)
    os.replace(path + ".tmp", path)
    hist = os.path.join(os.path.dirname(path) or ".", "monitor_history.jsonl")
    with open(hist, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": report["ts"], "champion": report["champion"], "any_flag": report["any_flag"],
                            "flags": report["flags"]}) + "\n")
    return report


def drift_events(since_s: float = 7 * 86400, now: float | None = None) -> dict:
    """Flag counts from monitor_history over the window - read by ml.retrain so a weekly retrain
    can see what the monitor saw."""
    hist = os.path.join(os.path.dirname(settings.MONITOR_PATH) or ".", "monitor_history.jsonl")
    rows = _read_jsonl(hist, (now or time.time()) - since_s)
    counts: dict[str, int] = {}
    for r in rows:
        for fl in r.get("flags", []):
            counts[fl] = counts.get(fl, 0) + 1
    return {"checks": len(rows), "checks_with_flags": sum(1 for r in rows if r.get("any_flag")), "flags": counts}


_last_flagged = {"v": False}


def log_to_mlflow(report: dict) -> None:
    """Tag a run in the '<experiment>_monitoring' experiment when drift appears (or clears),
    so drift is visible next to the training runs in the MLflow UI."""
    if not (report["any_flag"] or _last_flagged["v"]):
        return
    _last_flagged["v"] = report["any_flag"]
    from ml import registry
    mlflow, _ = registry.setup(settings.MLFLOW_TRACKING_URI or None, "orderflow_lgbm_monitoring")
    s = report["summary"]
    with mlflow.start_run(run_name="monitoring-" + time.strftime("%Y%m%dT%H%M%S", time.gmtime(report["ts"]))):
        mlflow.set_tags({"champion": str(report["champion"]), "any_flag": str(report["any_flag"]),
                         "drift_flags": ",".join(report["flags"]) or "none"})
        mlflow.log_metrics({k: float(v) for k, v in s.items()
                            if isinstance(v, (int, float)) and np.isfinite(v)})


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=3.0, help="live feature window for PSI")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--every-min", type=float, default=60.0)
    ap.add_argument("--mlflow", action="store_true", help="tag a monitoring run when drift appears/clears")
    a = ap.parse_args(argv)
    while True:
        try:
            rep = run_once(a.days)
            log.info("monitor: flags=%s summary=%s", rep["flags"] or "none", rep["summary"])
            if a.mlflow:
                log_to_mlflow(rep)
        except Exception:
            log.exception("monitor check failed")
            if not a.loop:
                return 1
        if not a.loop:
            return 0
        time.sleep(a.every_min * 60)


if __name__ == "__main__":
    raise SystemExit(main())
