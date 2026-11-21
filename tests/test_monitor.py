"""Phase 7.1: drift monitor flags real shifts and stays quiet on stable streams."""
import json
import time

import numpy as np
import pandas as pd
import pytest

from config import settings
from features.orderflow import FEATURE_COLUMNS
from ml import monitor


def snaps(n=4000, shift=0.0, seed=0, cols=("ofi_snap_l5", "vpin", "spread_bps")):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS})
    for c in cols:
        df[c] = df[c] + shift
    df["warm"] = 1.0
    return df


def dist(p_mean=0.40, thr=0.55, win_rate=0.50):
    base = monitor.build_training_distribution(
        snaps(seed=1), np.random.default_rng(2).normal(p_mean, 0.08, 3000), thr,
        {"win_rate": win_rate, "expectancy_r": 0.2, "n_taken": 80})
    return base


def test_psi_flags_shift_not_stable():
    d = dist()
    stable = monitor.feature_drift(d, snaps(2000, 0.0, seed=5))
    assert not any(v["flag"] for v in stable.values()) and max(v["psi"] for v in stable.values()) < 0.1
    shifted = monitor.feature_drift(d, snaps(2000, 1.5, seed=6))          # 1.5-sigma shift on 3 features
    flagged = {k for k, v in shifted.items() if v["flag"]}
    assert flagged == {"ofi_snap_l5", "vpin", "spread_bps"}
    assert monitor.feature_drift(d, snaps(50))["vpin"]["flag"] is False   # too little data -> no flag


def shadow(mean, n_bars=100, sd=0.08, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"ts": np.repeat(np.arange(n_bars), 2), "symbol": ["A", "B"] * n_bars,
                         "p_long": rng.normal(mean, sd, 2 * n_bars), "p_short": rng.normal(mean, sd, 2 * n_bars)})


def test_prediction_and_firing_drift():
    d = dist()
    ok = monitor.prediction_drift(d, shadow(0.40))
    assert not ok["flag"]
    bad = monitor.prediction_drift(d, shadow(0.46))                        # +0.75 train-std shift, n_eff=100
    assert bad["flag"] and bad["z"] > 2
    assert not monitor.prediction_drift(d, shadow(0.46, n_bars=10))["flag"]   # insufficient bars
    train_rate = d["gate"]["fire_rate"]
    assert train_rate > 0
    assert not monitor.firing_rate(d, shadow(0.40, seed=3))["flag"]
    assert monitor.firing_rate(d, shadow(0.55, seed=4))["flag"]            # fires far more often


def trade(r, ml=True):
    return {"r_multiple": r, "entry_reason": "TRADE LONG - A | ML P(target first)=0.61" if ml else "rules"}


def test_outcome_drift():
    d = dist(win_rate=0.55)
    rng = np.random.default_rng(0)
    immature = monitor.outcome_drift(d, [trade(1.0)] * 5)
    assert not immature["flag"] and "immature" in immature["note"]
    good = [trade(1.0 if rng.random() < 0.55 else -1.0) for _ in range(60)]
    assert not monitor.outcome_drift(d, good)["flag"]
    bad = [trade(1.0 if rng.random() < 0.15 else -1.0) for _ in range(60)]
    o = monitor.outcome_drift(d, bad)
    assert o["flag"] and o["realized_win_rate"] < 0.3
    assert monitor.outcome_drift(d, [trade(-1.0, ml=False)] * 60)["n_trades"] == 0   # rule-based trades ignored


@pytest.fixture
def env(tmp_path, monkeypatch):
    from features.store import FeatureStore
    champ = tmp_path / "models" / "champion"
    champ.mkdir(parents=True)
    (champ / "manifest.json").write_text(json.dumps({"model_version": "m1", "training_distribution": dist()}))
    monkeypatch.setattr(settings, "ML_MODEL_PATH", str(champ))
    monkeypatch.setattr(settings, "MONITOR_PATH", str(tmp_path / "ml" / "monitor.json"))
    monkeypatch.setattr(settings, "GATE_LOG_PATH", str(tmp_path / "ml" / "gate_log.jsonl"))
    monkeypatch.setattr(settings, "PAPER_TRADES_LOG", str(tmp_path / "trades.jsonl"))
    monkeypatch.setattr(settings, "FEATURE_STORE_PATH", str(tmp_path / "fs"))
    monkeypatch.setattr(settings, "FEATURE_HEARTBEAT_PATH", str(tmp_path / "hb.json"))
    return tmp_path, FeatureStore(str(tmp_path / "fs"))


def fill_store(store, shift, now):
    df = snaps(1500, shift, seed=9)
    df["symbol"], df["ts_ms"] = "BTCUSDT", (now * 1000 - np.arange(1500)[::-1] * 60_000).astype("int64")
    for c in ("open", "high", "low", "close", "mid", "volume", "buy_vol", "sell_vol", "n_trades"):
        df[c] = 1.0
    for r in df.to_dict("records"):
        store.append(r)
    store.flush()


def test_run_once_flags_and_health_roundtrip(env, monkeypatch):
    tmp, store = env
    now = time.time()
    fill_store(store, shift=1.5, now=now)
    (tmp / "ml").mkdir(exist_ok=True)
    with open(settings.GATE_LOG_PATH, "w") as f:
        for i in range(120):
            f.write(json.dumps({"ts": i, "symbol": "BTCUSDT", "p_long": 0.40, "p_short": 0.41}) + "\n")
    rep = monitor.run_once(days=3, now=now)
    assert rep["any_flag"] and "feature_psi:vpin" in rep["flags"] and rep["champion"] == "m1"
    assert "prediction_shift" not in rep["flags"]
    from scripts.health import health
    h = health()
    assert h["monitor"]["any_flag"] and "feature_psi:vpin" in h["monitor"]["flags"]
    assert monitor.drift_events()["checks_with_flags"] >= 1


def test_no_champion_or_legacy_manifest_is_graceful(env):
    tmp, _ = env
    (tmp / "models" / "champion" / "manifest.json").write_text(json.dumps({"model_version": "old"}))
    rep = monitor.run_once()
    assert not rep["any_flag"] and "retrain" in rep["summary"]["note"]
    (tmp / "models" / "champion" / "manifest.json").unlink()
    assert "no champion" in monitor.run_once()["summary"]["note"]
