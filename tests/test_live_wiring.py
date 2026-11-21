"""Binance-format frames -> FeatureRecorder -> store -> LiveTrendRunner -> ML gate decision.
No network: frames are synthesised in the documented combined-stream shapes."""
import asyncio
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import settings
from features.orderflow import MODEL_FEATURES, OrderFlowConfig
from features.recorder import FeatureRecorder
from features.store import FeatureStore
from ml.bundle import save_bundle
from ml.gate import MLGate

DT = 60_000


def trade_frame(sym, ts, p, q, maker):
    return json.dumps({"stream": f"{sym.lower()}@aggTrade", "data": {
        "e": "aggTrade", "s": sym, "T": ts, "p": str(p), "q": str(q), "m": maker, "a": ts}})


def depth_frame(sym, ts, p):
    return json.dumps({"stream": f"{sym.lower()}@depth5@100ms", "data": {
        "e": "depthUpdate", "s": sym, "T": ts, "E": ts,
        "b": [[str(p - 0.5 * (i + 1)), "3.0"] for i in range(5)],
        "a": [[str(p + 0.5 * (i + 1)), "1.0"] for i in range(5)]}})


def feed(rec, sym, t_start, minutes, rng):
    t, p = t_start, 100.0
    for i in range(minutes * 30):
        t += 2000
        rec.handle(trade_frame(sym, t, round(p, 2), round(float(rng.lognormal(-2, 1)), 4), bool(rng.random() > 0.6)))
        rec.handle(depth_frame(sym, t + 5, p))
    return t


def test_recorder_parses_frames_persists_and_serves_asof(tmp_path):
    rec = FeatureRecorder(["BTCUSDT"], FeatureStore(str(tmp_path)),
                          OrderFlowConfig(snapshot_s=60, vpin_bucket_volume=5.0, z_min=5,
                                          warm_min_snapshots=5, vpin_window=5))
    t0 = 1_700_000_040_000 // DT * DT
    rng = np.random.default_rng(0)
    t_end = feed(rec, "BTCUSDT", t0, 40, rng)
    assert rec.stats["trades"] > 1000 and rec.stats["books"] > 1000
    boundary = (t_end // DT) * DT
    row = rec.features_asof("BTCUSDT", boundary)
    assert row is not None and row["ts_ms"] == boundary and row["warm"] == 1.0
    assert row["qi_l1"] > 0.4                                  # bids 3.0 vs asks 1.0 => +0.5
    assert rec.features_asof("BTCUSDT", boundary + 2 * DT)["ts_ms"] == boundary + 2 * DT    # forces emit
    # a long feed outage resets state and yields NO snapshot -> the ML gate would stand aside
    assert rec.features_asof("BTCUSDT", boundary + 60 * DT) is None
    rec.store.flush()
    df = rec.store.read(["BTCUSDT"])
    assert len(df) >= 30 and df["ts_ms"].is_monotonic_increasing and df["ts_ms"].is_unique


def test_runner_uses_ml_gate_with_recorder_features(tmp_path, monkeypatch):
    from scripts import paper_live
    monkeypatch.setattr(paper_live, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(paper_live, "SIGNALS_LOG", str(tmp_path / "signals.jsonl"))
    monkeypatch.setattr(settings, "ACTIVE_STRATEGY", "trend")
    monkeypatch.setattr(settings, "TREND_LONDON_ONLY", False)
    monkeypatch.setattr(settings, "USE_ML_GATE", True)
    monkeypatch.setattr(settings, "ML_GATE_THRESHOLD", 0.5)

    # always-bullish tiny model (predicts P=high for every row)
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    m = lgb.train({"objective": "binary", "verbosity": -1}, lgb.Dataset(X, np.ones(300)), 5,)
    save_bundle(m, {"model_version": "t", "feature_names": MODEL_FEATURES, "objective": "binary",
                    "favorable_index": None, "created_utc": "now"}, str(tmp_path / "champ"))
    gate = MLGate(path=str(tmp_path / "champ"), source="local", tracking_uri="")

    H4 = 4 * 3600 * 1000
    rec = FeatureRecorder(["BTCUSDT"], FeatureStore(str(tmp_path / "fs")),
                          OrderFlowConfig(snapshot_s=60, vpin_bucket_volume=5.0, z_min=5,
                                          warm_min_snapshots=5, vpin_window=5))
    bar_open = (1_700_000_000_000 // H4) * H4 + H4            # a 4h boundary
    runner = paper_live.LiveTrendRunner("BTCUSDT", rec)
    runner.decider.use_ml, runner.decider._ml = True, gate
    for i in range(60, 0, -1):                                 # flat history
        runner.history.append({"symbol": "BTCUSDT", "interval": "4h", "open_time": bar_open - i * H4,
                               "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                               "volume": 1.0, "trades": 1, "quote_volume": 1.0})
    candle = {"open_time": bar_open, "open": 100.0, "high": 107.0, "low": 99.5, "close": 106.0,
              "volume": 5.0, "trades": 10, "quote_volume": 5.0}
    # continuous feed straddling the decision boundary (bar close = bar_open + 4h)
    feed(rec, "BTCUSDT", bar_open + H4 - 40 * DT, 41, np.random.default_rng(3))
    asyncio.run(runner.on_candle(candle))
    sig = [json.loads(l) for l in open(paper_live.SIGNALS_LOG)]
    assert sig and sig[-1]["gate"] == "ml" and sig[-1]["ml_p"] > 0.5 and sig[-1]["direction"] == "LONG"
    assert runner.trader.pending is not None                   # PaperTrader queued next-open entry
