import numpy as np
import pytest

from config import settings
from engine.decision import Decider, reset_sweep_diagnostics, ml_diagnostics_snapshot
from features.orderflow import FEATURE_COLUMNS, MODEL_FEATURES
from gate.buckets import Dir
from ml.gate import MLGate
from ml.train import decide_promotion

H4 = 4 * 3600 * 1000


def make_hist(direction="LONG", n=60):
    """Flat 4h bars then one Donchian breakout whose ENTRY (next open) is 12:00 UTC (London)."""
    last_open = 1_700_000_000_000 // (24 * H4 // 6 * 6) * (24 * H4 // 6 * 6)
    day = 86_400_000
    last_open = (1_700_000_000_000 // day) * day + 8 * 3600 * 1000       # 08:00 UTC
    bars = []
    for i in range(n):
        ot = last_open - (n - 1 - i) * H4
        bars.append({"open_time": ot, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                     "buy_vol": 1.0, "sell_vol": 1.0})
    b = bars[-1]
    if direction == "LONG":
        b.update(high=106.0, close=105.0)
    else:
        b.update(low=94.0, close=95.0)
    return bars


def feats(warm=1.0):
    f = {c: 0.1 for c in FEATURE_COLUMNS}
    f["warm"] = warm
    return f


class Stub:
    available = True

    def __init__(self, p):
        self.p, self.calls = p, []

    def maybe_reload(self):
        pass

    def p_favorable(self, f, side, atr_pct):
        self.calls.append((side, atr_pct))
        return self.p


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(settings, "ACTIVE_STRATEGY", "trend")
    monkeypatch.setattr(settings, "BUCKET_B_ENABLED", False)
    monkeypatch.setattr(settings, "TREND_LONDON_ONLY", True)
    monkeypatch.setattr(settings, "ML_GATE_THRESHOLD", 0.55)
    monkeypatch.setattr(settings, "ML_ON_MISSING_FEATURES", "reject")
    reset_sweep_diagnostics()


def test_rules_mode_unchanged():
    d = Decider(use_ml=False).decide(make_hist(), 105.0)
    assert d.allowed and d.direction == Dir.LONG and d.gate_mode == "rules" and d.ml_p is None


def test_ml_passes_above_threshold_and_fails_below():
    hist = make_hist()
    ok = Decider(use_ml=True, ml_gate=Stub(0.70)).decide(hist, 105.0, features=feats())
    assert ok.allowed and ok.direction == Dir.LONG and ok.gate_mode == "ml" and ok.ml_p == 0.70
    no = Decider(use_ml=True, ml_gate=Stub(0.50)).decide(hist, 105.0, features=feats())
    assert not no.allowed and no.gate_mode == "ml" and "ML P(target first)" in no.reason
    edge = Decider(use_ml=True, ml_gate=Stub(0.55)).decide(hist, 105.0, features=feats())
    assert not edge.allowed                       # strictly greater than threshold


def test_short_side_and_atr_pct_passed_to_model():
    stub = Stub(0.9)
    d = Decider(use_ml=True, ml_gate=stub).decide(make_hist("SHORT"), 95.0, features=feats())
    assert d.allowed and d.direction == Dir.SHORT
    side, atr_pct = stub.calls[0]
    expected = (13 * 2.0 + 7.0) / 14 / 95.0 * 100        # PaperTrader-style mean(high-low)/close
    assert side == -1 and abs(atr_pct - expected) < 1e-9


def test_no_signal_means_ml_never_consulted():
    hist = make_hist()
    hist[-1].update(high=101.0, close=100.0)       # no breakout
    stub = Stub(0.99)
    d = Decider(use_ml=True, ml_gate=stub).decide(hist, 100.0, features=feats())
    assert not d.allowed and stub.calls == []


def test_missing_or_cold_features_fail_closed_then_fallback_option(monkeypatch):
    hist = make_hist()
    d = Decider(use_ml=True, ml_gate=Stub(0.99)).decide(hist, 105.0, features=None)
    assert not d.allowed and "no warm order-flow features" in d.reason
    d = Decider(use_ml=True, ml_gate=Stub(0.99)).decide(hist, 105.0, features=feats(warm=0.0))
    assert not d.allowed
    assert ml_diagnostics_snapshot()["ml_no_features"] == 2
    monkeypatch.setattr(settings, "ML_ON_MISSING_FEATURES", "fallback")
    d = Decider(use_ml=True, ml_gate=Stub(0.0)).decide(hist, 105.0, features=None)
    assert d.allowed and d.gate_mode == "rules"


def test_warmup_replay_uses_rules():
    d = Decider(use_ml=True, ml_gate=Stub(0.0)).decide(make_hist(), 105.0, warmup_replay=True)
    assert d.allowed and d.gate_mode == "rules"


def test_missing_model_artifact_falls_back_to_rules(tmp_path):
    gate = MLGate(path=str(tmp_path / "nope"), source="local", tracking_uri="")
    assert not gate.available
    d = Decider(use_ml=True, ml_gate=gate).decide(make_hist(), 105.0, features=feats())
    assert d.allowed and d.gate_mode == "rules"


def test_env_flag_false_is_rules(monkeypatch):
    monkeypatch.setattr(settings, "USE_ML_GATE", False)
    d = Decider().decide(make_hist(), 105.0)
    assert d.allowed and d.gate_mode == "rules"


def test_real_lightgbm_bundle_roundtrip(tmp_path):
    import lightgbm as lgb
    from ml.bundle import save_bundle
    rng = np.random.default_rng(0)
    import pandas as pd
    X = pd.DataFrame(rng.normal(size=(600, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    y = (X["ofi_snap_l5"] > 0).astype(int)
    m = lgb.train({"objective": "binary", "verbosity": -1}, lgb.Dataset(X, y), 20)
    save_bundle(m, {"model_version": "t", "feature_names": MODEL_FEATURES, "objective": "binary",
                    "favorable_index": None, "created_utc": "now"}, str(tmp_path / "champ"))
    gate = MLGate(path=str(tmp_path / "champ"), source="local", tracking_uri="")
    assert gate.available
    f = feats(); f["ofi_snap_l5"] = 2.0
    hi_long = gate.p_favorable(f, +1, 1.0)
    hi_short = gate.p_favorable(f, -1, 1.0)      # sign flip => ofi_snap_l5 = -2 => lower P
    assert hi_long > 0.8 and hi_short < 0.2


def test_promotion_rules():
    good = {"n_taken": 100, "edge": 0.05, "expectancy_r": 0.1}
    assert decide_promotion(good, None, "edge", 0.01, 0.0, 30)[0]
    assert not decide_promotion({**good, "edge": -0.02}, None, "edge", 0.01, 0.0, 30)[0]     # floor
    assert not decide_promotion({**good, "n_taken": 5}, None, "edge", 0.01, 0.0, 30)[0]
    champ = {"n_taken": 80, "edge": 0.10, "expectancy_r": 0.2}
    assert not decide_promotion(good, champ, "edge", 0.01, 0.0, 30)[0]                       # worse by .05
    assert decide_promotion({**good, "edge": 0.095}, champ, "edge", 0.01, 0.0, 30)[0]        # within tol
    assert decide_promotion({**good, "edge": 0.0}, champ, "edge", 0.01, 0.0, 30, force=True)[0]
