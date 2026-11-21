"""Phase 7.2: every stand-aside path emits an explicit, structured reason."""
import time

import pytest

from config import settings
from engine.decision import Decider
from ml.gate import MLGate
from tests.test_decision_ml import Stub, feats, make_hist


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(settings, "ACTIVE_STRATEGY", "trend")
    monkeypatch.setattr(settings, "BUCKET_B_ENABLED", False)
    monkeypatch.setattr(settings, "TREND_LONDON_ONLY", True)
    monkeypatch.setattr(settings, "ML_GATE_THRESHOLD", 0.55)
    monkeypatch.setattr(settings, "ML_ON_MISSING_FEATURES", "reject")
    monkeypatch.setattr(settings, "ML_MAX_FEATURE_AGE_S", 180)


def fresh(**kw):
    f = feats(); f["ts_ms"] = time.time() * 1000 - 5000
    f.update(kw)
    return f


def dec(p=0.9, features="fresh", hist=None):
    d = Decider(use_ml=True, ml_gate=Stub(p))
    f = fresh() if features == "fresh" else features
    return d.decide(hist or make_hist(), 105.0, features=f)


def test_active():
    d = dec(0.7)
    assert d.allowed and d.gate_state == "active" and d.gate_reason == "" and 4 < d.feature_age_s < 10


def test_below_threshold():
    d = dec(0.3)
    assert not d.allowed and (d.gate_state, d.gate_reason) == ("stand_aside", "below_threshold") and d.ml_p == 0.3


def test_stale_features_missing_snapshot():
    d = dec(features=None)
    assert (d.gate_state, d.gate_reason) == ("stand_aside", "stale_features") and d.feature_age_s is None


def test_stale_features_by_age():
    d = dec(features=fresh(ts_ms=time.time() * 1000 - 47_000 - 200_000))
    assert d.gate_reason == "stale_features" and d.feature_age_s > 240 and "old" in d.reason


def test_feature_nan_when_not_warm_or_mostly_nan():
    assert dec(features=fresh(warm=0.0)).gate_reason == "feature_nan"
    nan = fresh()
    for k in list(nan)[:25]:
        if k not in ("ts_ms", "warm"):
            nan[k] = float("nan")
    assert dec(features=nan).gate_reason == "feature_nan"


def test_missing_model_is_fallback_to_rules(tmp_path):
    gate = MLGate(path=str(tmp_path / "none"), source="local", tracking_uri="")
    d = Decider(use_ml=True, ml_gate=gate).decide(make_hist(), 105.0, features=fresh())
    assert d.allowed and d.gate_mode == "rules" and (d.gate_state, d.gate_reason) == ("fallback", "missing_model")


def test_fallback_option_for_missing_features(monkeypatch):
    monkeypatch.setattr(settings, "ML_ON_MISSING_FEATURES", "fallback")
    d = dec(features=None)
    assert d.allowed and d.gate_mode == "rules" and d.gate_state == "fallback" and d.gate_reason == "stale_features"


def test_idle_when_no_candidate_and_rules_when_disabled():
    hist = make_hist(); hist[-1].update(high=101.0, close=100.0)
    assert Decider(use_ml=True, ml_gate=Stub(0.9)).decide(hist, 100.0, features=fresh()).gate_state == "idle"
    assert Decider(use_ml=False).decide(make_hist(), 105.0).gate_state == "rules"
