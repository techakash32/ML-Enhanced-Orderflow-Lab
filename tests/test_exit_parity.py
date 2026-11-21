"""Label/exit parity: the labeler's trade simulator must equal the REAL PaperTrader."""
import numpy as np
import pytest

from config import settings
from execution.paper_trader import PaperTrader
from gate.buckets import Dir
from ml.exit_sim import ExitRules, bar_atr, label_from_net_r, simulate_trade

BAR = settings.INTERVAL_MS


@pytest.fixture(autouse=True)
def _live_cfg(monkeypatch):
    monkeypatch.setattr(settings, "ACTIVE_STRATEGY", "trend")
    monkeypatch.setattr(settings, "PAPER_MAX_HOLD_BARS", 25)
    monkeypatch.setattr(settings, "PAPER_BREAKEVEN_AT_R", 0.0)
    monkeypatch.setattr(settings, "PAPER_TARGET_R_MULTIPLE", 0.0)


def make_path(rng, n=160, drift=0.0):
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    o = np.concatenate([[100.0], c[:-1]]) * (1 + rng.normal(0, 0.001, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    t = np.arange(n) * BAR
    return o, h, l, c, t


def live_trade(o, h, l, c, t, i, side):
    tr = PaperTrader(state_path=None, persist=False, log_trades=False)
    bars = [{"open_time": int(t[k]), "open": o[k], "high": h[k], "low": l[k], "close": c[k]}
            for k in range(len(o))]
    for k in range(i):
        tr.manage(bars[k])
    tr.on_bar(bars[i], True, Dir.LONG if side > 0 else Dir.SHORT, "test")
    for k in range(i + 1, len(bars)):
        tr.manage(bars[k])
        if tr.position is None and tr.trades:
            return tr.trades[-1], k
    return None, None


@pytest.mark.parametrize("seed,drift", [(0, 0.0), (1, 0.003), (2, -0.003), (3, 0.001)])
def test_simulator_matches_paper_trader(seed, drift):
    rng = np.random.default_rng(seed)
    rules = ExitRules.from_settings()
    o, h, l, c, t = make_path(rng, drift=drift)
    atr = bar_atr(h, l, t, rules.atr_window, BAR)
    compared, reasons = 0, set()
    for i in range(rules.atr_window, 120, 3):
        for side in (1, -1):
            live, _ = live_trade(o, h, l, c, t, i, side)
            sim = simulate_trade(o, h, l, c, t, atr, i, side, rules, BAR)
            if live is None or sim is None:
                assert (live is None) == (sim is None), (i, side)
                continue
            assert live["exit_reason"] == sim["reason"], (i, side)
            assert live["exit"] == pytest.approx(sim["exit_price"], rel=1e-12)
            assert live["r_multiple"] == pytest.approx(sim["net_r"], rel=1e-9, abs=1e-9)
            assert live["bars_held"] == sim["held"]
            compared += 1
            reasons.add(sim["reason"])
    assert compared > 30 and {"STOP"} <= reasons


def test_time_stop_parity():
    rules = ExitRules.from_settings(max_hold_bars=5)
    settings.PAPER_MAX_HOLD_BARS = 5
    n = 40
    t = np.arange(n) * BAR
    o = np.full(n, 100.0); c = o.copy(); h = o + 0.5; l = o - 0.5      # dead flat: never stopped
    atr = bar_atr(h, l, t, rules.atr_window, BAR)
    sim = simulate_trade(o, h, l, c, t, atr, 20, 1, rules, BAR)
    live, _ = live_trade(o, h, l, c, t, 20, 1)
    assert sim["reason"] == live["exit_reason"] == "TIME_STOP" and sim["held"] == 5
    assert live["r_multiple"] == pytest.approx(sim["net_r"])
    assert label_from_net_r(sim["net_r"], 0.1) == 0                      # flat -> breakeven


def test_label_semantics():
    rules = ExitRules.from_settings()
    n = 60
    t = np.arange(n) * BAR
    base = np.full(n, 100.0)
    h, l = base + 1.0, base - 1.0                                        # ATR = 2 -> stop distance 4
    atr = bar_atr(h, l, t, rules.atr_window, BAR)
    # (a) price collapses right after entry -> initial stop -> LOSS (-1)
    o, c = base.copy(), base.copy()
    l2 = l.copy(); l2[21:] = 90.0
    sim = simulate_trade(o, h, l2, c, t, atr, 20, 1, rules, BAR)
    assert sim["reason"] == "STOP" and label_from_net_r(sim["net_r"], 0.1) == -1
    # (b) steady rally then pullback -> TRAILING stop exits in profit -> WIN (+1)
    px = np.concatenate([np.full(21, 100.0), 100 + 3.0 * np.arange(1, 15), np.full(n - 35, 130.0)])
    o3 = np.concatenate([[100.0], px[:-1]]); h3 = np.maximum(o3, px) + 1; l3 = np.minimum(o3, px) - 1
    l3[36:] = 100.0                                                      # crash after the run-up
    atr3 = bar_atr(h3, l3, t, rules.atr_window, BAR)
    sim = simulate_trade(o3, h3, l3, px, t, atr3, 20, 1, rules, BAR)
    assert sim["reason"] == "STOP" and sim["exit_price"] > sim["entry"] and label_from_net_r(sim["net_r"], 0.1) == 1
    # (c) fixed profit target (non-trend rule set) -> +1
    r_t = ExitRules.from_settings(trail=False, target_r=1.5, stop_pct=2.0)
    h4 = base.copy() + 1; h4[22:] = 110.0
    sim = simulate_trade(o, h4, l, c, t, atr, 20, 1, r_t, BAR)
    assert sim["reason"] == "TARGET" and label_from_net_r(sim["net_r"], 0.1) == 1


def test_incomplete_or_gapped_path_is_unlabelable():
    rules = ExitRules.from_settings()
    n = 30
    t = np.arange(n) * BAR
    o = np.full(n, 100.0); h = o + 1; l = o - 1; c = o.copy()
    atr = bar_atr(h, l, t, rules.atr_window, BAR)
    assert simulate_trade(o, h, l, c, t, atr, 20, 1, rules, BAR) is None      # still open at end
    t2 = t.copy(); t2[24:] += BAR                                               # missing bar
    assert simulate_trade(o, h, l, c, t2, atr, 20, 1, rules, BAR) is None
