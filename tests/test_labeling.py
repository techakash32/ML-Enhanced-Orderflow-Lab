import numpy as np
import pandas as pd

from ml.labeling import BarrierConfig, PurgedWalkForward, triple_barrier, atr_asof, build_decision_bars


def path(highs, lows, opens=None, closes=None, dt=60_000):
    n = len(highs)
    ts = np.arange(n) * dt
    o = np.array(opens if opens is not None else [100.0] * n, float)
    c = np.array(closes if closes is not None else [100.0] * n, float)
    return ts, o, np.array(highs, float), np.array(lows, float), c


CFG = BarrierConfig(tp_atr=1.5, sl_atr=1.0, max_hold_bars=1, bar_s=300, snapshot_s=60)   # 5-step horizon


def test_target_first_long():
    ts, o, h, l, c = path([100, 100, 100, 101, 103, 100, 100], [100, 100, 99.5, 99.5, 99.5, 99, 99])
    # entry=o[1]=100, atr=2 -> tp=103, sl=98 ; high hits 103 at index 4
    r = triple_barrier(ts, o, h, l, c, 0, +1, 2.0, CFG)
    assert r[0] == 1 and r[1] == 1.5 and r[3] == 4


def test_stop_first_and_tie_goes_to_stop():
    ts, o, h, l, c = path([100, 103, 100, 100, 100, 100, 100], [100, 97, 100, 100, 100, 100, 100])
    r = triple_barrier(ts, o, h, l, c, 0, +1, 2.0, CFG)       # both touched in same interval
    assert r[0] == -1 and r[1] == -1.0


def test_short_side_mirrors():
    ts, o, h, l, c = path([100, 100, 100.5, 100.5, 100, 100, 100], [100, 100, 99, 96.5, 99, 99, 99])
    r = triple_barrier(ts, o, h, l, c, 0, -1, 2.0, CFG)       # tp = 97, sl=102 ; low 96.5 hits tp
    assert r[0] == 1


def test_timeout_marks_to_market():
    ts, o, h, l, c = path([100] * 8, [100] * 8, closes=[100, 100, 100, 100, 100, 100, 100, 100])
    c[5] = 101.0           # vertical barrier at j1 = 0+5
    r = triple_barrier(ts, o, h, l, c, 0, +1, 2.0, CFG)
    assert r[0] == 0 and abs(r[1] - 0.5) < 1e-9


def test_incomplete_future_returns_none_but_early_hit_ok():
    ts, o, h, l, c = path([100, 100, 100], [100, 100, 100])
    assert triple_barrier(ts, o, h, l, c, 0, +1, 2.0, CFG) is None
    ts, o, h, l, c = path([100, 100, 104], [100, 100, 100])
    assert triple_barrier(ts, o, h, l, c, 0, +1, 2.0, CFG)[0] == 1


def test_gap_in_feed_invalidates():
    ts, o, h, l, c = path([100] * 8, [100] * 8)
    ts = ts.copy(); ts[4:] += 600_000                          # 10-minute hole
    assert triple_barrier(ts, o, h, l, c, 0, +1, 2.0, CFG) is None


def test_atr_uses_only_completed_bars():
    snaps = pd.DataFrame({"ts_ms": np.arange(1, 41) * 60_000, "open": 100.0,
                          "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0})
    snaps.loc[snaps.index[-1], "high"] = 500.0                 # the last (future) bar explodes
    bars = build_decision_bars(snaps, bar_s=600, snapshot_s=60, min_coverage=0.9)
    atr = atr_asof(np.array([30 * 60_000]), bars, 2)           # at T = end of 3rd bar
    assert abs(atr[0] - 2.0) < 1e-9                            # unaffected by later data


def test_purged_walk_forward_no_leak():
    rng = np.random.default_rng(0)
    t0 = np.sort(rng.integers(0, 1_000_000, 2000))
    t1 = t0 + rng.integers(1000, 60_000, 2000)
    cv = PurgedWalkForward(n_splits=4, embargo_s=5)
    folds = list(cv.split(t0, t1))
    assert len(folds) == 4
    for tr, te in folds:
        assert t1[tr].max() < t0[te].min() - 5000              # no label overlap + embargo
        assert t0[tr].max() < t0[te].min()
