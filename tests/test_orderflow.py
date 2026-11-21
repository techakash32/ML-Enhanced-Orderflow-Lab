import math

import numpy as np

from features.orderflow import (FEATURE_COLUMNS, MODEL_FEATURES, OrderFlowConfig,
                                OrderFlowFeatureEngine, to_model_frame)
from ml.synthetic import generate_events


def run(events, cfg=None, upto=None):
    rows = []
    eng = OrderFlowFeatureEngine("T", cfg or OrderFlowConfig(), sink=rows.append)
    for ev in events:
        if upto is not None and ev[1] >= upto:
            break
        if ev[0] == "t":
            eng.on_trade(*ev[1:])
        else:
            eng.on_book(*ev[1:])
    return rows


def test_causality_future_events_do_not_change_past_rows():
    events = list(generate_events(days=0.5, seed=1))
    full = run(events)
    cut_ts = events[len(events) // 2][1]
    part = run(events, upto=cut_ts)
    assert len(part) > 100
    for a, b in zip(part, full):          # every row emitted before the cut is identical
        for k in FEATURE_COLUMNS + ["close", "volume", "warm"]:
            x, y = a[k], b[k]
            assert (math.isnan(x) and math.isnan(y)) or x == y, (a["ts_ms"], k)


def test_mutating_future_leaves_past_unchanged():
    events = list(generate_events(days=0.4, seed=2))
    cut = events[len(events) // 2][1]
    mutated = [(e[0], e[1], e[2] * 1.5, e[3], e[4]) if e[0] == "t" and e[1] >= cut else e for e in events]
    a, b = run(events), run(mutated)
    past = [r for r in a if r["ts_ms"] <= cut]
    for ra, rb in zip(past, b):
        assert ra["ts_ms"] == rb["ts_ms"]
        for k in FEATURE_COLUMNS:
            x, y = ra[k], rb[k]
            assert (math.isnan(x) and math.isnan(y)) or x == y


def test_vpin_buckets_on_toy_stream():
    cfg = OrderFlowConfig(snapshot_s=60, vpin_bucket_volume=10.0, vpin_window=3, z_min=1,
                          warm_min_snapshots=1)
    eng = OrderFlowFeatureEngine("T", cfg)
    t0 = 60_000
    eng.on_book(t0, [(99, 1)], [(101, 1)])
    # three buckets: 10 buy (|imb|=1), 5/5 (0), 10 sell (1)  => VPIN = 2/3
    eng.on_trade(t0 + 1, 100, 10, False)
    eng.on_trade(t0 + 2, 100, 5, False)
    eng.on_trade(t0 + 3, 100, 5, True)
    eng.on_trade(t0 + 4, 100, 10, True)
    rows = eng.advance_to(t0 + 130_000)
    assert abs(rows[0]["vpin"] - 2 / 3) < 1e-9
    assert abs(rows[0]["vpin_signed"] - 0.0) < 1e-9


def test_book_features_math():
    cfg = OrderFlowConfig(snapshot_s=60, depth_levels=2)
    eng = OrderFlowFeatureEngine("T", cfg)
    t0 = 120_000
    eng.on_book(t0, [(100, 3), (99, 1)], [(102, 1), (103, 1)])
    eng.on_trade(t0 + 10, 101, 1, False)
    row = eng.advance_to(t0 + 60_000)[0]
    assert abs(row["qi_l1"] - 0.5) < 1e-9                       # (3-1)/(3+1)
    assert abs(row["ofi_snap_l5"] - (4 - 2) / 6) < 1e-9
    micro = (100 * 1 + 102 * 3) / 4
    assert abs(row["micro_dev_bps"] - (micro - 101) / 101 * 1e4) < 1e-6
    assert abs(row["spread_bps"] - 2 / 101 * 1e4) < 1e-6


def test_gap_resets_state():
    eng = OrderFlowFeatureEngine("T", OrderFlowConfig(snapshot_s=60, max_gap_intervals=3))
    eng.on_book(60_000, [(99, 1)], [(101, 1)])
    eng.on_trade(61_000, 100, 1, False)
    eng.advance_to(60_000 + 3_600_000)
    assert eng.resets == 1


def test_to_model_frame_flips_directional_only():
    row = {c: 1.0 for c in FEATURE_COLUMNS}
    long_ = to_model_frame(row, +1, 0.5).iloc[0]
    short = to_model_frame(row, -1, 0.5).iloc[0]
    assert list(long_.index) == MODEL_FEATURES
    assert short["ofi_snap_l5"] == -1.0 and short["vpin"] == 1.0 and short["side"] == -1.0
