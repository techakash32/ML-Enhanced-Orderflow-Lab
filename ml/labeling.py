# ml/labeling.py
"""
Phase 2: volatility-scaled triple-barrier labels + purged walk-forward CV.

Conventions (chosen to mirror execution/paper_trader.py)
  * Decision at snapshot time T (a bar close for live trading). Entry is the
    OPEN of the next snapshot interval - "you cannot trade the close you just
    observed". The entry interval's own range is scanned, like PaperTrader.
  * ATR = simple mean of (high-low) over the last `atr_window` COMPLETED
    decision bars (same definition as PaperTrader._average_range), taken from
    bars that ended at or before T. Nothing later is read.
  * Barriers are entry +/- k*ATR: target +tp_atr, stop -sl_atr, vertical
    max_hold_bars decision bars. If stop and target are touched in the same
    interval the STOP is assumed first (PaperTrader does the same).
  * Label (side-relative): +1 target first, -1 stop first, 0 time expiry.
  * ret_atr = realised move in ATR units (+tp / -sl / mark-to-market at expiry),
    cost_atr = round-trip fees+slippage in ATR units, t1 = barrier exit time.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from ml.exit_sim import ExitRules, bar_atr, label_from_net_r, simulate_trade

# ----------------------------------------------------------------------------
# LABEL / EXIT PARITY RULE (Phase 6.1)
#   exit_mode="live_trail" (default): the label is the OUTCOME OF A SIMULATED LIVE TRADE -
#   same entry, ATR stop, trailing stop, time stop and costs as execution/paper_trader.py,
#   with all parameters read from config/settings.py via ml.exit_sim.ExitRules. Parity with
#   the real PaperTrader is enforced by tests/test_exit_parity.py, and models record the rules
#   they were trained on (manifest["exit_rules"]); the live gate refuses a mismatching model.
#   exit_mode="static_barrier": the original research-style triple barrier (below). It does NOT
#   describe how live trades exit - use for experiments only.
# ----------------------------------------------------------------------------


@dataclass
class BarrierConfig:
    tp_atr: float = 1.5
    sl_atr: float = 1.0
    max_hold_bars: int = 6
    bar_s: int = 14400            # decision bar (live: settings.INTERVAL)
    atr_window: int = 14
    snapshot_s: int = 60
    stride_s: int = 1800          # sample spacing; use bar_s for bar-close-only
    cost_pct: float = 0.18        # round trip % of price: 2*taker + 2*slippage
    min_bar_coverage: float = 0.95
    sample_mode: str = "all"      # all | candidates (Donchian breakout bar closes only)
    exit_mode: str = "live_trail"            # live_trail | static_barrier
    rules: Optional[ExitRules] = None        # live_trail rules; None -> ExitRules.from_settings()

    def live_rules(self) -> ExitRules:
        return self.rules or ExitRules.from_settings()

    def to_dict(self):
        d = asdict(self)
        d["rules"] = self.live_rules().to_dict()
        return d


# ------------------------------------------------------------- bars / ATR --
def build_decision_bars(snaps: pd.DataFrame, bar_s: int, snapshot_s: int,
                        min_coverage: float = 0.95) -> pd.DataFrame:
    """Aggregate snapshot intervals (stamped by END time) into decision bars.
    Bars with too little coverage (recorder downtime) are dropped."""
    d = snaps[["ts_ms", "open", "high", "low", "close", "volume"]].copy()
    start = d["ts_ms"] - snapshot_s * 1000                       # interval start
    d["open_time"] = (start // (bar_s * 1000)) * (bar_s * 1000)
    g = d.groupby("open_time")
    bars = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(), "n": g.size(),
    }).reset_index()
    need = bar_s / snapshot_s * min_coverage
    bars = bars[bars["n"] >= need].reset_index(drop=True)
    bars["end_ts"] = bars["open_time"] + bar_s * 1000            # == boundary stamp
    return bars


def atr_asof(ts: np.ndarray, bars: pd.DataFrame, window: int) -> np.ndarray:
    """ATR using only bars with end_ts <= ts. Requires `window` CONTIGUOUS bars."""
    out = np.full(len(ts), np.nan)
    if len(bars) < window:
        return out
    rng = (bars["high"] - bars["low"]).to_numpy()
    ot = bars["open_time"].to_numpy()
    contiguous = np.ones(len(bars), dtype=bool)
    step = np.diff(ot)
    bar_ms = int(np.median(step)) if len(step) else 0
    # roll: window bars ending at i are contiguous iff all steps == bar_ms
    ok = np.concatenate([[False] * (window - 1),
                         np.convolve((step == bar_ms).astype(int), np.ones(window - 1, int), "valid") == window - 1]) \
        if window > 1 else contiguous
    roll = pd.Series(rng).rolling(window).mean().to_numpy().copy()
    roll[~ok] = np.nan
    idx = np.searchsorted(bars["end_ts"].to_numpy(), ts, side="right") - 1
    valid = idx >= 0
    out[valid] = roll[idx[valid]]
    return out


# ----------------------------------------------------------- triple barrier --
def triple_barrier(ts: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                   i: int, side: int, atr: float, cfg: BarrierConfig):
    """
    Label one (sample, side). Arrays are the snapshot-interval series; `i` is the
    decision snapshot index. Returns (label, ret_atr, t1_ms, bars_to_exit) or
    None if the path is incomplete (feed gap / not enough future data).
    """
    steps = int(cfg.max_hold_bars * cfg.bar_s / cfg.snapshot_s)
    j0, j1 = i + 1, i + steps
    if j1 >= len(ts) or not np.isfinite(atr) or atr <= 0:
        # not enough future: only valid if a barrier is hit earlier, handled below
        j1 = min(j1, len(ts) - 1)
        short_path = True
    else:
        short_path = False
    if j0 > j1:
        return None
    entry = o[j0]
    if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        return None
    up_tp, dn_sl = entry + side * cfg.tp_atr * atr, entry - side * cfg.sl_atr * atr
    hh, ll = h[j0:j1 + 1], l[j0:j1 + 1]
    if side > 0:
        tp_hit, sl_hit = hh >= up_tp, ll <= dn_sl
    else:
        tp_hit, sl_hit = ll <= up_tp, hh >= dn_sl
    first_tp = int(np.argmax(tp_hit)) if tp_hit.any() else None
    first_sl = int(np.argmax(sl_hit)) if sl_hit.any() else None
    # feed gaps: path must be contiguous up to the exit point
    gaps = np.diff(ts[i:j1 + 1]) > 1.5 * cfg.snapshot_s * 1000
    first_gap = int(np.argmax(gaps)) if gaps.any() else None        # gap before interval j0+first_gap

    def contiguous_until(k):          # exit at relative index k
        return first_gap is None or first_gap > k

    if first_sl is not None and (first_tp is None or first_sl <= first_tp):
        k, label, ret = first_sl, -1, -cfg.sl_atr
    elif first_tp is not None:
        k, label, ret = first_tp, 1, cfg.tp_atr
    else:
        if short_path or not contiguous_until(len(hh) - 1):
            return None
        k, label = len(hh) - 1, 0
        ret = side * (c[j1] - entry) / atr
    if not contiguous_until(k):
        return None
    return label, float(ret), int(ts[j0 + k]), int(k + 1)


def make_labels(snaps: pd.DataFrame, cfg: BarrierConfig, symbol: str) -> pd.DataFrame:
    """All (sample, side) labels for one symbol. Returns one row per labelled sample."""
    from engine.decision import evaluate_bucket_a
    from gate.buckets import Dir
    snaps = snaps.sort_values("ts_ms").reset_index(drop=True)
    if cfg.exit_mode == "live_trail":
        return make_labels_live(snaps, cfg, symbol)
    ts = snaps["ts_ms"].to_numpy()
    o, h, l, c = (snaps[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    bars = build_decision_bars(snaps, cfg.bar_s, cfg.snapshot_s, cfg.min_bar_coverage)
    atr = atr_asof(ts, bars, cfg.atr_window)
    stride_ms = cfg.stride_s * 1000
    ok = (snaps["warm"].to_numpy() == 1.0) & np.isfinite(atr) & (ts % stride_ms == 0)
    cand_side: dict[int, int] = {}
    if cfg.sample_mode == "candidates":
        ok &= ts % (cfg.bar_s * 1000) == 0
        # reuse the LIVE Donchian signal so training candidates == live candidates
        hist = bars.to_dict("records")
        for bi in range(len(hist)):
            a, _ = evaluate_bucket_a(hist[:bi + 1], None, hist[bi]["close"])
            if a.direction != Dir.NEUTRAL:
                cand_side[int(hist[bi]["end_ts"])] = 1 if a.direction == Dir.LONG else -1
    rows = []
    for i in np.flatnonzero(ok):
        sides = (cand_side.get(int(ts[i])),) if cfg.sample_mode == "candidates" else (1, -1)
        for side in sides:
            if side is None:
                continue
            r = triple_barrier(ts, o, h, l, c, int(i), side, atr[i], cfg)
            if r is None:
                continue
            label, ret, t1, held = r
            price = o[i + 1]
            rows.append((symbol, int(ts[i]), side, label, ret, price * cfg.cost_pct / 100.0 / atr[i],
                         t1, held, atr[i], atr[i] / c[i] * 100.0, int(i)))
    out = pd.DataFrame(rows, columns=["symbol", "ts_ms", "side", "y", "ret_atr", "cost_atr",
                                      "t1_ms", "held", "atr", "atr_pct", "row"])
    out["net_r"] = (out["ret_atr"] - out["cost_atr"]) / cfg.sl_atr
    out["cand"] = int(cfg.sample_mode == "candidates")
    return out


def make_labels_live(snaps: pd.DataFrame, cfg: BarrierConfig, symbol: str) -> pd.DataFrame:
    """
    Parity labels: one row per (decision bar close, side). Decision points are bar closes ONLY
    (that is when the live Decider runs); cfg.stride_s is ignored. Columns as make_labels plus
    net_r (== PaperTrader r_multiple) and cand (1 if the live Bucket-A Donchian rule fires in
    that direction on that bar, i.e. the trade the live gate would actually be asked about).
    """
    from engine.decision import evaluate_bucket_a
    from gate.buckets import Dir
    rules = cfg.live_rules()
    bars = build_decision_bars(snaps, cfg.bar_s, cfg.snapshot_s, cfg.min_bar_coverage)
    cols = ["symbol", "ts_ms", "side", "y", "ret_atr", "cost_atr", "t1_ms", "held", "atr", "atr_pct",
            "row", "net_r", "cand", "exit_reason"]
    if len(bars) < rules.atr_window + 2:
        return pd.DataFrame(columns=cols)
    bar_ms = cfg.bar_s * 1000
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ot = bars["open_time"].to_numpy()
    end_ts = bars["end_ts"].to_numpy()
    atr = bar_atr(h, l, ot, rules.atr_window, bar_ms)
    row_of = pd.Series(np.arange(len(snaps)), index=snaps["ts_ms"].to_numpy())
    warm = snaps["warm"].to_numpy() == 1.0
    hist = bars.to_dict("records")
    rows = []
    for i in range(len(bars) - 1):
        r = row_of.get(int(end_ts[i]))
        if r is None or not warm[r] or not np.isfinite(atr[i]):
            continue
        a, _ = evaluate_bucket_a(hist[:i + 1], None, hist[i]["close"])
        cand_dir = 0 if a.direction == Dir.NEUTRAL else (1 if a.direction == Dir.LONG else -1)
        if cfg.sample_mode == "candidates" and cand_dir == 0:
            continue
        for side in ((cand_dir,) if cfg.sample_mode == "candidates" else (1, -1)):
            sim = simulate_trade(o, h, l, c, ot, atr, i, side, rules, bar_ms)
            if sim is None:
                continue
            move = sim["gross_r"] * sim["stop_dist"] / atr[i]
            cost = (sim["gross_r"] - sim["net_r"]) * sim["stop_dist"] / atr[i]
            rows.append((symbol, int(end_ts[i]), side, label_from_net_r(sim["net_r"], rules.be_band_r),
                         move, cost, int(end_ts[sim["exit_bar"]]), sim["held"], atr[i],
                         atr[i] / c[i] * 100.0, int(r), sim["net_r"], int(cand_dir == side), sim["reason"]))
    return pd.DataFrame(rows, columns=cols)


def continuation_label(snaps: pd.DataFrame, cfg: BarrierConfig, lab: pd.DataFrame) -> pd.Series:
    """Optional simple baseline: 1 if price at the vertical barrier is beyond entry in the
    side's direction (continuation), else 0. Uses the same decision points."""
    c = snaps.sort_values("ts_ms")["close"].to_numpy(float)
    o = snaps.sort_values("ts_ms")["open"].to_numpy(float)
    steps = int(cfg.max_hold_bars * cfg.bar_s / cfg.snapshot_s)
    ix = lab["row"].to_numpy()
    j1 = np.minimum(ix + steps, len(c) - 1)
    return pd.Series(((c[j1] - o[ix + 1]) * lab["side"].to_numpy() > 0).astype(int), index=lab.index)


# ------------------------------------------------------- purged walk-forward --
class PurgedWalkForward:
    """
    Expanding-window walk-forward split on TIME with purging + embargo.
    The span [t_min, t_max] is cut into n_splits+1 equal-time segments; the first
    is the minimum train window and each later segment is a test fold. For fold k:
        test  = samples whose decision time t0 lies in segment k+1
        train = samples with t0 before the segment AND label end t1 < test_start - embargo
    so no training label overlaps the test period (purge) and a gap protects
    against serial correlation (embargo).
    """

    def __init__(self, n_splits: int = 5, embargo_s: int = 0):
        self.n_splits, self.embargo_ms = n_splits, int(embargo_s * 1000)

    def split(self, t0: np.ndarray, t1: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        t0, t1 = np.asarray(t0), np.asarray(t1)
        lo, hi = t0.min(), t0.max() + 1
        edges = np.linspace(lo, hi, self.n_splits + 2)
        for k in range(1, self.n_splits + 1):
            s, e = edges[k], edges[k + 1]
            test = np.flatnonzero((t0 >= s) & (t0 < e))
            train = np.flatnonzero((t0 < s) & (t1 < s - self.embargo_ms))
            if len(train) and len(test):
                yield train, test


def purge_before(t0: np.ndarray, t1: np.ndarray, boundary_ms: int, embargo_s: int = 0) -> np.ndarray:
    """Indices of samples usable for training strictly before `boundary_ms`
    (decision before it, label resolved before it minus embargo)."""
    t0, t1 = np.asarray(t0), np.asarray(t1)
    return np.flatnonzero((t0 < boundary_ms) & (t1 < boundary_ms - embargo_s * 1000))
