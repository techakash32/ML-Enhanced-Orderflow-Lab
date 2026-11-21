# ml/exit_sim.py
"""
Label/exit parity (Phase 6.1).

Simulates ONE trade on a series of decision bars with the SAME rules as
execution/paper_trader.py. The rules come from config/settings.py - the single source
of truth the PaperTrader also reads (TREND_ATR_MULTIPLE, TREND_ATR_WINDOW,
PAPER_MAX_HOLD_BARS, PAPER_SLIPPAGE_PCT, PAPER_FEE_*, PAPER_TARGET_R_MULTIPLE) - so the
labeler cannot silently drift. tests/test_exit_parity.py replays random paths through the
REAL PaperTrader and asserts this simulator gives identical exits and R-multiples.

Live rules mirrored (trend mode):
  * signal on bar i's close -> fill at bar i+1 OPEN, plus adverse slippage
  * initial stop = entry -/+ ATR_MULT * ATR(window) measured on bars <= i
  * the ENTRY bar's own range is checked against the stop
  * no profit target (a fixed R target only if PAPER_TARGET_R_MULTIPLE > 0 outside trend mode)
  * after each completed, non-exited bar j: stop = max(stop, high_j - ATR_MULT*ATR_j)
    (long; mirror for short), ATR_j includes bar j; active from bar j+1 only
  * stop hit when low <= stop (long) / high >= stop (short), filled AT the stop (+ slippage)
  * TIME_STOP at the close of the bar where held >= max_hold_bars (+ slippage)
  * costs: taker fee on entry and exit (maker on a TARGET exit); R = net pnl / (stop distance)
Label: +1 win / -1 loss / 0 breakeven by NET R with a dead band (be_band_r).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from config import settings


@dataclass(frozen=True)
class ExitRules:
    trail: bool = True
    atr_mult: float = 2.0
    atr_window: int = 14
    max_hold_bars: int = 90
    target_r: float | None = None
    stop_pct: float = 2.0                # non-trend fallback stop (% of entry)
    slippage_pct: float = 0.04
    taker_pct: float = 0.05
    maker_pct: float = 0.02
    be_band_r: float = 0.10              # |net R| below this counts as breakeven (label 0)

    @classmethod
    def from_settings(cls, **override) -> "ExitRules":
        trend = getattr(settings, "ACTIVE_STRATEGY", "sweep") == "trend"
        r = cls(trail=trend, atr_mult=settings.TREND_ATR_MULTIPLE, atr_window=settings.TREND_ATR_WINDOW,
                max_hold_bars=int(settings.PAPER_MAX_HOLD_BARS),
                target_r=None if (trend or settings.PAPER_TARGET_R_MULTIPLE <= 0)
                else float(settings.PAPER_TARGET_R_MULTIPLE),
                stop_pct=settings.PAPER_STOP_PCT, slippage_pct=settings.PAPER_SLIPPAGE_PCT,
                taker_pct=settings.PAPER_FEE_TAKER_PCT, maker_pct=settings.PAPER_FEE_MAKER_PCT)
        return cls(**{**asdict(r), **override}) if override else r

    def to_dict(self) -> dict:
        return asdict(self)

    def parity_key(self) -> dict:
        """The fields that define live exit behaviour (compared against a model's manifest)."""
        d = self.to_dict()
        d.pop("be_band_r")
        return d


def bar_atr(high: np.ndarray, low: np.ndarray, open_time: np.ndarray, window: int, bar_ms: int) -> np.ndarray:
    """ATR_j = mean(high-low) of the `window` bars ending at j (inclusive); NaN unless contiguous."""
    n = len(high)
    out = np.full(n, np.nan)
    if n < window:
        return out
    rng = high - low
    cs = np.concatenate([[0.0], np.cumsum(rng)])
    out[window - 1:] = (cs[window:] - cs[:-window]) / window
    step_ok = np.concatenate([[0], (np.diff(open_time) == bar_ms).astype(int)])
    cc = np.concatenate([[0], np.cumsum(step_ok)])
    for j in range(window - 1, n):                      # window bars => window-1 steps all ok
        if cc[j + 1] - cc[j - window + 2] != window - 1:
            out[j] = np.nan
    return out


def simulate_trade(o, h, l, c, open_time, atr, i: int, side: int, rules: ExitRules, bar_ms: int):
    """
    Trade signalled on bar i's close. Returns dict(reason, exit_bar, exit_price, net_r, gross_r,
    held, stop_dist, entry) or None if the path is incomplete/non-contiguous (not labelable).
    """
    j0 = i + 1
    n = len(o)
    if j0 >= n or open_time[j0] - open_time[i] != bar_ms:
        return None
    a0 = atr[i]
    if rules.trail:
        if not np.isfinite(a0) or a0 <= 0:
            return None
        stop_dist_of = lambda entry: rules.atr_mult * a0                       # noqa: E731
    else:
        stop_dist_of = lambda entry: entry * rules.stop_pct / 100.0            # noqa: E731
    entry = o[j0] * (1 + side * rules.slippage_pct / 100.0)
    stop_dist = stop_dist_of(entry)
    if entry <= 0 or stop_dist <= 0:
        return None
    stop = entry - side * stop_dist
    target = entry + side * stop_dist * rules.target_r if rules.target_r else None
    for j in range(j0, n):
        if j > j0 and open_time[j] - open_time[j - 1] != bar_ms:
            return None
        hit = (l[j] <= stop) if side > 0 else (h[j] >= stop)
        thit = target is not None and ((h[j] >= target) if side > 0 else (l[j] <= target))
        if hit:
            return _finish("STOP", j, stop, entry, stop_dist, side, j - j0 + 1, rules)
        if thit:
            return _finish("TARGET", j, target, entry, stop_dist, side, j - j0 + 1, rules)
        held = j - j0 + 1
        if rules.max_hold_bars and held >= rules.max_hold_bars:
            return _finish("TIME_STOP", j, c[j], entry, stop_dist, side, held, rules)
        if rules.trail:
            aj = atr[j]
            if np.isfinite(aj) and aj > 0:
                stop = max(stop, h[j] - rules.atr_mult * aj) if side > 0 else min(stop, l[j] + rules.atr_mult * aj)
    return None                                           # still open at end of data


def _finish(reason, j, price, entry, stop_dist, side, held, r: ExitRules) -> dict:
    if reason in ("STOP", "TIME_STOP"):
        price = price * (1 - side * r.slippage_pct / 100.0)
    gross = (price - entry) * side
    exit_rate = r.maker_pct if reason == "TARGET" else r.taker_pct
    fees = entry * r.taker_pct / 100.0 + price * exit_rate / 100.0
    return {"reason": reason, "exit_bar": j, "exit_price": price, "held": held, "entry": entry,
            "stop_dist": stop_dist, "gross_r": gross / stop_dist, "net_r": (gross - fees) / stop_dist}


def label_from_net_r(net_r: float, band: float) -> int:
    return 1 if net_r > band else (-1 if net_r < -band else 0)
