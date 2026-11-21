# features/orderflow.py
"""
Causal order-flow feature extraction (Phase 1).

One OrderFlowFeatureEngine per symbol consumes
  * aggTrades  : on_trade(ts_ms, price, qty, is_buyer_maker)
  * book depth : on_book(ts_ms, bids, asks)       (top-N levels, best first)
and emits one feature *snapshot* every `snapshot_s` seconds of EVENT time.

Causality contract
------------------
A snapshot stamped T (the interval END) is computed only from events that were
processed before the first event with ts >= T. Nothing after T is ever read.
Rolling statistics (z-scores, percentiles, VPIN history) use only data up to
and including the current snapshot. A late event (ts earlier than the open
interval, e.g. from the other websocket) is applied to the open interval: it
is older information than the snapshot it lands in, so this cannot leak the
future. Offline replay of the same event stream reproduces the same rows.

Features (see FEATURE_COLUMNS). "Directional" ones flip sign for a SHORT
sample - see to_model_frame() - so one model serves both sides.

  OFI        ofi_snap_l5 (+z)        multi-level snapshot imbalance, time-weighted
             ofi_ev_1m / ofi_ev_5m (+z)   Cont-Kukanov-Stoikov event OFI at L1,
                                     normalised by average top-of-book size
  Queue      qi_l1, qi_roc           best-level (bid-ask)/(bid+ask) = the
                                     "standard snapshot OFI" formula, and its
                                     rate of change per minute
  Microprice micro_dev_bps, micro_dev_avg, micro_drift_bps
  VPIN       vpin, vpin_z, vpin_band, vpin_signed, vpin_age_s   (volume buckets,
             exchange-labelled aggressor side, so no bulk classification needed)
  Trades     tsize_mean/std/skew, large_vol_share, large_cnt_frac, large_net,
             trade_imb, trade_rate
  Context    spread_bps, ret_15m/1h/4h, rvol_1h
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)
NAN = float("nan")

# ---------------------------------------------------------------- schema ---
BASE_COLUMNS = ["ts_ms", "symbol", "open", "high", "low", "close", "mid",
                "volume", "buy_vol", "sell_vol", "n_trades", "warm"]

# sign-flipped for SHORT samples (directional)
DIRECTIONAL_FEATURES = [
    "ofi_snap_l5", "ofi_snap_l5_z", "ofi_ev_1m", "ofi_ev_5m", "ofi_ev_5m_z",
    "qi_l1", "qi_roc", "micro_dev_bps", "micro_dev_avg", "micro_drift_bps",
    "vpin_signed", "large_net", "trade_imb", "ret_15m", "ret_1h", "ret_4h",
]
# side-neutral
NEUTRAL_FEATURES = [
    "vpin", "vpin_z", "vpin_band", "vpin_age_s",
    "tsize_mean", "tsize_std", "tsize_skew", "large_vol_share", "large_cnt_frac",
    "trade_rate", "spread_bps", "rvol_1h",
]
FEATURE_COLUMNS = DIRECTIONAL_FEATURES + NEUTRAL_FEATURES
ALL_COLUMNS = BASE_COLUMNS + FEATURE_COLUMNS
# Extra model inputs added by to_model_frame (not stored in the feature store)
MODEL_EXTRA = ["side", "atr_pct"]
MODEL_FEATURES = DIRECTIONAL_FEATURES + NEUTRAL_FEATURES + MODEL_EXTRA


@dataclass
class OrderFlowConfig:
    snapshot_s: int = 60
    depth_levels: int = 5
    # VPIN
    vpin_window: int = 50              # buckets averaged
    vpin_buckets_per_day: float = 200  # auto bucket volume = est. daily volume / this
    vpin_bucket_volume: Optional[float] = None   # fixed override (base-asset units)
    vpin_hist: int = 500
    vpin_bands: tuple = (50.0, 75.0, 95.0)       # percentile edges -> band 0..3
    # rolling z-scores (in snapshots)
    z_window: int = 240
    z_min: int = 30
    # trade-size stats
    trade_window_s: int = 300
    trade_ref_n: int = 20000
    large_q: float = 95.0
    # horizons
    ret_windows_s: tuple = (900, 3600, 14400)
    qi_roc_s: int = 300
    micro_drift_s: int = 300
    ofi_ev_windows_s: tuple = (60, 300)
    # gaps longer than this many intervals reset history (continuity broken)
    max_gap_intervals: int = 5
    warm_min_snapshots: int = 30


def _zscore(hist: deque, x: float, min_n: int) -> float:
    """z of x against PRIOR values in hist (caller appends x afterwards)."""
    if not math.isfinite(x) or len(hist) < min_n:
        return NAN
    a = np.fromiter(hist, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < min_n:
        return NAN
    sd = a.std()
    return float((x - a.mean()) / sd) if sd > 1e-12 else 0.0


def _imb(a: float, b: float) -> float:
    t = a + b
    return (a - b) / t if t > 0 else 0.0


class OrderFlowFeatureEngine:
    def __init__(self, symbol: str, cfg: OrderFlowConfig | None = None,
                 sink: Optional[Callable[[dict], None]] = None):
        self.symbol = symbol
        self.cfg = cfg or OrderFlowConfig()
        self.sink = sink
        self.snapshots: deque = deque(maxlen=64)       # recent emitted rows
        self.late_events = 0
        self.resets = 0
        self._dt_ms = self.cfg.snapshot_s * 1000
        self._next_b: Optional[int] = None              # next boundary (ms)
        self._pb = None                                 # previous L1 (bp,bq,ap,aq)
        self._spread_bps = NAN
        self._reset_history()
        self._reset_interval(None)

    # ------------------------------------------------------------ state ---
    def _reset_history(self) -> None:
        c = self.cfg
        n_ret = int(max(c.ret_windows_s) / c.snapshot_s) + 2
        self._closes: deque = deque(maxlen=n_ret)
        self._qi_hist: deque = deque(maxlen=int(c.qi_roc_s / c.snapshot_s) + 2)
        self._md_hist: deque = deque(maxlen=int(c.micro_drift_s / c.snapshot_s) + 2)
        self._ofi5_hist: deque = deque(maxlen=c.z_window)
        self._ofiev5_hist: deque = deque(maxlen=c.z_window)
        n_ev = int(max(c.ofi_ev_windows_s) / c.snapshot_s)
        self._ofi_ev_ring: deque = deque(maxlen=max(n_ev, 1))   # per-interval normalised OFI
        self._vol_ring: deque = deque(maxlen=int(86400 / c.snapshot_s))
        self._tops_ema = NAN                                      # avg top-of-book size
        self._tw_trades: deque = deque()                          # (ts, qty, signed_qty, is_buy)
        self._ref_qty: deque = deque(maxlen=c.trade_ref_n)
        self._buckets_abs: deque = deque(maxlen=c.vpin_window)
        self._buckets_sgn: deque = deque(maxlen=c.vpin_window)
        self._vpin_hist: deque = deque(maxlen=c.vpin_hist)
        self._vpin_last_ts: Optional[int] = None
        self._bk_buy = self._bk_sell = 0.0
        self._bk_target: Optional[float] = None
        self._n_snap = 0
        self._book_seen = False

    def _reset_interval(self, start_ms: Optional[int]) -> None:
        self._i_start = start_ms
        self._o = self._h = self._l = self._c = None
        self._buy = self._sell = 0.0
        self._ntr = 0
        self._ofi_ev = 0.0
        self._tw_sum = np.zeros(3)         # ofi5, qi1, micro_dev
        self._tw_t = start_ms
        self._tw_cur = None                # current (ofi5, qi1, micro_dev)
        self._tw_cover = 0.0
        self._mid_last = NAN

    # ------------------------------------------------------------ events ---
    def advance_to(self, ts_ms: int) -> list[dict]:
        """Emit every snapshot whose boundary <= ts_ms. Safe to call any time."""
        out = []
        if self._next_b is None:
            self._next_b = (ts_ms // self._dt_ms + 1) * self._dt_ms
            self._reset_interval(self._next_b - self._dt_ms)
            return out
        if ts_ms >= self._next_b + self.cfg.max_gap_intervals * self._dt_ms:
            # Feed was down: continuity broken. Drop rolling state, resume.
            log.warning("%s: %.0fs gap in feed - resetting rolling state",
                        self.symbol, (ts_ms - self._next_b) / 1000)
            self.resets += 1
            self._reset_history()
            self._pb = None
            self._next_b = (ts_ms // self._dt_ms + 1) * self._dt_ms
            self._reset_interval(self._next_b - self._dt_ms)
            return out
        while ts_ms >= self._next_b:
            out.append(self._emit(self._next_b))
            self._next_b += self._dt_ms
        return out

    def on_trade(self, ts_ms: int, price: float, qty: float, is_buyer_maker: bool) -> None:
        self.advance_to(ts_ms)
        if self._i_start is not None and ts_ms < self._i_start:
            self.late_events += 1
        is_buy = not is_buyer_maker            # buyer-maker => aggressor sold
        signed = qty if is_buy else -qty
        if self._o is None:
            self._o = self._h = self._l = price
        self._h = max(self._h, price)
        self._l = min(self._l, price)
        self._c = price
        if is_buy:
            self._buy += qty
        else:
            self._sell += qty
        self._ntr += 1
        self._tw_trades.append((ts_ms, qty, signed, is_buy))
        self._ref_qty.append(qty)
        self._add_to_bucket(ts_ms, qty, is_buy)

    def on_book(self, ts_ms: int, bids: Sequence, asks: Sequence) -> None:
        """bids/asks: [(price, qty), ...] best first."""
        self.advance_to(ts_ms)
        if not bids or not asks:
            return
        c = self.cfg
        k = c.depth_levels
        bp, bq = float(bids[0][0]), float(bids[0][1])
        ap, aq = float(asks[0][0]), float(asks[0][1])
        if bp <= 0 or ap <= 0 or bq < 0 or aq < 0 or bp > ap:
            return                                         # crossed/garbage
        # time-weighted accumulators: close out the previous state first
        t = max(ts_ms, self._i_start or ts_ms)
        if self._tw_cur is not None and self._tw_t is not None:
            dt = max(0, t - self._tw_t)
            self._tw_sum += dt * np.array(self._tw_cur)
            self._tw_cover += dt
        self._tw_t = t
        mid = 0.5 * (bp + ap)
        bsum = sum(float(q) for _, q in bids[:k])
        asum = sum(float(q) for _, q in asks[:k])
        micro = (bp * aq + ap * bq) / (bq + aq) if (bq + aq) > 0 else mid
        self._tw_cur = (_imb(bsum, asum), _imb(bq, aq), (micro - mid) / mid * 1e4)
        # event OFI at L1 (Cont et al. 2014)
        if self._pb is not None:
            ppb, pbq, ppa, paq = self._pb
            e = ((bq if bp >= ppb else 0.0) - (pbq if bp <= ppb else 0.0)
                 - (aq if ap <= ppa else 0.0) + (paq if ap >= ppa else 0.0))
            self._ofi_ev += e
        self._pb = (bp, bq, ap, aq)
        top = 0.5 * (bq + aq)
        self._tops_ema = top if not math.isfinite(self._tops_ema) else 0.999 * self._tops_ema + 0.001 * top
        self._spread_bps = (ap - bp) / mid * 1e4
        self._mid_last = mid
        self._book_seen = True

    # ------------------------------------------------------------ VPIN ----
    def _est_bucket_volume(self) -> Optional[float]:
        c = self.cfg
        if c.vpin_bucket_volume:
            return c.vpin_bucket_volume
        n = len(self._vol_ring)
        if n < c.warm_min_snapshots:
            return None
        daily = sum(self._vol_ring) * (86400 / c.snapshot_s) / n
        return daily / c.vpin_buckets_per_day if daily > 0 else None

    def _add_to_bucket(self, ts_ms: int, qty: float, is_buy: bool) -> None:
        if self._bk_target is None:
            self._bk_target = self._est_bucket_volume()
            if self._bk_target is None:
                return
        remaining = qty
        while remaining > 0:
            room = self._bk_target - (self._bk_buy + self._bk_sell)
            take = min(room, remaining)
            if is_buy:
                self._bk_buy += take
            else:
                self._bk_sell += take
            remaining -= take
            if self._bk_buy + self._bk_sell >= self._bk_target - 1e-12:
                v = self._bk_target
                self._buckets_abs.append(abs(self._bk_buy - self._bk_sell) / v)
                self._buckets_sgn.append((self._bk_buy - self._bk_sell) / v)
                if len(self._buckets_abs) == self.cfg.vpin_window:
                    self._vpin_hist.append(float(np.mean(self._buckets_abs)))
                self._vpin_last_ts = ts_ms
                self._bk_buy = self._bk_sell = 0.0
                self._bk_target = self._est_bucket_volume()
                if self._bk_target is None:
                    return

    # --------------------------------------------------------- snapshot ----
    def _emit(self, boundary: int) -> dict:
        c = self.cfg
        start = boundary - self._dt_ms
        # --- interval price / flow
        if self._o is None:                       # no trades: carry the mid / last close
            ref = self._c if self._c is not None else self._mid_last
            if not math.isfinite(ref if ref is not None else NAN):
                ref = self._closes[-1] if self._closes else NAN
            o = h = l = cl = ref
        else:
            o, h, l, cl = self._o, self._h, self._l, self._c
        vol = self._buy + self._sell
        self._vol_ring.append(vol)
        # --- time-weighted book averages
        if self._tw_cur is not None and self._tw_t is not None:
            dt = max(0, boundary - self._tw_t)
            self._tw_sum += dt * np.array(self._tw_cur)
            self._tw_cover += dt
        if self._tw_cover > 0:
            ofi5, qi1, mdev_avg = (self._tw_sum / self._tw_cover).tolist()
            ofi5_ok = True
        else:
            ofi5 = qi1 = mdev_avg = NAN
            ofi5_ok = False
        cur = self._tw_cur
        micro_dev = cur[2] if cur else NAN
        # --- event OFI (normalised by avg top-of-book size)
        norm = self._tops_ema if (math.isfinite(self._tops_ema) and self._tops_ema > 0) else NAN
        ofi_ev_now = self._ofi_ev / norm if math.isfinite(norm) else NAN
        self._ofi_ev_ring.append(ofi_ev_now)
        ring = list(self._ofi_ev_ring)
        n5 = max(1, int(300 / c.snapshot_s))
        ofi_ev_5m = float(np.nansum(ring[-n5:])) if len(ring) >= 1 and np.isfinite(ring[-n5:]).any() else NAN
        # --- z-scores against prior history
        ofi5_z = _zscore(self._ofi5_hist, ofi5, c.z_min)
        ofiev5_z = _zscore(self._ofiev5_hist, ofi_ev_5m, c.z_min)
        if math.isfinite(ofi5):
            self._ofi5_hist.append(ofi5)
        if math.isfinite(ofi_ev_5m):
            self._ofiev5_hist.append(ofi_ev_5m)
        # --- queue imbalance rate of change (per minute)
        lag_q = int(c.qi_roc_s / c.snapshot_s)
        qi_roc = NAN
        if math.isfinite(qi1) and len(self._qi_hist) >= lag_q:
            qi_roc = (qi1 - self._qi_hist[-lag_q]) / (c.qi_roc_s / 60.0)
        self._qi_hist.append(qi1)
        # --- microprice drift vs mid
        lag_m = int(c.micro_drift_s / c.snapshot_s)
        micro_drift = NAN
        if math.isfinite(micro_dev) and len(self._md_hist) >= lag_m:
            micro_drift = micro_dev - self._md_hist[-lag_m]
        self._md_hist.append(micro_dev)
        # --- VPIN
        vpin = vpin_z = vpin_sgn = NAN
        band = NAN
        vpin_ok = len(self._buckets_abs) == c.vpin_window
        if vpin_ok:
            vpin = float(np.mean(self._buckets_abs))
            vpin_sgn = float(np.mean(self._buckets_sgn))
            prior = deque(list(self._vpin_hist)[:-1], maxlen=c.vpin_hist)  # exclude current
            vpin_z = _zscore(prior, vpin, c.z_min)
            if len(prior) >= c.z_min:
                edges = np.percentile(np.fromiter(prior, float), c.vpin_bands)
                band = float(np.searchsorted(edges, vpin, side="right"))
        vpin_age = (boundary - self._vpin_last_ts) / 1000.0 if self._vpin_last_ts else NAN
        # --- trade-size stats over the short window
        cutoff = boundary - c.trade_window_s * 1000
        while self._tw_trades and self._tw_trades[0][0] < cutoff:
            self._tw_trades.popleft()
        tsz = np.array([t[1] for t in self._tw_trades], dtype=float)
        mean = std = skew = NAN
        lv_share = lc_frac = l_net = t_imb = NAN
        rate = len(tsz) / (c.trade_window_s / 60.0)
        if len(tsz) >= 5:
            mean, std = float(tsz.mean()), float(tsz.std())
            skew = float(((tsz - mean) ** 3).mean() / std ** 3) if std > 1e-12 else 0.0
            ref = np.fromiter(self._ref_qty, float)
            if len(ref) >= 200:
                thr = np.percentile(ref, c.large_q)
                big = tsz > thr
                tot = tsz.sum()
                lv_share = float(tsz[big].sum() / tot) if tot > 0 else 0.0
                lc_frac = float(big.mean())
                sg = np.array([t[2] for t in self._tw_trades], dtype=float)
                lv = tsz[big].sum()
                l_net = float(sg[big].sum() / lv) if lv > 0 else 0.0
            sg_all = np.array([t[2] for t in self._tw_trades], dtype=float)
            t_imb = float(sg_all.sum() / tsz.sum()) if tsz.sum() > 0 else 0.0
        # --- price context
        self._closes.append(cl)
        rets = {}
        for w in c.ret_windows_s:
            lag = int(w / c.snapshot_s)
            if len(self._closes) > lag and self._closes[-1 - lag] and math.isfinite(self._closes[-1 - lag]):
                rets[w] = (cl - self._closes[-1 - lag]) / self._closes[-1 - lag] * 1e4
            else:
                rets[w] = NAN
        n1h = int(3600 / c.snapshot_s)
        rvol = NAN
        if len(self._closes) > n1h:
            cl_arr = np.array(list(self._closes)[-(n1h + 1):], dtype=float)
            if np.all(cl_arr > 0):
                rvol = float(np.diff(np.log(cl_arr)).std() * 1e4)
        self._n_snap += 1
        warm = float(vpin_ok and ofi5_ok and self._n_snap >= c.warm_min_snapshots
                     and math.isfinite(ofi5_z) and math.isfinite(vpin_z))
        mid_end = self._mid_last
        row = {
            "ts_ms": int(boundary), "symbol": self.symbol,
            "open": o, "high": h, "low": l, "close": cl,
            "mid": mid_end if math.isfinite(mid_end) else cl,
            "volume": vol, "buy_vol": self._buy, "sell_vol": self._sell,
            "n_trades": self._ntr, "warm": warm,
            "ofi_snap_l5": ofi5, "ofi_snap_l5_z": ofi5_z,
            "ofi_ev_1m": ofi_ev_now, "ofi_ev_5m": ofi_ev_5m, "ofi_ev_5m_z": ofiev5_z,
            "qi_l1": qi1, "qi_roc": qi_roc,
            "micro_dev_bps": micro_dev, "micro_dev_avg": mdev_avg, "micro_drift_bps": micro_drift,
            "vpin_signed": vpin_sgn, "large_net": l_net, "trade_imb": t_imb,
            "ret_15m": rets.get(900, NAN), "ret_1h": rets.get(3600, NAN), "ret_4h": rets.get(14400, NAN),
            "vpin": vpin, "vpin_z": vpin_z, "vpin_band": band, "vpin_age_s": vpin_age,
            "tsize_mean": mean, "tsize_std": std, "tsize_skew": skew,
            "large_vol_share": lv_share, "large_cnt_frac": lc_frac, "trade_rate": rate,
            "spread_bps": self._spread_bps, "rvol_1h": rvol,
        }
        self.snapshots.append(row)
        if self.sink:
            try:
                self.sink(row)
            except Exception:
                log.exception("feature sink failed")
        self._reset_interval(boundary)
        return row

    # ------------------------------------------------------------ query ----
    def snapshot_at(self, ts_ms: int) -> Optional[dict]:
        """Snapshot stamped exactly ts_ms (forces emission if the boundary has
        passed in wall time but no later event has arrived yet)."""
        self.advance_to(ts_ms)
        for row in reversed(self.snapshots):
            if row["ts_ms"] == ts_ms:
                return row
            if row["ts_ms"] < ts_ms:
                break
        return None


# ------------------------------------------------- model-input assembly ----
def to_model_frame(feats, side, atr_pct):
    """
    Shared by training and live inference (no train/serve skew).
    feats  : DataFrame (or dict of scalars) with FEATURE_COLUMNS
    side   : +1 LONG / -1 SHORT (scalar or array)
    atr_pct: ATR / price * 100 at the decision time
    Returns DataFrame with columns == MODEL_FEATURES, in that order.
    """
    import pandas as pd
    df = pd.DataFrame([feats]) if isinstance(feats, dict) else feats.copy()
    df = df.reset_index(drop=True)
    sd = np.broadcast_to(np.asarray(side, dtype=float), (len(df),))
    out = pd.DataFrame(index=df.index)
    for col in DIRECTIONAL_FEATURES:
        out[col] = df[col].astype(float).to_numpy() * sd
    for col in NEUTRAL_FEATURES:
        out[col] = df[col].astype(float).to_numpy()
    out["side"] = sd
    out["atr_pct"] = np.broadcast_to(np.asarray(atr_pct, dtype=float), (len(df),))
    return out[MODEL_FEATURES]
