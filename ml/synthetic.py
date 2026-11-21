# ml/synthetic.py
"""
SYNTHETIC market-event generator - for tests and pipeline smoke runs ONLY.

A latent slow "pressure" process drives both order-flow imbalance (book sizes,
aggressor mix) and, with a lag, price drift. The model can therefore find a
planted edge. Results on this data say nothing about real markets; it exists to
prove that features -> labels -> training -> gate wiring work end to end.
"""
from __future__ import annotations

import numpy as np

from features.orderflow import OrderFlowFeatureEngine, OrderFlowConfig
from features.store import FeatureStore


def generate_events(days: float = 3.0, seed: int = 0, start_ms: int = 1_700_000_000_000,
                    price0: float = 30_000.0, edge: float = 0.35, step_s: float = 2.0):
    """Yield ('t', ts, price, qty, is_buyer_maker) / ('b', ts, bids, asks) in time order."""
    rng = np.random.default_rng(seed)
    n = int(days * 86400 / step_s)
    pressure = 0.0
    price = price0
    for i in range(n):
        ts = start_ms + int(i * step_s * 1000)
        pressure = 0.995 * pressure + rng.normal(0, 0.12)           # slow latent factor
        p_buy = 1 / (1 + np.exp(-1.5 * pressure))
        # price: pressure drives drift (edge) + noise
        ret = edge * 1e-4 * pressure + rng.normal(0, 1.2e-4)
        price *= 1 + ret
        # a few trades per step
        for k in range(rng.poisson(1.6)):
            qty = float(rng.lognormal(-3.0, 1.0))
            buyer_maker = rng.random() > p_buy
            yield ("t", ts + k * 40, price * (1 + rng.normal(0, 1e-6)), qty, bool(buyer_maker))
        # book: sizes tilt with pressure
        tilt = np.exp(0.8 * pressure)
        bids = [(price - 0.5 * (j + 1), float(rng.lognormal(0.5, 0.3) * tilt)) for j in range(5)]
        asks = [(price + 0.5 * (j + 1), float(rng.lognormal(0.5, 0.3) / tilt)) for j in range(5)]
        yield ("b", ts + 500, bids, asks)


def build_synthetic_store(root: str, symbol: str = "SYNTHUSDT", days: float = 3.0, seed: int = 0,
                          snapshot_s: int = 60, edge: float = 0.35) -> int:
    """Run the REAL feature engine over synthetic events and persist the snapshots."""
    store = FeatureStore(root)
    eng = OrderFlowFeatureEngine(symbol, OrderFlowConfig(snapshot_s=snapshot_s), sink=store.append)
    for ev in generate_events(days=days, seed=seed, edge=edge):
        if ev[0] == "t":
            eng.on_trade(ev[1], ev[2], ev[3], ev[4])
        else:
            eng.on_book(ev[1], ev[2], ev[3])
    return store.flush()
