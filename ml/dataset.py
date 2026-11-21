# ml/dataset.py
"""Feature store -> labelled, side-expanded model dataset (shared by train/retrain)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from features.orderflow import FEATURE_COLUMNS, MODEL_FEATURES, to_model_frame
from features.store import FeatureStore
from ml.labeling import BarrierConfig, make_labels

log = logging.getLogger(__name__)


def build_dataset(store_path: str, symbols: list[str] | None, start_ms: int | None,
                  end_ms: int | None, cfg: BarrierConfig) -> pd.DataFrame:
    """
    One row per (symbol, decision time, side). Columns: symbol, ts_ms, side, y, ret_atr,
    cost_atr, t1_ms, held, atr, atr_pct, plus MODEL_FEATURES (sign-adjusted for side).
    Rows are sorted by ts_ms.
    """
    store = FeatureStore(store_path)
    snaps = store.read(symbols, start_ms, end_ms)
    if snaps.empty:
        raise RuntimeError(f"feature store {store_path!r} has no rows for the requested range")
    parts = []
    for sym, g in snaps.groupby("symbol"):
        g = g.sort_values("ts_ms").reset_index(drop=True)
        lab = make_labels(g, cfg, sym)
        if lab.empty:
            log.warning("%s: no labelled samples (rows=%d)", sym, len(g))
            continue
        feats = g.iloc[lab["row"].to_numpy()][FEATURE_COLUMNS]
        X = to_model_frame(feats, lab["side"].to_numpy(), lab["atr_pct"].to_numpy())
        parts.append(pd.concat([lab.drop(columns=["row", "side", "atr_pct"]).reset_index(drop=True), X], axis=1))
        log.info("%s: %d snapshots -> %d labelled rows", sym, len(g), len(lab))
    if not parts:
        raise RuntimeError("no labelled samples produced; need more recorded data "
                           "(ATR warm-up + max-hold horizon + feature warm-up)")
    ds = pd.concat(parts, ignore_index=True).sort_values("ts_ms", kind="stable").reset_index(drop=True)
    return ds


def target(ds: pd.DataFrame, objective: str) -> np.ndarray:
    """multiclass: classes {-1,0,1} -> {0,1,2}; binary: target-first vs not."""
    return (ds["y"].to_numpy() + 1).astype(int) if objective == "multiclass" \
        else (ds["y"].to_numpy() == 1).astype(int)


__all__ = ["build_dataset", "target", "MODEL_FEATURES"]
