# ml/metrics.py
"""Classification + economic metrics. The economic one mirrors the repo's
philosophy: win rate vs the breakeven win rate the barrier geometry and costs
require, with a bootstrap confidence interval."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support, roc_auc_score


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, objective: str) -> dict:
    out: dict = {}
    if objective == "multiclass":
        labels = [0, 1, 2]
        pred = proba.argmax(1)
        out["logloss"] = float(log_loss(y_true, proba, labels=labels))
        p, r, _, _ = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
        for k, name in enumerate(["stop", "timeout", "target"]):
            out[f"precision_{name}"], out[f"recall_{name}"] = float(p[k]), float(r[k])
        pos = proba[:, 2]
        yb = (y_true == 2).astype(int)
    else:
        pred = (proba >= 0.5).astype(int)
        out["logloss"] = float(log_loss(y_true, proba, labels=[0, 1]))
        p, r, _, _ = precision_recall_fscore_support(y_true, pred, labels=[1], zero_division=0)
        out["precision_target"], out["recall_target"] = float(p[0]), float(r[0])
        pos, yb = proba, y_true
    out["accuracy"] = float(accuracy_score(y_true, pred))
    out["auc_target"] = float(roc_auc_score(yb, pos)) if 0 < yb.sum() < len(yb) else float("nan")
    return out


def economic_metrics(y: np.ndarray, ret_atr: np.ndarray, cost_atr: np.ndarray, mask: np.ndarray,
                     tp: float, sl: float, t0: np.ndarray | None = None, block: int = 1,
                     n_boot: int = 500, seed: int = 0, net_r: np.ndarray | None = None) -> dict:
    """
    y        : barrier labels in {-1,0,1}
    mask     : which samples the gate would TAKE
    win rate : fraction with target hit first (time expiries count as non-wins)
    breakeven: mean over taken of (sl+cost)/(tp+sl) - the win rate needed after costs
    expectancy_r: mean net result in units of risk (stop distance), incl. expiries
    net_r    : if given ("realised" mode, used for live_trail labels where winners and losers
               have different sizes): expectancy = mean(net_r); win/loss sizes W, L come from the
               taken set; breakeven WR = L/(W+L); edge = win_rate - breakeven = expectancy/(W+L).
    CI       : moving-block bootstrap of `edge` (block ~ label horizon in samples,
               because overlapping labels make i.i.d. bootstrap too optimistic)
    """
    n_all = len(y)
    m = np.asarray(mask, bool)
    n = int(m.sum())
    res = {"n_taken": n, "take_rate": n / n_all if n_all else float("nan")}
    if n == 0:
        res.update(win_rate=float("nan"), breakeven_wr=float("nan"), edge=float("nan"),
                   expectancy_r=float("nan"), edge_ci_low=float("nan"), edge_ci_high=float("nan"),
                   timeout_rate=float("nan"))
        return res
    order = np.argsort(t0[m], kind="stable") if t0 is not None else np.arange(n)
    win = (y[m] == 1).astype(float)[order]
    if net_r is not None:
        nr = np.asarray(net_r, float)[m][order]
        yy = y[m][order]
        W = float(nr[yy == 1].mean()) if (yy == 1).any() else float("nan")
        L = float((-nr[yy != 1]).mean()) if (yy != 1).any() else float("nan")
        scale = W + L
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.abs(nr).mean()) or 1.0
        d = nr / scale
        be_v = float(win.mean() - d.mean())
        res.update(win_rate=float(win.mean()), breakeven_wr=be_v, edge=float(d.mean()),
                   expectancy_r=float(nr.mean()), timeout_rate=float((yy == 0).mean()),
                   avg_win_r=W, avg_loss_r=L)
    else:
        be = ((sl + cost_atr[m]) / (tp + sl))[order]
        net = ((ret_atr[m] - cost_atr[m]) / sl)[order]
        d = win - be
        res.update(win_rate=float(win.mean()), breakeven_wr=float(be.mean()),
                   edge=float(d.mean()), expectancy_r=float(net.mean()),
                   timeout_rate=float((y[m] == 0).mean()))
    rng = np.random.default_rng(seed)
    nb = max(1, int(np.ceil(n / block)))
    edges = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        edges.append(d[np.minimum(idx, n - 1)].mean())
    res["edge_ci_low"], res["edge_ci_high"] = (float(np.percentile(edges, 2.5)),
                                               float(np.percentile(edges, 97.5)))
    return res


def threshold_table(p: np.ndarray, y, ret_atr, cost_atr, tp, sl, t0, block,
                    thresholds=(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7), net_r=None, cand=None) -> dict:
    out = {}
    for t in thresholds:
        e = economic_metrics(y, ret_atr, cost_atr, p > t, tp, sl, t0, block, n_boot=200, net_r=net_r)
        if cand is not None:
            e["n_taken_candidates"] = int(((p > t) & (np.asarray(cand) == 1)).sum())
        out[f"{t:.2f}"] = e
    return out
