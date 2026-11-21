# ml/train.py
"""
Phase 3 + 5: LightGBM training with purged walk-forward CV, an economic
(win rate vs breakeven) evaluation, MLflow tracking, and champion/challenger
promotion.

    python -m ml.train --lookback-days 60
    python -m ml.train --start 2026-09-01 --end 2026-11-15 --objective binary \
        --tp-atr 1.5 --sl-atr 1.0 --max-hold-bars 6 --threshold 0.58

Pipeline: feature store -> triple-barrier labels (side-expanded) -> time-based
HOLDOUT (last --holdout-frac, purged) -> purged walk-forward CV on the rest ->
refit -> evaluate on holdout -> save bundle -> MLflow -> promote only if the
holdout economic metric is not worse than the current champion by > tolerance.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import settings
from features.orderflow import MODEL_FEATURES
from ml.bundle import (compatible, copy_bundle_atomic, load_bundle, predict_favorable, save_bundle)
from ml.dataset import build_dataset, target
from ml.exit_sim import ExitRules
from ml.labeling import BarrierConfig, PurgedWalkForward, purge_before
from ml.metrics import classification_metrics, economic_metrics, threshold_table

log = logging.getLogger("ml.train")


# ------------------------------------------------------------------- CLI ---
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m ml.train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--store", default=settings.FEATURE_STORE_PATH)
    g.add_argument("--symbols", default=",".join(settings.FEATURE_SYMBOLS))
    g.add_argument("--start", help="UTC date YYYY-MM-DD")
    g.add_argument("--end", help="UTC date YYYY-MM-DD (inclusive)")
    g.add_argument("--lookback-days", type=float, help="alternative to --start: last N days")
    g = p.add_argument_group("labels")
    g.add_argument("--exit-mode", choices=["live_trail", "static_barrier"], default="live_trail",
                   help="live_trail (default): label = outcome of a simulated trade with the LIVE "
                        "PaperTrader exit rules (settings.py). static_barrier: research-only triple barrier "
                        "using --tp-atr/--sl-atr/--max-hold-bars")
    g.add_argument("--exit-atr-mult", type=float, help="override live stop/trail multiple (breaks parity!)")
    g.add_argument("--exit-hold-bars", type=int, help="override live time stop (breaks parity!)")
    g.add_argument("--be-band-r", type=float, default=0.10,
                   help="live_trail: |net R| below this is labelled breakeven (0)")
    g.add_argument("--objective", choices=["multiclass", "binary"], default="multiclass")
    g.add_argument("--tp-atr", type=float, default=1.5, help="static_barrier only")
    g.add_argument("--sl-atr", type=float, default=1.0)
    g.add_argument("--max-hold-bars", type=int, default=6)
    g.add_argument("--bar-s", type=int, default=settings.INTERVAL_MS // 1000)
    g.add_argument("--atr-window", type=int, default=settings.TREND_ATR_WINDOW)
    g.add_argument("--snapshot-s", type=int, default=settings.FEATURE_SNAPSHOT_S)
    g.add_argument("--stride-s", type=int, default=1800,
                   help="static_barrier sample spacing; live_trail always samples bar closes")
    g.add_argument("--cost-pct", type=float,
                   default=2 * settings.PAPER_FEE_TAKER_PCT + 2 * settings.PAPER_SLIPPAGE_PCT,
                   help="round-trip fees+slippage, %% of price")
    g.add_argument("--sample-mode", choices=["all", "candidates"], default="all",
                   help="all: every stride point, both sides; candidates: only Donchian "
                        "breakout bar-closes (what the live Bucket A proposes)")
    g = p.add_argument_group("validation")
    g.add_argument("--n-splits", type=int, default=4)
    g.add_argument("--embargo-bars", type=float, default=1.0)
    g.add_argument("--holdout-frac", type=float, default=0.2)
    g.add_argument("--threshold", type=float, default=settings.ML_GATE_THRESHOLD)
    g = p.add_argument_group("lightgbm")
    g.add_argument("--learning-rate", type=float, default=0.03)
    g.add_argument("--num-leaves", type=int, default=15)
    g.add_argument("--max-depth", type=int, default=4)
    g.add_argument("--min-child-samples", type=int, default=100)
    g.add_argument("--feature-fraction", type=float, default=0.7)
    g.add_argument("--bagging-fraction", type=float, default=0.8)
    g.add_argument("--lambda-l2", type=float, default=10.0)
    g.add_argument("--n-estimators", type=int, default=1000)
    g.add_argument("--early-stopping", type=int, default=50)
    g.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="balanced distorts probabilities; the gate threshold assumes calibrated P")
    g.add_argument("--seed", type=int, default=42)
    g = p.add_argument_group("mlflow / registry")
    g.add_argument("--no-mlflow", action="store_true")
    g.add_argument("--tracking-uri", default=settings.MLFLOW_TRACKING_URI or None)
    g.add_argument("--experiment", default="orderflow_lgbm")
    g.add_argument("--registered-model", default=settings.ML_REGISTERED_MODEL)
    g.add_argument("--out-dir", default=os.path.dirname(settings.ML_MODEL_PATH.rstrip("/\\")) or "models")
    g = p.add_argument_group("promotion gate")
    g.add_argument("--no-promote", action="store_true", help="train + log, never touch champion")
    g.add_argument("--promotion-metric", choices=["auto", "edge", "expectancy_r"], default="auto",
                   help="auto: expectancy_r for live_trail, edge for static_barrier")
    g.add_argument("--tolerance", type=float, default=0.01,
                   help="challenger may be at most this much worse than champion")
    g.add_argument("--min-metric", type=float, default=0.0,
                   help="absolute floor on the holdout metric for the FIRST champion")
    g.add_argument("--min-holdout-trades", type=int, default=settings.MIN_HOLDOUT_TRADES,
                   help="below this the challenger is NOT promoted: status needs_more_data")
    g.add_argument("--force-promote", action="store_true")
    return p.parse_args(argv)


def _ms(d: str, end: bool = False) -> int:
    t = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(t.timestamp() * 1000) + (86_400_000 - 1 if end else 0)


# --------------------------------------------------------------- helpers ---
def lgb_params(a, n_class: int) -> dict:
    p = dict(learning_rate=a.learning_rate, num_leaves=a.num_leaves, max_depth=a.max_depth,
             min_data_in_leaf=a.min_child_samples, feature_fraction=a.feature_fraction,
             bagging_fraction=a.bagging_fraction, bagging_freq=1, lambda_l2=a.lambda_l2,
             seed=a.seed, verbosity=-1, num_threads=0, deterministic=True, force_row_wise=True)
    if a.objective == "multiclass":
        p.update(objective="multiclass", num_class=3, metric="multi_logloss")
    else:
        p.update(objective="binary", metric="binary_logloss")
    return p


def _weights(y: np.ndarray, mode: str):
    if mode != "balanced":
        return None
    cnt = np.bincount(y)
    w = len(y) / (len(cnt) * np.maximum(cnt, 1))
    return w[y]


def fit(Xtr, ytr, params, n_estimators, Xva=None, yva=None, early=50, cw="none"):
    dtr = lgb.Dataset(Xtr, ytr, weight=_weights(ytr, cw), feature_name=list(Xtr.columns))
    if Xva is None:
        return lgb.train(params, dtr, num_boost_round=n_estimators)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=n_estimators, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(early, verbose=False)])


def inner_split(t0, t1, idx, embargo_s, frac=0.2):
    """Last `frac` of idx by time = validation; training rows purged against it."""
    idx = idx[np.argsort(t0[idx], kind="stable")]
    cut = t0[idx[int(len(idx) * (1 - frac))]]
    va = idx[t0[idx] >= cut]
    tr = idx[(t0[idx] < cut) & (t1[idx] < cut - embargo_s * 1000)]
    return tr, va


def p_fav(p, objective):
    return p[:, 2] if objective == "multiclass" else p


def eco(ds, p, thr, cfg, block):
    nr = ds["net_r"].to_numpy() if cfg.exit_mode == "live_trail" else None
    m = economic_metrics(ds["y"].to_numpy(), ds["ret_atr"].to_numpy(), ds["cost_atr"].to_numpy(),
                         p > thr, cfg.tp_atr, cfg.sl_atr, ds["ts_ms"].to_numpy(), block, net_r=nr)
    m["n_taken_candidates"] = int(((p > thr) & (ds["cand"].to_numpy() == 1)).sum())
    return m


def table(ds, p, cfg, block):
    return threshold_table(p, ds["y"].to_numpy(), ds["ret_atr"].to_numpy(), ds["cost_atr"].to_numpy(),
                           cfg.tp_atr, cfg.sl_atr, ds["ts_ms"].to_numpy(), block,
                           net_r=ds["net_r"].to_numpy() if cfg.exit_mode == "live_trail" else None,
                           cand=ds["cand"].to_numpy())


def decide_promotion(chall: dict, champ: dict | None, metric: str, tol: float, min_metric: float,
                     min_trades: int, force: bool = False) -> tuple[bool, str]:
    """Pure function, unit-tested. chall/champ are holdout economic-metric dicts
    evaluated on the SAME holdout."""
    if force:
        return True, "forced"
    if chall["n_taken"] < min_trades:
        return False, f"challenger took only {chall['n_taken']} holdout trades (< {min_trades})"
    c = chall[metric]
    if not np.isfinite(c):
        return False, f"challenger {metric} is not finite"
    if champ is None or not np.isfinite(champ.get(metric, float("nan"))):
        if c >= min_metric:
            return True, f"no comparable champion; {metric}={c:+.4f} >= floor {min_metric}"
        return False, f"no comparable champion and {metric}={c:+.4f} < floor {min_metric}"
    h = champ[metric]
    if c >= h - tol:
        return True, f"{metric} {c:+.4f} >= champion {h:+.4f} - tol {tol}"
    return False, f"{metric} {c:+.4f} < champion {h:+.4f} - tol {tol} (silent-degradation guard)"


def _flat(d: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": float(v) for k, v in d.items()
            if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v)}


# ------------------------------------------------------------------ main ---
def run(a: argparse.Namespace) -> dict:
    ov = {k: v for k, v in (("atr_mult", a.exit_atr_mult), ("max_hold_bars", a.exit_hold_bars)) if v is not None}
    rules = ExitRules.from_settings(be_band_r=a.be_band_r, **ov)
    parity = rules.parity_key() == ExitRules.from_settings().parity_key()
    if a.exit_mode == "live_trail" and not parity:
        log.warning("exit rules overridden (%s): model will NOT match live exits and the live gate will "
                    "refuse it unless ML_ALLOW_EXIT_MISMATCH=true", ov)
    cfg = BarrierConfig(tp_atr=a.tp_atr, sl_atr=a.sl_atr, max_hold_bars=a.max_hold_bars,
                        bar_s=a.bar_s, atr_window=a.atr_window, snapshot_s=a.snapshot_s,
                        stride_s=a.stride_s, cost_pct=a.cost_pct, sample_mode=a.sample_mode,
                        exit_mode=a.exit_mode, rules=rules)
    if a.exit_mode == "live_trail":
        cfg.atr_window = rules.atr_window
    now = datetime.now(timezone.utc)
    start_ms = _ms(a.start) if a.start else (
        int((now.timestamp() - a.lookback_days * 86400) * 1000) if a.lookback_days else None)
    end_ms = _ms(a.end, True) if a.end else None
    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()] or None

    ds = build_dataset(a.store, symbols, start_ms, end_ms, cfg)
    t0, t1 = ds["ts_ms"].to_numpy(), ds["t1_ms"].to_numpy()
    X, y = ds[MODEL_FEATURES], target(ds, a.objective)
    n_class = 3 if a.objective == "multiclass" else 2
    if cfg.exit_mode == "live_trail":
        block = max(1, int(np.ceil(np.median(ds["held"]))) * 2)     # a trade's life, both sides interleaved
    else:
        block = max(1, int(cfg.max_hold_bars * cfg.bar_s / cfg.stride_s))
    metric = a.promotion_metric if a.promotion_metric != "auto" else (
        "expectancy_r" if cfg.exit_mode == "live_trail" else "edge")
    embargo_s = int(a.embargo_bars * cfg.bar_s)
    params = lgb_params(a, n_class)
    log.info("dataset: %d rows, %s -> %s, label mix %s", len(ds),
             pd.to_datetime(t0.min(), unit="ms", utc=True), pd.to_datetime(t0.max(), unit="ms", utc=True),
             dict(zip(*np.unique(ds["y"], return_counts=True))))

    # ---- time-based holdout
    boundary = int(t0[int(len(ds) * (1 - a.holdout_frac))])
    hold = np.flatnonzero(t0 >= boundary)
    dev = purge_before(t0, t1, boundary, embargo_s)
    if len(dev) < 500 or len(hold) < 50:
        raise RuntimeError(f"not enough data: dev={len(dev)} holdout={len(hold)} rows - "
                           "record more data or lower --max-hold-bars / --stride-s")

    # ---- purged walk-forward CV on dev
    cv = PurgedWalkForward(a.n_splits, embargo_s)
    folds, best_iters, oof_idx, oof_p = [], [], [], []
    for k, (tr, te) in enumerate(cv.split(t0[dev], t1[dev])):
        tr, te = dev[tr], dev[te]
        itr, iva = inner_split(t0, t1, tr, embargo_s)
        if len(itr) < 200 or len(iva) < 50:
            log.warning("fold %d skipped: train too small", k)
            continue
        m = fit(X.iloc[itr], y[itr], params, a.n_estimators, X.iloc[iva], y[iva], a.early_stopping,
                a.class_weight)
        bi = m.best_iteration or a.n_estimators
        p = m.predict(X.iloc[te], num_iteration=bi)
        cm = classification_metrics(y[te], p, a.objective)
        em = eco(ds.iloc[te], p_fav(p, a.objective), a.threshold, cfg, block)
        base = eco(ds.iloc[te], np.ones(len(te)), -1, cfg, block)           # ungated: take everything
        folds.append({"fold": k, "n_train": len(itr), "n_test": len(te), "best_iter": bi,
                      **{f"cls_{x}": v for x, v in cm.items()},
                      **{f"eco_{x}": v for x, v in em.items()},
                      "base_win_rate": base["win_rate"], "base_edge": base["edge"]})
        best_iters.append(bi)
        oof_idx.append(te)
        oof_p.append(p_fav(p, a.objective))
        log.info("fold %d: n_test=%d logloss=%.4f AUC=%.3f | gate@%.2f took %d WR=%.3f BE=%.3f edge=%+.4f",
                 k, len(te), cm["logloss"], cm["auc_target"], a.threshold, em["n_taken"],
                 em["win_rate"], em["breakeven_wr"], em["edge"])
    if not folds:
        raise RuntimeError("no usable CV folds")
    fdf = pd.DataFrame(folds)
    cv_summary = {c: float(np.nanmean(fdf[c])) for c in fdf.columns if c not in ("fold",)}
    oi = np.concatenate(oof_idx)
    op = np.concatenate(oof_p)
    oof_eco = eco(ds.iloc[oi], op, a.threshold, cfg, block)
    oof_table = table(ds.iloc[oi], op, cfg, block)

    # ---- refit on all dev rows (iterations from CV), evaluate on holdout
    n_final = max(50, int(np.median(best_iters) * 1.1))
    booster = fit(X.iloc[dev], y[dev], params, n_final, cw=a.class_weight)
    ph = booster.predict(X.iloc[hold])
    hold_cls = classification_metrics(y[hold], ph, a.objective)
    hold_eco = eco(ds.iloc[hold], p_fav(ph, a.objective), a.threshold, cfg, block)
    hold_base = eco(ds.iloc[hold], np.ones(len(hold)), -1, cfg, block)
    hold_table = table(ds.iloc[hold], p_fav(ph, a.objective), cfg, block)
    # ---- trade-count report (Phase 6.3)
    cls_ct = {int(k): int(v) for k, v in zip(*np.unique(ds["y"], return_counts=True))}
    report_lines = [
        f"TRADE-COUNT REPORT: labelled rows={len(ds)} (loss/flat/win={cls_ct.get(-1, 0)}/{cls_ct.get(0, 0)}/{cls_ct.get(1, 0)}), "
        f"dev={len(dev)}, holdout={len(hold)}, live-candidate rows in holdout={int(ds['cand'].to_numpy()[hold].sum())}",
        "  threshold -> holdout trades (all rows / live-candidate rows) | win rate | net R"]
    for k, v in hold_table.items():
        report_lines.append(f"  P>{k}: {v['n_taken']:5d} / {v['n_taken_candidates']:4d} | "
                            f"{v['win_rate']:.3f} | {v['expectancy_r']:+.3f}R" if v["n_taken"] else
                            f"  P>{k}:     0 /    0 |   -   |   -")
    log.info("\n".join(report_lines))
    if hold_eco["n_taken"] < a.min_holdout_trades:
        feasible = [k for k, v in hold_table.items() if v["n_taken"] >= a.min_holdout_trades]
        log.warning("holdout takes %d trades at P>%.2f (< floor %d). Thresholds that would reach the floor: %s. "
                    "Other levers: more symbols (FEATURE_SYMBOLS), longer history, 1h bars.",
                    hold_eco["n_taken"], a.threshold, a.min_holdout_trades, feasible or "none")
    log.info("HOLDOUT n=%d | gate@%.2f took %d: WR=%.3f vs breakeven %.3f -> edge %+.4f "
             "(95%% CI %+.4f..%+.4f), net %+.3fR | ungated edge %+.4f",
             len(hold), a.threshold, hold_eco["n_taken"], hold_eco["win_rate"], hold_eco["breakeven_wr"],
             hold_eco["edge"], hold_eco["edge_ci_low"], hold_eco["edge_ci_high"],
             hold_eco["expectancy_r"], hold_base["edge"])

    imp = pd.DataFrame({"feature": MODEL_FEATURES,
                        "gain": booster.feature_importance("gain")}).sort_values("gain", ascending=False)

    run_name = "lgbm-" + now.strftime("%Y%m%dT%H%M%S")
    bundle_dir = os.path.join(a.out_dir, "runs", run_name)
    manifest = {
        "model_version": run_name, "created_utc": now.isoformat(),
        "feature_names": MODEL_FEATURES,
        "feature_types": {c: "float64" for c in MODEL_FEATURES},
        "objective": a.objective, "classes": [-1, 0, 1] if a.objective == "multiclass" else [0, 1],
        "favorable_index": 2 if a.objective == "multiclass" else None,
        "default_threshold": a.threshold, "barrier": cfg.to_dict(), "lgb_params": params,
        "exit_mode": a.exit_mode, "exit_rules": rules.parity_key() if a.exit_mode == "live_trail" else None,
        "exit_parity_with_settings": bool(parity and a.exit_mode == "live_trail"),
        "trade_count_report": {"threshold_table_holdout": hold_table, "class_counts": cls_ct},
        "n_estimators": n_final,
        "data": {"symbols": symbols, "n_rows": len(ds), "n_dev": int(len(dev)), "n_holdout": int(len(hold)),
                 "from_utc": str(pd.to_datetime(t0.min(), unit="ms", utc=True)),
                 "to_utc": str(pd.to_datetime(t0.max(), unit="ms", utc=True)),
                 "holdout_from_utc": str(pd.to_datetime(boundary, unit="ms", utc=True))},
        "holdout": {"classification": hold_cls, "economic": hold_eco, "ungated": hold_base},
        "cv": {"oof_economic": oof_eco},
        "lightgbm_version": lgb.__version__,
    }
    save_bundle(booster, manifest, bundle_dir)
    imp.to_csv(os.path.join(bundle_dir, "feature_importance.csv"), index=False)
    with open(os.path.join(bundle_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"folds": folds, "cv_summary": cv_summary, "oof_thresholds": oof_table,
                   "holdout_thresholds": hold_table}, f, indent=2, default=float)

    # ---- champion / challenger
    champ_dir = settings.ML_MODEL_PATH if a.out_dir == os.path.dirname(
        settings.ML_MODEL_PATH.rstrip("/\\")) else os.path.join(a.out_dir, "champion")
    mlflow = client = run_id = None
    if not a.no_mlflow:
        from ml import registry
        mlflow, client = registry.setup(a.tracking_uri, a.experiment)
    champ_eco, champ_ver, champ_src = None, None, "none"
    try:
        champ_booster = champ_man = None
        if client is not None:
            mv = registry.champion(client, a.registered_model)
            if mv is not None:
                got = registry.download_champion(a.tracking_uri, a.registered_model,
                                                 os.path.join(a.out_dir, "_champion_cache"))
                champ_booster, champ_man = load_bundle(got[0])
                champ_ver, champ_src = got[1], "registry"
        if champ_booster is None and os.path.exists(os.path.join(champ_dir, "manifest.json")):
            champ_booster, champ_man = load_bundle(champ_dir)
            champ_ver, champ_src = "local", "local"
        if champ_booster is not None and compatible(champ_man):
            pc = predict_favorable(champ_booster, champ_man, X.iloc[hold])
            champ_eco = eco(ds.iloc[hold], pc, a.threshold, cfg, block)
            log.info("CHAMPION (%s v%s) on same holdout: took %d, edge %+.4f, net %+.3fR",
                     champ_src, champ_ver, champ_eco["n_taken"], champ_eco["edge"], champ_eco["expectancy_r"])
        elif champ_booster is not None:
            log.warning("champion feature schema differs from current code - not comparable")
    except Exception as exc:
        log.warning("could not evaluate champion (%s) - treating as none", exc)

    promote, reason = (False, "promotion disabled (--no-promote)") if a.no_promote else decide_promotion(
        hold_eco, champ_eco, metric, a.tolerance, a.min_metric, a.min_holdout_trades,
        a.force_promote)
    status = ("needs_more_data" if hold_eco["n_taken"] < a.min_holdout_trades and not a.force_promote
              else "promoted" if promote else "kept_champion")
    log.info("PROMOTE=%s - %s", promote, reason)

    version = None
    if mlflow is not None:
        with mlflow.start_run(run_name=run_name) as r:
            run_id = r.info.run_id
            mlflow.log_params({**{f"lgb_{k}": v for k, v in params.items()},
                               **{f"label_{k}": v for k, v in cfg.to_dict().items()},
                               "objective": a.objective, "threshold": a.threshold, "n_splits": a.n_splits,
                               "holdout_frac": a.holdout_frac, "embargo_bars": a.embargo_bars,
                               "n_estimators_final": n_final, "symbols": ",".join(symbols or []),
                               "n_rows": len(ds), "class_weight": a.class_weight})
            mlflow.log_metrics({**{f"cv_{k}": v for k, v in cv_summary.items() if np.isfinite(v)},
                                **_flat(oof_eco, "oof_eco"), **_flat(hold_cls, "holdout_cls"),
                                **_flat(hold_eco, "holdout_eco"), **_flat(hold_base, "holdout_ungated")})
            for _, row in imp.iterrows():
                mlflow.log_metric(f"importance_gain.{row['feature']}", float(row["gain"]))
            mlflow.log_artifacts(bundle_dir, "bundle")
            mlflow.set_tags({"promoted": str(promote), "promotion_reason": reason,
                             "champion_before": str(champ_ver)})
        version = registry.register(mlflow, client, run_id, a.registered_model).version
        client.set_model_version_tag(a.registered_model, str(version), "holdout_edge",
                                     f"{hold_eco['edge']:.4f}")
        if promote:
            registry.promote(client, a.registered_model, version)
            log.info("registered model %s v%s -> alias 'champion'", a.registered_model, version)
    if promote:
        copy_bundle_atomic(bundle_dir, champ_dir)
        log.info("local champion updated: %s", champ_dir)

    return {"status": status, "run_name": run_name, "bundle_dir": bundle_dir, "mlflow_run_id": run_id,
            "registry_version": version, "promoted": promote, "reason": reason,
            "holdout": hold_eco, "holdout_ungated": hold_base, "champion_holdout": champ_eco,
            "cv_summary": cv_summary, "oof": oof_eco, "holdout_table": hold_table}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    a = parse_args(argv)
    try:
        res = run(a)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2
    h = res["holdout"]
    print(json.dumps({k: res[k] for k in ("run_name", "bundle_dir", "registry_version", "promoted", "reason")},
                     indent=2))
    print(f"holdout: took {h['n_taken']} | win rate {h['win_rate']:.3f} vs breakeven "
          f"{h['breakeven_wr']:.3f} (edge {h['edge']:+.4f}, CI {h['edge_ci_low']:+.4f}..{h['edge_ci_high']:+.4f}) "
          f"| net {h['expectancy_r']:+.3f}R")
    return 0


if __name__ == "__main__":
    sys.exit(main())
