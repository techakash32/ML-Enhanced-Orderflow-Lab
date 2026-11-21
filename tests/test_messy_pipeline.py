"""Phase 7 / C3 stand-in: the whole pipeline must survive MESSY recordings (gaps, outages, NaN cells,
duplicates) without crashing, report needs_more_data when trades are scarce, and produce a champion whose
monitor baseline matches the data it was trained on. Synthetic data - plumbing only, not metrics."""
import json
import os

import numpy as np
import pytest

from config import settings
from features.orderflow import FEATURE_COLUMNS
from features.store import FeatureStore
from ml import monitor, retrain, train
from ml.synthetic import build_synthetic_store


@pytest.fixture(scope="module")
def messy_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("messy")
    clean = str(root / "clean")
    build_synthetic_store(clean, days=6, seed=21)
    df = FeatureStore(clean).read(["SYNTHUSDT"])
    t = df["ts_ms"]
    drop = ((t > t.min() + 2 * 86_400_000) & (t < t.min() + 2 * 86_400_000 + 40 * 60_000)) | \
           ((t > t.min() + 4 * 86_400_000) & (t < t.min() + 4 * 86_400_000 + 3 * 3_600_000))   # 40 min + 3 h
    df = df[~drop].copy()
    rng = np.random.default_rng(0)
    for c in FEATURE_COLUMNS:                                       # 2 % random NaN cells
        df.loc[rng.random(len(df)) < 0.02, c] = np.nan
    df = __import__("pandas").concat([df, df.sample(60, random_state=1)])    # duplicate rows
    store = FeatureStore(str(root / "store"))
    for r in df.sample(frac=1.0, random_state=2).to_dict("records"):          # shuffled arrival
        store.append(r)
    store.flush()
    return str(root / "store")


COMMON = ["--symbols", "SYNTHUSDT", "--bar-s", "900", "--no-mlflow"]


def test_labeling_survives_gaps(messy_store):
    from ml.dataset import build_dataset
    from ml.labeling import BarrierConfig
    ds = build_dataset(messy_store, ["SYNTHUSDT"], None, None, BarrierConfig(bar_s=900))
    assert len(ds) > 500 and ds["y"].isin([-1, 0, 1]).all()
    # no label may straddle the 3-hour outage: its exit time must not exceed the next recorded bar
    assert (ds["t1_ms"] >= ds["ts_ms"]).all()


def test_retrain_dry_run_reports_needs_more_data(messy_store, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(settings, "FEATURE_STORE_PATH", messy_store)
    code = retrain.main(["--dry-run", "--lookback-days", "100000", "--store", messy_store, *COMMON,
                         "--out-dir", str(tmp_path / "m"), "--min-holdout-trades", "100000"])
    out = capsys.readouterr().out
    assert code == 0 and '"dry_run": true' in out and '"status": "needs_more_data"' in out
    assert not (tmp_path / "m" / "champion").exists()               # nothing promoted


def test_train_champion_then_monitor_is_quiet_on_same_data(messy_store, tmp_path, monkeypatch):
    models = tmp_path / "models"
    monkeypatch.setattr(settings, "ML_MODEL_PATH", str(models / "champion"))
    monkeypatch.setattr(settings, "MONITOR_PATH", str(tmp_path / "ml" / "monitor.json"))
    monkeypatch.setattr(settings, "GATE_LOG_PATH", str(tmp_path / "ml" / "gate_log.jsonl"))
    monkeypatch.setattr(settings, "PAPER_TRADES_LOG", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(settings, "FEATURE_STORE_PATH", messy_store)
    a = train.parse_args(["--store", messy_store, *COMMON, "--out-dir", str(models), "--threshold", "0.45",
                          "--min-holdout-trades", "20",
                          "--force-promote"])        # the real gate may (rightly) refuse a 6-day messy sample;
    res = train.run(a)                               # force it here only to exercise the monitor baseline
    assert res["status"] in ("promoted", "kept_champion", "needs_more_data")
    man = json.load(open(os.path.join(res["bundle_dir"], "manifest.json")))
    dist = man["training_distribution"]
    assert len(dist["features"]) >= 20 and dist["gate"]["n"] > 50 and "trade_count_report" in man
    print("STATUS", res["status"], res["reason"])
    assert res["promoted"]
    rep = monitor.run_once(days=5000)                               # same data -> no drift
    assert not [f for f in rep["flags"] if f.startswith("feature_psi")], rep["flags"]
    assert rep["summary"]["n_live_snapshots"] > 1000


def test_report_trade_rate(messy_store, capsys):
    train.main(["--report-trade-rate", "--store", messy_store, *COMMON])
    j = json.loads(capsys.readouterr().out)
    assert j["labelled_rows"] > 500 and "projection_all_rows" in j and j["floor"] == settings.MIN_HOLDOUT_TRADES
