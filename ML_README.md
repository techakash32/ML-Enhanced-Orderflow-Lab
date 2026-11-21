# ML entry gate (LightGBM) — quick guide

The ML model replaces **only the rule-based confirmation step** in `engine/decision.py`.
Bucket A (the Donchian breakout) still proposes a direction; the model then estimates
**P(profit target hit before stop)** for that side from order-flow features, and the trade is
taken only if `P > ML_GATE_THRESHOLD`. The optional Bucket C (GEX) veto, cooldown, `PaperTrader`,
exits, dashboard and evaluation framework are unchanged. **Paper trading only — no order execution was added.**

```
Binance /market aggTrade ─┐
                          ├─► features/recorder.py ─► features/orderflow.py ─► Parquet store (logs/feature_store)
Binance /public depth5 ───┘                                   │                       │
                                                  snapshot at bar close        ml/train.py (labels, purged CV,
                                                              ▼                 LightGBM, MLflow, champion)
 Bucket A ─► engine/decision.py ── ML gate (P > thr) ─► PaperTrader ─► dashboard        │
                    ▲                                                                   │
                    └──────────── champion bundle (logs/ml/models/champion) ◄───────────┘
```

## 1. Record data (nothing to train on yet)
Order-book history is not available from Binance, so training data is **recorded live** by the
feature recorder. It starts automatically with the paper trader when `RECORD_FEATURES=true` or
`USE_ML_GATE=true` (the Docker image sets the latter), or standalone: `python -m features.recorder`.
One snapshot per symbol per minute is written to `FEATURE_STORE_PATH` (default `logs/feature_store`;
inside Docker this is on the existing `/app/logs` volume — **mount a persistent disk on Render or the
data is lost on redeploy**).

Features per snapshot (`features/orderflow.py`): multi-level snapshot OFI (+ rolling z), event OFI (Cont et al.,
1m/5m, + z), best-level queue imbalance and its rate of change, microprice deviation and drift, VPIN (volume buckets,
exchange-labelled aggressor side, z-score, toxicity band, signed VPIN), trade-size mean/std/skew, large-trade
(> p95) share and net direction, spread, multi-horizon returns, realised vol. Features are strictly causal
(a snapshot stamped *T* only sees events before *T*; covered by `tests/test_orderflow.py`).
A snapshot is flagged `warm` after ~6 h of continuous data (VPIN window + z-score history); the model is
trained and served only on warm rows.

## 2. Train the first model
Needs roughly **4+ weeks** of recorded data with the default 4h bars (ATR warm-up 2.3 days, up to 15 days for a trade
to resolve, plus the CV and holdout splits; see docs/PARITY_REPORT.md).
```bash
python -m ml.train --lookback-days 60            # defaults: live-exit parity labels
python -m ml.train --start 2026-12-01 --end 2027-01-20 --objective binary \
    --tp-atr 1.5 --sl-atr 1.0 --max-hold-bars 6 --threshold 0.58 --num-leaves 15 --max-depth 4
python -m ml.train --help                        # all hyper-parameters
```
What it does: labels every bar close for **both sides** with the outcome of a simulated live trade (see *Label/exit parity*
below), features sign-flipped for shorts so one model serves both → time-based holdout (last 20 %) → **purged walk-forward CV**
(purging by each trade's exit time, plus embargo) → refit → evaluate on the holdout. It reports accuracy, per-class
precision/recall, log-loss, AUC and the **economic metric**: win rate vs breakeven win rate (`L/(W+L)` from the taken trades'
average win/loss in R), net expectancy in R and a block-bootstrap 95 % CI, the same for the *ungated* baseline, and a
**trade-count table** per threshold (all rows vs live-candidate rows). `--sample-mode candidates` trains only on Donchian
bar-closes (what the live gate sees; far fewer rows). Promotion uses holdout `expectancy_r` (`--promotion-metric`).

Artifacts: `logs/ml/models/runs/<run>/{model.txt, manifest.json, feature_importance.csv, report.json}`
(manifest = feature order/types, objective, barrier geometry, training date, metrics). The first model that
clears the promotion floor becomes **champion** (copied to `ML_MODEL_PATH` and aliased `champion` in MLflow).

## 3. Enable / disable the gate
| Variable | Default | Meaning |
|---|---|---|
| `USE_ML_GATE` | `false` locally, `true` in Docker | `false` ⇒ original rule-based gate, exactly as before |
| `ML_MODEL_PATH` | `logs/ml/models/champion` | local champion bundle |
| `ML_GATE_THRESHOLD` | `0.55` | enter only if P(target first) **>** threshold |
| `MLFLOW_TRACKING_URI` | *(unset)* | set to use the registry champion (`sqlite:///logs/ml/mlruns/mlflow.db`) |
| `ML_MODEL_SOURCE` | `auto` | `auto` / `registry` / `local` |
| `ML_ON_MISSING_FEATURES` | `reject` | model loaded but no warm live features: `reject` (stand aside) or `fallback` (rules) |
| `RECORD_FEATURES`, `FEATURE_STORE_PATH`, `FEATURE_SNAPSHOT_S` | | recorder controls |

Behaviour: no model artifact ⇒ automatic fallback to the rule-based gate (one warning logged). Model present but
the recorder has no warm snapshot ⇒ stand aside (fail closed) unless `ML_ON_MISSING_FEATURES=fallback`. A newly
promoted champion is hot-reloaded within `ML_RELOAD_CHECK_S` (300 s) — no restart. Signals log
(`logs/paper_live/signals.jsonl`) gains `gate` and `ml_p` fields; the dashboard ignores extra keys.
`scripts/backtest_step5a.py` always uses the rule-based gate (historical klines have no order-book features).

## 4. MLflow
Tracking uses SQLite under `logs/ml/mlruns/` (no server needed). Every run logs params, CV/holdout metrics, per-feature
gain, and the bundle; each is registered as a model version of `orderflow_lgbm`.
```bash
mlflow ui --backend-store-uri sqlite:///logs/ml/mlruns/mlflow.db --port 5000
# or inside the container: MLFLOW_SERVER=true  (Render exposes only $PORT, so use it locally / on a VM)
```

## 5. Weekly retraining (champion/challenger)
`ml/retrain.sh` → `python -m ml.retrain`: compacts the store, relabels, trains on the last 90 days
(`RETRAIN_LOOKBACK_DAYS`), evaluates on a time holdout, logs to MLflow, and promotes **only if** the challenger's
holdout `edge` (win-rate − breakeven) is ≥ champion's − `--tolerance` (0.01), took ≥ 30 holdout trades, and — for a first
champion — is ≥ 0. Both models are scored on the *same* holdout. History: `logs/ml/retrain_history.jsonl`.
```bash
# Linux cron (Sundays 03:15 UTC)
15 3 * * 0  /path/to/repo/ml/retrain.sh
# Windows Task Scheduler
schtasks /Create /SC WEEKLY /D SUN /ST 03:15 /TN "OrderflowRetrain" /TR "cmd /c cd /d C:\path\to\repo && python -m ml.retrain"
# Inside the container (no cron on Render): ENABLE_RETRAIN_LOOP=true  RETRAIN_EVERY_DAYS=7
```

## Docker / Render
`COPY . .` already ships `features/` and `ml/`; `requirements.txt` adds lightgbm, mlflow, pandas, pyarrow, numpy,
scikit-learn. `start.sh` starts paper-live (+recorder) and the dashboard as before; `MLFLOW_SERVER` and
`ENABLE_RETRAIN_LOOP` are opt-in and cannot take the trader down. Flow: Binance WS → recorder → ML gate → PaperTrader → dashboard.

## Label/exit parity (Phase 6.1) — the rule
Labels are the outcome of a **simulated live trade** (same entry, 2×ATR initial + trailing stop, 90-bar time stop, costs), not
static barriers. Parameters come from `config/settings.py` (what `PaperTrader` reads); `tests/test_exit_parity.py` replays paths
through the real `PaperTrader` to prove equality; each model stores its `exit_rules` and the live gate refuses a mismatch.
Details and the trade-count analysis: [docs/PARITY_REPORT.md](docs/PARITY_REPORT.md). `--exit-mode static_barrier` is research-only.

## Endpoints (Phase 6.2) — **not yet verified against the live service**
The recorder uses (`config/settings.py`, overridable by env): `wss://fstream.binance.com/market/stream?streams=<sym>@aggTrade`
and `wss://fstream.binance.com/public/stream?streams=<sym>@depth5@100ms` (Binance's /public /market /private split).
**Before starting a multi-week recording, on a machine with internet run:**
```bash
python scripts/verify_endpoints.py            # PASS/FAIL per URL, incl. depth20, raw @trade and the legacy single-socket /stream
python -m features.recorder --dry-run         # prints real trade + depth messages, writes nothing
```
If the legacy combined socket passes you may set `WS_RECORDER_MODE=combined` (one connection). Record the working URLs in
`docs/ML_LIVE_LOG.md` and re-verify if Binance changes the split. (This sandbox's network blocks Binance, so only the failure
path of the script and a local fake-server test of the recorder were exercised.)

## Health, compose, Render
`GET /health` (also a badge on the dashboard): uptime, last feature timestamp/age, recorder heartbeat, last paper trade, model
present/champion version/parity flag, last retrain status; `status: degraded` if the recorder heartbeat is stale while recording
is expected. `docker-compose.yml` runs trader + MLflow UI (+ optional retrain profile); `render.yaml` is an example blueprint —
**mount the persistent disk at `/app/logs`** (all state lives there), Starter plan or higher. `scripts/smoke_deploy.sh` automates the
10 smoke checks on a machine with internet + Docker.

## Things to know before trusting a number
* **Overlapping labels.** A trade can last up to 90 bars, so neighbouring labels overlap heavily; purging uses each trade's
  exit time and the bootstrap uses blocks, but the effective sample size is far below the row count.
* **Threshold semantics.** P(+1) is a probability only if not re-weighted (`--class-weight none`, the default). With
  base rate ≈ SL/(TP+SL) = 40 %, 0.55 demands a large lift and may select very few trades; check `report.json` thresholds.
* **Few trades.** Live candidates (Donchian + London) are rare; below `MIN_HOLDOUT_TRADES` (30) no challenger is promoted
  (`needs_more_data`). See docs/PARITY_REPORT.md for the levers.
* `tests/` uses synthetic data (`ml/synthetic.py`) only to prove the plumbing; its metrics mean nothing about markets.

Tests: `python -m pytest tests -q`
