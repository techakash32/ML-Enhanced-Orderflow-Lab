# Label / exit parity and trade-count report (Phase 6.1 + 6.3)

## 1. The rule
**The model must be trained on the outcome of the trade the bot will actually run.** Labels are produced by
`ml/exit_sim.py::simulate_trade`, which replays a trade with the live `PaperTrader` rules. Its parameters are read from
`config/settings.py` — the single source of truth that `execution/paper_trader.py` also reads — via
`ExitRules.from_settings()`:

| Live rule (`PaperTrader`, trend mode) | Setting | Simulator |
|---|---|---|
| Signal on bar *i* close → fill at bar *i+1* open, adverse slippage | `PAPER_SLIPPAGE_PCT` | same |
| Initial stop = entry ∓ 2 × ATR(14) measured on bars ≤ *i* | `TREND_ATR_MULTIPLE`, `TREND_ATR_WINDOW` | same |
| Entry bar's own range is checked against the stop | — | same |
| No profit target (only outside trend mode: `PAPER_TARGET_R_MULTIPLE`) | `PAPER_TARGET_R_MULTIPLE` | same |
| After each surviving bar: stop = max(stop, high − 2×ATR incl. that bar); active next bar | `TREND_ATR_MULTIPLE` | same |
| Stop fills at the stop price (+ slippage); stop wins a same-bar tie | — | same |
| Time stop after 90 bars at the bar close (+ slippage) | `PAPER_MAX_HOLD_BARS` | same |
| Taker fee entry+exit (maker on a TARGET exit) | `PAPER_FEE_*` | same |

**Label:** by *net R* (`pnl / initial stop distance`, identical to `PaperTrader`'s `r_multiple`): `+1` if > +0.10R,
`−1` if < −0.10R, else `0` (breakeven band, `--be-band-r`). A trailing stop that exits in profit is therefore a **win**; a
trade stopped at the initial stop is a **loss (−1)**; a flat time-out is `0`.
(Decision points are bar closes only — that is when the live Decider runs.)

**Enforcement, so it cannot drift**
1. `tests/test_exit_parity.py` replays random price paths through the **real `PaperTrader`** and through the simulator and
   asserts identical exit reason, exit price, R-multiple and bars-held (all four seeds pass). Any change to the trader's
   exit logic that is not mirrored breaks this test.
2. Each model's `manifest.json` stores the `exit_rules` it was trained on. `ml/gate.py` **refuses to load** a model whose
   rules differ from the current settings (`ML_ALLOW_EXIT_MISMATCH=true` overrides, with a warning).
3. `--exit-mode static_barrier` keeps the old 1.5/1.0/6-bar labels for research only; such models log a mismatch warning.

Consequences to be aware of: (a) a label needs up to 90 bars (15 days at 4h) to resolve, so the newest ~15 days are
unlabelled and the minimum useful history is longer than before; (b) the entry-bar stop check and the trailing logic make
labels path-dependent — a purge/embargo of one bar is applied between train and test, but a trade can span many bars, so
CV purging uses each trade's actual exit time (`t1`).

## 2. Does the edge survive the parity labels? (synthetic data only)
Run: 14 synthetic days, 15-min bars, `python -m ml.train --bar-s 900 --threshold 0.45` (planted edge, so this only proves the
plumbing, **not** that real markets have one):

| | trades | win rate | net R/trade |
|---|---|---|---|
| ungated holdout (all rows) | 520 | — | edge +0.136 |
| gate P>0.45 | 181 | 0.442 | **+0.604R** (CI of edge +0.099…+0.397) |

The holdout metric stayed positive after switching from static barriers to the live trailing exit. **Real-data validation
has not been done** — there is no recorded order-book history yet.

## 3. Trade-count report
`ml/train.py` now prints (and stores in `manifest.json` / `report.json`) a table per threshold: holdout trades over *all
labelled rows* and over *live-candidate rows* (those where Bucket A's Donchian rule would actually ask the gate), with win
rate and net R. Synthetic example (threshold sweep on the holdout, 520 rows, only 10 of them live candidates):

| P > | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 |
|---|---|---|---|---|---|---|---|
| all rows | 357 | 269 | 181 | 86 | 25 | 3 | 0 |
| live candidates | 8 | 7 | 5 | 1 | 0 | 0 | 0 |

Two different counts matter:
* **Statistical trades** (all rows the gate accepts) — what the 30-trade promotion floor measures.
* **Live trades** (candidates only) — what the bot will actually execute. Donchian(40) breakouts in the London window are
  rare (~2 % of rows in the synthetic example; the real rate must be measured from recorded data / the existing backtest).
  Expect the live gate to fire only a handful of times per month, which also limits how fast live paper results can confirm the model.

**Promotion rule:** a challenger with fewer than `MIN_HOLDOUT_TRADES` (env, default 30) holdout trades is **not promoted**;
`ml.train`/`ml.retrain` report status `needs_more_data`, keep the current champion, and the retrain history records it.

**Levers (in the order the plan suggests) and what was done**
| Lever | Status |
|---|---|
| Lower `ML_GATE_THRESHOLD` (0.58 → 0.52–0.55) | Default 0.55. The table above is the sweep to use on real data; pick from the *holdout/OOF* table, not by eyeballing live results. |
| More symbols | **Done:** `FEATURE_SYMBOLS` records 8 liquid perps (BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX) in the Docker image; the model is pooled across symbols; `SYMBOLS` (traded) is unchanged. |
| All sessions instead of London-only | Training already samples **all** bar closes. The *live* London-only filter is a strategy rule (`TREND_LONDON_ONLY`); relaxing it changes what the bot trades, so it is left to you — not changed. |
| 1h bars instead of 4h | Not changed: `BACKTEST_INTERVAL` rescales the whole strategy. The recorder is interval-agnostic (1-min snapshots), so this stays possible later. |

**Rough data requirement (estimate, not measured):** 8 symbols × 6 bar-closes × 2 sides ≈ 96 rows/day. With a ~15 % gate take-rate,
≥ 30 holdout trades needs ≈ 200 holdout rows ⇒ ≈ 1000 labelled rows ⇒ ≈ 10 days of *resolved* bars, plus the 15-day label horizon,
2.3-day ATR warm-up and ~6 h feature warm-up ⇒ **roughly 4 weeks of recording** for a first promotable champion; more if the model is selective.
