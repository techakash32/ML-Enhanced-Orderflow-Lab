# ML live log (fill in as milestones happen — dates in UTC)

| Milestone | Date | Notes |
|---|---|---|
| Endpoints verified (`scripts/verify_endpoints.py` all green) | | paste the PASS lines + which URLs work |
| Smoke test 6.4 passed (`scripts/smoke_deploy.sh`) | | machine, Docker version |
| Render deploy live (disk mounted at `/app/logs`) | | service URL, plan, disk size |
| Recording started (`/health` shows recorder heartbeat) | | symbols, `FEATURE_SYMBOLS` |
| 7 days of continuous recording | | feed resets / gaps seen |
| First training run (MLflow run id) | | rows, holdout trades, outcome (`needs_more_data`?) |
| First champion promoted | | registry version, holdout net R, win rate vs breakeven (with CI) |
| `USE_ML_GATE=true` flipped | | |
| First ML-gated paper trade | | `signals.jsonl` line with `"gate": "ml"` and `ml_p` |

## Notes
