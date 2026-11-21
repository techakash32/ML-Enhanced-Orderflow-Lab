#!/usr/bin/env bash
# Phase 6.4 smoke test - run on a machine WITH internet + Docker. Prints PASS/FAIL per check
# and a final summary; exits non-zero if anything failed. Nothing here places orders.
#   scripts/smoke_deploy.sh            # everything
#   SKIP_DOCKER=1 scripts/smoke_deploy.sh
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"; OUT=smoke_out; mkdir -p "$OUT"; FAILS=0
ok(){ echo "PASS  $1"; }; bad(){ echo "FAIL  $1"; FAILS=$((FAILS+1)); }
chk(){ local n="$1"; shift; if "$@" >"$OUT/$(echo "$n"|tr ' /' '__').log" 2>&1; then ok "$n"; else bad "$n  (see $OUT/)"; fi; }

echo "== 1 endpoints";            chk "1 verify_endpoints" $PY scripts/verify_endpoints.py --seconds 20
echo "== 2 recorder dry-run";     chk "2 recorder --dry-run" $PY -m features.recorder --symbols BTCUSDT --dry-run 5
echo "== 3 record 10 min BTCUSDT"
export FEATURE_STORE_PATH="$OUT/store" FEATURE_HEARTBEAT_PATH="$OUT/hb.json"
timeout 620 $PY -m features.recorder --symbols BTCUSDT >"$OUT/record.log" 2>&1 || true
n=$(find "$OUT/store" -name '*.parquet' 2>/dev/null | wc -l); [ "$n" -gt 0 ] && ok "3 parquet files written ($n)" || bad "3 no parquet written"
echo "== 4 feature check"
chk "4 feature sanity" $PY - <<'PY'
import numpy as np, os
from features.store import FeatureStore
df = FeatureStore(os.environ["FEATURE_STORE_PATH"]).read(["BTCUSDT"])
assert len(df) >= 8, f"only {len(df)} snapshots"
assert df.ts_ms.is_monotonic_increasing and df.ts_ms.is_unique
assert (np.diff(df.ts_ms) == 60_000).mean() > 0.9, "snapshots not 1/min"
need = ["close", "spread_bps", "qi_l1", "ofi_snap_l5", "micro_dev_bps", "volume", "n_trades"]
assert not df[need].iloc[1:].isna().any().any(), df[need].isna().sum()
# manual spot check: buy/sell volume partition and OFI/qi bounds
assert np.allclose(df.buy_vol + df.sell_vol, df.volume)
assert df.qi_l1.between(-1, 1).all() and df.ofi_snap_l5.between(-1, 1).all()
print(df[["ts_ms","close","volume","qi_l1","ofi_snap_l5","micro_dev_bps","vpin","warm"]].tail(3))
print("NOTE: vpin/z-scores/warm are NaN/0 until ~6 h of data - expected after 10 min")
PY
echo "== 5 training pipeline"
chk "5a real 10-min sample -> graceful 'not enough data' (rc 2)" bash -c "$PY -m ml.train --store $OUT/store --symbols BTCUSDT; [ \$? -eq 2 ]"
$PY -m ml.synthetic --root "$OUT/synth" --days 14 >/dev/null 2>&1
chk "5b synthetic end-to-end train" $PY -m ml.train --store "$OUT/synth" --symbols SYNTHUSDT --bar-s 900 --out-dir "$OUT/models" --tracking-uri "sqlite:///$OUT/mlflow.db" --threshold 0.45
if [ -z "${SKIP_DOCKER:-}" ] && command -v docker >/dev/null; then
  echo "== 6 docker build";       chk "6 docker build" docker build -t ofl-ml .
  echo "== 7/9 docker run (ML gate on / off)"
  for g in true false; do
    docker rm -f ofl-smoke >/dev/null 2>&1
    docker run -d --name ofl-smoke -p 8000:8000 -e USE_ML_GATE=$g ofl-ml >/dev/null; sleep 25
    curl -fsS localhost:8000/health >"$OUT/health_$g.json" && ok "7 dashboard+/health reachable (USE_ML_GATE=$g)" || bad "7 /health (USE_ML_GATE=$g)"
    docker logs ofl-smoke 2>&1 | grep -E "ML gate|Entry gate" | head -3
    if [ $g = true ]; then docker logs ofl-smoke 2>&1 | grep -qE "ML gate active|falling back to rule-based" && ok "7 log shows ML gate state" || bad "7 no ML gate log line"
    else docker logs ofl-smoke 2>&1 | grep -q "rule-based (USE_ML_GATE=false)" && ok "9 rule-based gate active" || bad "9 rule-based log line missing"; fi
  done; docker rm -f ofl-smoke >/dev/null 2>&1
  echo "== 8 compose"
  docker compose up -d --build >"$OUT/compose.log" 2>&1; sleep 40
  curl -fsS localhost:8000/health >/dev/null && curl -fsS localhost:5000 >/dev/null && ok "8 compose: dashboard + MLflow UI" || bad "8 compose services"
  docker compose down >/dev/null 2>&1
else echo "SKIP  6-9 (docker not available or SKIP_DOCKER set)"; fi
echo; [ $FAILS -eq 0 ] && echo "ALL CHECKS PASSED" || echo "$FAILS CHECK(S) FAILED"; exit $FAILS
