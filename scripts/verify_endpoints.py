"""
Verify the Binance USD-M futures websocket endpoints the recorder depends on.

    python scripts/verify_endpoints.py            # 30 s per endpoint, >= 5 messages each
    python scripts/verify_endpoints.py --seconds 15 --min-msgs 3

Checks the EXACT URLs the recorder builds (settings.WS_*), plus alternatives (depth20, the
raw @trade stream, and the legacy single combined /stream socket) so you can see which layout
works from your network. Needs internet; read-only market data, no keys.
Exit code 0 only if every endpoint the recorder REQUIRES works. Re-run if Binance changes
its futures websocket split (/public, /market, /private).
"""
import argparse
import asyncio
import json
import sys
import time
import urllib.request

import websockets

sys.path.insert(0, ".")
from config import settings  # noqa: E402

SYM = "btcusdt"


def checks():
    M, P, C = settings.WS_MARKET_BASE, settings.WS_PUBLIC_BASE, settings.WS_COMBINED_BASE
    d = settings.FEATURE_DEPTH_LEVELS
    return [
        # (label, url, expected event names, required-by-recorder-in-split-mode)
        ("market  aggTrade (recorder)", f"{M}{SYM}@aggTrade", {"aggTrade"}, True),
        ("public  depth%d@100ms (recorder)" % d, f"{P}{SYM}@depth{d}@100ms", {"depthUpdate"}, True),
        ("public  depth20@100ms", f"{P}{SYM}@depth20@100ms", {"depthUpdate"}, False),
        ("market  trade (raw)", f"{M}{SYM}@trade", {"trade"}, False),
        ("legacy combined aggTrade+depth (single socket)",
         f"{C}{SYM}@aggTrade/{SYM}@depth{d}@100ms", {"aggTrade", "depthUpdate"}, False),
    ]


async def probe(url: str, want: set, seconds: float, min_msgs: int):
    seen, first, t0 = {}, None, time.time()
    try:
        async with websockets.connect(url, open_timeout=15, ping_interval=20) as ws:
            while time.time() - t0 < seconds and not all(seen.get(e, 0) >= min_msgs for e in want):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, seconds - (time.time() - t0)))
                except asyncio.TimeoutError:
                    break
                d = json.loads(raw)
                ev = (d.get("data") or d).get("e")
                seen[ev] = seen.get(ev, 0) + 1
                first = first or raw[:160]
    except Exception as exc:                      # 403/404/429/418, DNS, TLS...
        return False, seen, f"{type(exc).__name__}: {exc}"
    ok = all(seen.get(e, 0) >= min_msgs for e in want)
    return ok, seen, (first if ok else f"only saw {seen or 'no messages'}")


def clock_skew_ms():
    try:
        t = time.time()
        with urllib.request.urlopen("https://fapi.binance.com/fapi/v1/time", timeout=10) as r:
            srv = json.loads(r.read())["serverTime"]
        return srv - (t + time.time()) / 2 * 1000
    except Exception:
        return None


async def main(seconds: float, min_msgs: int) -> int:
    print(f"Probing Binance futures websockets ({seconds:.0f}s each, need >= {min_msgs} msgs)\n")
    required_ok, rows = True, []
    for label, url, want, req in checks():
        ok, seen, info = await probe(url, want, seconds, min_msgs)
        rows.append((label, url, ok, req, seen, info))
        print(f"[{'PASS' if ok else 'FAIL'}] {label}{'  (REQUIRED)' if req else ''}\n       {url}\n       {info}\n")
        required_ok &= ok or not req
        await asyncio.sleep(1)                    # be gentle with rate limits
    sk = clock_skew_ms()
    if sk is not None:
        print(f"Local clock vs Binance server time: {sk:+.0f} ms "
              f"{'(OK)' if abs(sk) < 2000 else '(WARNING: >2 s skew; recorder quiet-market flush uses wall clock)'}")
    print("\n=== SUMMARY ===")
    print("Working :", [u for _, u, ok, *_ in rows if ok] or "none")
    print("Failing :", [u for _, u, ok, *_ in rows if not ok] or "none")
    comb = [r for r in rows if r[0].startswith("legacy")][0]
    if comb[2]:
        print("Note: the legacy combined socket works - you may set WS_RECORDER_MODE=combined (one connection).")
    if not required_ok:
        print("\nREQUIRED endpoint(s) failed. Check https://developers.binance.com/docs/derivatives/usds-margined-futures/"
              "websocket-market-streams for the current /public /market /private split and set "
              "WS_MARKET_BASE / WS_PUBLIC_BASE / WS_COMBINED_BASE accordingly.")
    return 0 if required_ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--min-msgs", type=int, default=5)
    a = ap.parse_args()
    raise SystemExit(asyncio.run(main(a.seconds, a.min_msgs)))
