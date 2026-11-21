# features/recorder.py
"""
Live order-flow recorder (Phase 1). Runs inside the paper-live process (or
standalone: `python -m features.recorder`).

Binance USD-M split its websocket endpoints: aggTrade lives on /market,
depth/bookTicker on /public, so this opens two combined-stream sockets and
feeds one OrderFlowFeatureEngine per symbol. Snapshots go to the Parquet
FeatureStore; the live Decider reads them via features_asof().
This module only reads market data - it never places orders.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time

import websockets

from config import settings
from features.orderflow import OrderFlowConfig, OrderFlowFeatureEngine
from features.store import FeatureStore

log = logging.getLogger("recorder")


class FeatureRecorder:
    def __init__(self, symbols: list[str], store: FeatureStore | None = None,
                 cfg: OrderFlowConfig | None = None):
        self.symbols = [s.upper() for s in symbols]
        self.cfg = cfg or OrderFlowConfig(snapshot_s=settings.FEATURE_SNAPSHOT_S,
                                          depth_levels=settings.FEATURE_DEPTH_LEVELS)
        self.store = store if store is not None else FeatureStore(settings.FEATURE_STORE_PATH)
        self.engines = {s: OrderFlowFeatureEngine(s, self.cfg, sink=self.store.append)
                        for s in self.symbols}
        self._running = False
        self.stats = {"trades": 0, "books": 0, "reconnects": 0}
        self.started = time.time()
        self.last_event_ts = None

    # ---- query for the Decider ----
    def features_asof(self, symbol: str, ts_ms: int) -> dict | None:
        """Snapshot stamped ts_ms (a bar boundary). None if unavailable."""
        eng = self.engines.get(symbol.upper())
        return eng.snapshot_at(ts_ms) if eng else None

    # ---- ws plumbing ----
    def _urls(self):
        """[(name, url)] - two sockets (split) or one legacy combined socket."""
        low = [s.lower() for s in self.symbols]
        trades = [f"{s}@aggTrade" for s in low]
        depth = [f"{s}@depth{self.cfg.depth_levels}@100ms" for s in low]
        if settings.WS_RECORDER_MODE == "combined":
            return [("combined", settings.WS_COMBINED_BASE + "/".join(trades + depth))]
        return [("aggTrade", settings.WS_MARKET_BASE + "/".join(trades)),
                ("depth", settings.WS_PUBLIC_BASE + "/".join(depth))]

    def handle(self, raw) -> None:
        """Parse one combined-stream frame (sync; also used by tests/replay)."""
        try:
            data = json.loads(raw).get("data")
        except ValueError:
            return
        if not data:
            return
        sym = data.get("s")
        eng = self.engines.get(sym)
        if eng is None:
            return
        e = data.get("e")
        self.last_event_ts = time.time()
        if e == "aggTrade":
            self.stats["trades"] += 1
            eng.on_trade(int(data["T"]), float(data["p"]), float(data["q"]), bool(data["m"]))
        elif e == "depthUpdate":
            self.stats["books"] += 1
            eng.on_book(int(data.get("T", data.get("E"))),
                        [(float(p), float(q)) for p, q in data["b"]],
                        [(float(p), float(q)) for p, q in data["a"]])

    async def _ws_loop(self, url: str, name: str) -> None:
        delay = settings.RECONNECT_BASE_DELAY
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20,
                                              open_timeout=15, max_queue=8192) as ws:
                    log.info("recorder %s connected", name)
                    delay = settings.RECONNECT_BASE_DELAY
                    async for raw in ws:
                        if not self._running:
                            break
                        self.handle(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.stats["reconnects"] += 1
                log.warning("recorder %s error (%s); retry in %.1fs", name, exc, delay)
            if self._running:
                await asyncio.sleep(delay)
                delay = min(delay * 2, settings.RECONNECT_MAX_DELAY)

    def _heartbeat(self) -> None:
        """Small JSON file the dashboard /health endpoint reads (separate process)."""
        last = {sym: (eng.snapshots[-1]["ts_ms"] if eng.snapshots else None)
                for sym, eng in self.engines.items()}
        path = settings.FEATURE_HEARTBEAT_PATH
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"ts": time.time(), "started": self.started, "stats": self.stats,
                           "last_event": self.last_event_ts, "last_snapshot_ms": last,
                           "resets": {s: e.resets for s, e in self.engines.items()}}, f)
            os.replace(path + ".tmp", path)
        except OSError:
            pass

    async def _ticker(self) -> None:
        """Emit boundaries on quiet markets, write the heartbeat, flush the store every
        FEATURE_FLUSH_S (one small Parquet part per symbol per flush; compacted daily)."""
        last_flush = time.time()
        while self._running:
            await asyncio.sleep(max(5, self.cfg.snapshot_s // 2))
            now = int(time.time() * 1000) - 1500       # small lag for in-flight events
            for eng in self.engines.values():
                eng.advance_to(now)
            self._heartbeat()
            if time.time() - last_flush >= settings.FEATURE_FLUSH_S:
                last_flush = time.time()
                await asyncio.to_thread(self.store.flush)

    async def run(self) -> None:
        self._running = True
        log.info("recorder starting for %s (snapshot every %ds, mode=%s)", self.symbols,
                 self.cfg.snapshot_s, settings.WS_RECORDER_MODE)
        tasks = [asyncio.create_task(self._ws_loop(url, name)) for name, url in self._urls()]
        tasks.append(asyncio.create_task(self._ticker()))
        try:
            await asyncio.gather(*tasks)
        finally:
            self._running = False
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.store.flush()

    def stop(self) -> None:
        self._running = False


async def dry_run(symbols: list[str], n: int = 10, timeout: float = 30.0) -> int:
    """Connect, print the first n trade + n depth messages, write NOTHING, exit 0 if both arrived."""
    rec = FeatureRecorder(symbols, store=FeatureStore("/nonexistent-dry-run"))
    got = {"aggTrade": 0, "depthUpdate": 0}

    async def one(name, url):
        async with websockets.connect(url, open_timeout=15, ping_interval=20) as ws:
            print(f"[{name}] connected: {url}")
            while any(v < n for v in got.values()):
                raw = await ws.recv()
                d = json.loads(raw).get("data") or {}
                e = d.get("e")
                if e in got and got[e] < n:
                    got[e] += 1
                    keys = ("s", "T", "p", "q", "m") if e == "aggTrade" else ("s", "T", "b", "a")
                    brief = {k: (d[k][:2] if isinstance(d[k], list) else d[k]) for k in keys if k in d}
                    print(f"  {e} #{got[e]}: {brief}")
                    rec.handle(raw)

    try:
        await asyncio.wait_for(asyncio.gather(*(one(nm, u) for nm, u in rec._urls())), timeout)
    except asyncio.TimeoutError:
        pass
    except Exception as exc:
        print(f"ERROR: {exc!r}")
    ok = all(v >= n for v in got.values())
    print(f"dry-run {'OK' if ok else 'FAILED'}: {got} (needed {n} each within {timeout:.0f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Order-flow feature recorder (reads market data only)")
    ap.add_argument("--symbols", default=",".join(settings.FEATURE_SYMBOLS))
    ap.add_argument("--dry-run", type=int, nargs="?", const=10, metavar="N",
                    help="connect, print N trade + N depth messages, write nothing, exit")
    ap.add_argument("--timeout", type=float, default=30.0, help="dry-run time limit (s)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    syms = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    if args.dry_run:
        raise SystemExit(asyncio.run(dry_run(syms, args.dry_run, args.timeout)))
    try:
        asyncio.run(FeatureRecorder(syms).run())
    except KeyboardInterrupt:
        print("\nStopped.")
