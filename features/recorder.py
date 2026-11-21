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

import asyncio
import json
import logging
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

    # ---- query for the Decider ----
    def features_asof(self, symbol: str, ts_ms: int) -> dict | None:
        """Snapshot stamped ts_ms (a bar boundary). None if unavailable."""
        eng = self.engines.get(symbol.upper())
        return eng.snapshot_at(ts_ms) if eng else None

    # ---- ws plumbing ----
    def _urls(self):
        low = [s.lower() for s in self.symbols]
        mk = settings.WS_MARKET_BASE + "/".join(f"{s}@aggTrade" for s in low)
        pub = settings.WS_PUBLIC_BASE + "/".join(
            f"{s}@depth{self.cfg.depth_levels}@100ms" for s in low)
        return mk, pub

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

    async def _ticker(self) -> None:
        """Emit boundaries on quiet markets and flush the store regularly."""
        while self._running:
            await asyncio.sleep(max(5, self.cfg.snapshot_s // 2))
            now = int(time.time() * 1000) - 1500       # small lag for in-flight events
            for eng in self.engines.values():
                eng.advance_to(now)
            await asyncio.to_thread(self.store.flush)

    async def run(self) -> None:
        self._running = True
        mk, pub = self._urls()
        log.info("recorder starting for %s (snapshot every %ds)", self.symbols, self.cfg.snapshot_s)
        tasks = [asyncio.create_task(self._ws_loop(mk, "aggTrade")),
                 asyncio.create_task(self._ws_loop(pub, "depth")),
                 asyncio.create_task(self._ticker())]
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    try:
        asyncio.run(FeatureRecorder(settings.SYMBOLS).run())
    except KeyboardInterrupt:
        print("\nStopped.")
