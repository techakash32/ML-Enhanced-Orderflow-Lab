# data/candle_monitor.py
"""Display-only candle feed monitor.

Subscribes to extra kline streams (default 1m) and prints every CLOSED candle, so you can
see that candles are arriving. It never touches trading: the paper trader keeps using its
own closed-4h stream. Enable with SHOW_CANDLES=true (see config/settings.py).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from config import settings
from data.ws_manager import BinanceMarketStream

log = logging.getLogger("candles")


def _fmt(c: dict) -> str:
    t = datetime.fromtimestamp(c["open_time"] / 1000, timezone.utc).strftime("%H:%M")
    return (f"{c['symbol']:<8} {c['interval']:>3} candle {t} UTC  "
            f"O={c['open']:.4f} H={c['high']:.4f} L={c['low']:.4f} C={c['close']:.4f}  "
            f"vol={c['volume']:.2f} trades={c['trades']}")


class CandleMonitor:
    def __init__(self, symbols: list[str], intervals: list[str], save_path: str | None = None):
        self.streams = [BinanceMarketStream(symbols, i, include_trades=False) for i in intervals]
        self.counts: dict[str, int] = {}
        self._save = save_path
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        for s in self.streams:
            s.on_candle(self._on_candle)

    async def _on_candle(self, c: dict) -> None:
        key = f"{c['symbol']}:{c['interval']}"
        self.counts[key] = self.counts.get(key, 0) + 1
        log.info("%s  [#%d]", _fmt(c), self.counts[key])
        if self._save:
            with open(self._save, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(c) + "\n")

    async def run(self) -> None:
        import asyncio
        await asyncio.gather(*(s.run() for s in self.streams))


def from_settings() -> CandleMonitor | None:
    if not settings.SHOW_CANDLES:
        return None
    return CandleMonitor(list(settings.SYMBOLS), list(settings.SHOW_CANDLE_INTERVALS),
                         settings.CANDLE_LOG_PATH or None)