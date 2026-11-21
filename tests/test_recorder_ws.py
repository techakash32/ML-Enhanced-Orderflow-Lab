"""Recorder websocket loop + --dry-run against a LOCAL fake Binance-style server (no internet)."""
import asyncio
import json
import time

import pytest
import websockets

from config import settings
from features.orderflow import OrderFlowConfig
from features.recorder import FeatureRecorder, dry_run
from features.store import FeatureStore


async def fake_binance(ws):
    t = int(time.time() * 1000)
    for i in range(400):
        t += 20
        await ws.send(json.dumps({"stream": "btcusdt@aggTrade", "data": {
            "e": "aggTrade", "s": "BTCUSDT", "T": t, "p": "100.0", "q": "0.5", "m": i % 3 == 0}}))
        await ws.send(json.dumps({"stream": "btcusdt@depth5@100ms", "data": {
            "e": "depthUpdate", "s": "BTCUSDT", "T": t, "b": [["99.5", "2"]], "a": [["100.5", "1"]]}}))
        await asyncio.sleep(0.002)
    await asyncio.sleep(2)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    return None


def test_dry_run_and_recorder_run(server, monkeypatch, tmp_path):
    async def go():
        async with websockets.serve(fake_binance, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            base = f"ws://127.0.0.1:{port}/x?streams="
            monkeypatch.setattr(settings, "WS_MARKET_BASE", base)
            monkeypatch.setattr(settings, "WS_PUBLIC_BASE", base)
            monkeypatch.setattr(settings, "FEATURE_HEARTBEAT_PATH", str(tmp_path / "hb.json"))
            assert await dry_run(["BTCUSDT"], n=5, timeout=10) == 0
            rec = FeatureRecorder(["BTCUSDT"], FeatureStore(str(tmp_path / "fs")),
                                  OrderFlowConfig(snapshot_s=60))
            task = asyncio.create_task(rec.run())
            await asyncio.sleep(3)
            rec.stop(); task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return rec
    rec = asyncio.run(go())
    assert rec.stats["trades"] > 100 and rec.stats["books"] > 100


def test_dry_run_fails_cleanly_when_unreachable(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(settings, "WS_MARKET_BASE", "ws://127.0.0.1:9/x?streams=")
    monkeypatch.setattr(settings, "WS_PUBLIC_BASE", "ws://127.0.0.1:9/x?streams=")
    assert asyncio.run(dry_run(["BTCUSDT"], n=2, timeout=5)) == 1
