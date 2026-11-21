# features/store.py
"""Parquet feature store: one small file per flush under
<root>/symbol=<SYM>/date=<YYYY-MM-DD>/part-<ms>.parquet. Used for both live
inference history and offline training. compact() merges a day's parts."""
from __future__ import annotations

import glob
import logging
import os
import threading
import time
from datetime import datetime, timezone

import pandas as pd

from features.orderflow import ALL_COLUMNS

log = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, root: str, flush_rows: int = 30):
        self.root = root
        self.flush_rows = flush_rows
        self._buf: dict[str, list] = {}
        self._lock = threading.Lock()

    # ---- write ----
    def append(self, row: dict) -> None:
        with self._lock:
            buf = self._buf.setdefault(row["symbol"], [])
            buf.append(row)
            full = len(buf) >= self.flush_rows
        if full:
            self.flush(row["symbol"])

    def flush(self, symbol: str | None = None) -> int:
        with self._lock:
            syms = [symbol] if symbol else list(self._buf)
            taken = {s: self._buf.pop(s, []) for s in syms}
        n = 0
        for sym, rows in taken.items():
            if not rows:
                continue
            df = pd.DataFrame(rows).reindex(columns=ALL_COLUMNS)
            df["_date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
            for date, part in df.groupby("_date"):
                d = os.path.join(self.root, f"symbol={sym}", f"date={date}")
                os.makedirs(d, exist_ok=True)
                tmp = os.path.join(d, f".tmp-{int(time.time() * 1000)}")
                part.drop(columns="_date").to_parquet(tmp, index=False)
                os.replace(tmp, os.path.join(d, f"part-{int(time.time() * 1000)}-{n}.parquet"))
                n += len(part)
        return n

    # ---- read ----
    def read(self, symbols: list[str] | None = None, start_ms: int | None = None,
             end_ms: int | None = None) -> pd.DataFrame:
        syms = symbols or [p.split("=", 1)[1] for p in
                           (os.path.basename(x) for x in glob.glob(os.path.join(self.root, "symbol=*")))]
        frames = []
        for sym in syms:
            for f in sorted(glob.glob(os.path.join(self.root, f"symbol={sym}", "date=*", "*.parquet"))):
                date = os.path.basename(os.path.dirname(f))[5:]
                day0 = int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
                if end_ms is not None and day0 > end_ms:
                    continue
                if start_ms is not None and day0 + 86_400_000 <= start_ms:
                    continue
                frames.append(pd.read_parquet(f))
        if not frames:
            return pd.DataFrame(columns=ALL_COLUMNS)
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(["symbol", "ts_ms"], keep="last")
        if start_ms is not None:
            df = df[df["ts_ms"] >= start_ms]
        if end_ms is not None:
            df = df[df["ts_ms"] <= end_ms]
        return df.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)

    def compact(self, keep_today: bool = True) -> int:
        """Merge each day's part files into one. Returns files removed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        removed = 0
        for d in glob.glob(os.path.join(self.root, "symbol=*", "date=*")):
            if keep_today and d.endswith(today):
                continue
            parts = sorted(glob.glob(os.path.join(d, "part-*.parquet")))
            if len(parts) < 2:
                continue
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            df = df.drop_duplicates(["symbol", "ts_ms"], keep="last").sort_values("ts_ms")
            tmp = os.path.join(d, f".tmp-compact-{int(time.time() * 1000)}")
            df.to_parquet(tmp, index=False)
            for p in parts:
                os.remove(p)
            os.replace(tmp, os.path.join(d, f"part-{int(time.time() * 1000)}-compact.parquet"))
            removed += len(parts) - 1
        return removed
