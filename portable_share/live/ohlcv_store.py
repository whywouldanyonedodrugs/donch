from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


class OHLCVSqliteStore:


    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT NOT NULL,
                tf     TEXT NOT NULL,
                ts     INTEGER NOT NULL,
                open   REAL NOT NULL,
                high   REAL NOT NULL,
                low    REAL NOT NULL,
                close  REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY(symbol, tf, ts)
            );
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_ts ON ohlcv(symbol, tf, ts);"
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    @staticmethod
    def _to_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if idx.tz is None:
            return idx.tz_localize("UTC")
        return idx.tz_convert("UTC")

    @staticmethod
    def _require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"OHLCV df missing columns: {missing} (have={list(df.columns)})")

    def load(
        self,
        symbol: str,
        tf: str,
        *,
        limit_bars: int,
    ) -> Optional[pd.DataFrame]:
        if limit_bars <= 0:
            return None

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol=? AND tf=?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (symbol, tf, int(limit_bars)),
            )
            rows = cur.fetchall()

        if not rows:
            return None

        rows = list(reversed(rows))
        ts_ms = np.asarray([r[0] for r in rows], dtype=np.int64)
        idx = pd.to_datetime(ts_ms, unit="ms", utc=True)

        arr = np.asarray([r[1:] for r in rows], dtype=float)
        df = pd.DataFrame(arr, index=idx, columns=["open", "high", "low", "close", "volume"])


        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if df.empty:
            return None
        return df

    def upsert_df(
        self,
        symbol: str,
        tf: str,
        df: pd.DataFrame,
        *,
        keep_last: Optional[int] = None,
    ) -> int:
        if df is None or df.empty:
            return 0

        self._require_cols(df, ["open", "high", "low", "close", "volume"])

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if df.empty:
            return 0

        idx = self._to_utc_index(df.index)
        ts_ms = (idx.view("int64") // 1_000_000).astype(np.int64)

        vals = df[["open", "high", "low", "close", "volume"]].astype(float).to_numpy()
        payload = [
            (
                symbol,
                tf,
                int(ts_ms[i]),
                float(vals[i, 0]),
                float(vals[i, 1]),
                float(vals[i, 2]),
                float(vals[i, 3]),
                float(vals[i, 4]),
            )
            for i in range(len(df))
        ]

        with self._lock:
            cur = self._conn.cursor()
            cur.executemany(
                """
                INSERT OR REPLACE INTO ohlcv(symbol, tf, ts, open, high, low, close, volume)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                payload,
            )
            self._conn.commit()

            if keep_last is not None and int(keep_last) > 0:
                self._trim_locked(symbol, tf, int(keep_last))

        return len(payload)

    def _trim_locked(self, symbol: str, tf: str, keep_last: int) -> None:

        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT ts
            FROM ohlcv
            WHERE symbol=? AND tf=?
            ORDER BY ts DESC
            LIMIT 1 OFFSET ?
            """,
            (symbol, tf, keep_last - 1),
        )
        row = cur.fetchone()
        if not row:
            return
        cutoff_ts = int(row[0])
        cur.execute(
            """
            DELETE FROM ohlcv
            WHERE symbol=? AND tf=? AND ts < ?
            """,
            (symbol, tf, cutoff_ts),
        )
        self._conn.commit()
