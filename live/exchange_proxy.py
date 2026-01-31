from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential


LOG = logging.getLogger(__name__)


def _ensure_utc_ts(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _to_ms(ts: pd.Timestamp) -> int:
    ts = pd.to_datetime(ts, utc=True)
    return int(ts.value // 1_000_000)


def _tf_to_ms(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1]) * 60_000
    if tf.endswith("h"):
        return int(tf[:-1]) * 3_600_000
    if tf.endswith("d"):
        return int(tf[:-1]) * 86_400_000
    raise ValueError(f"Unsupported timeframe: {tf}")


class ExchangeProxy:
    """
    Thin wrapper around a CCXT async exchange providing deterministic pagination and
    strict as-of semantics helpers for live trading parity.
    """

    def __init__(self, ex: Any, *, cfg: Optional[dict] = None):
        self.ex = ex
        self.cfg = cfg or {}

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.ex, name)

        if not callable(attr):
            return attr

        async def _wrap(*args, **kwargs):
            return await attr(*args, **kwargs)

        return _wrap

    # ------------------------
    # OHLCV helpers
    # ------------------------

    async def fetch_ohlcv_df(self, symbol: str, tf: str, *, limit: int = 500) -> pd.DataFrame:
        ohlcv = await self.ex.fetch_ohlcv(symbol, timeframe=tf, limit=int(limit))
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    # ------------------------
    # Derivatives helpers
    # ------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
    async def fetch_open_interest_history_5m(
        self,
        symbol: str,
        *,
        lookback_days: int = 7,
        end_ts: Optional[pd.Timestamp] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch open-interest history (5m) from CCXT-bybit if supported.
        Returns list of dicts with at least:
          - timestamp (ms)
          - openInterest or open_interest (depending on adapter)
        """
        end_ts = _ensure_utc_ts(end_ts)
        lookback_days = int(lookback_days)
        if lookback_days <= 0:
            raise ValueError("lookback_days must be > 0")

        tf_ms = _tf_to_ms("5m")
        if end_ts is None:
            # NOTE: call-site should pass end_ts=decision_ts; if missing, this uses exchange clock
            # but downstream must still slice as-of. Kept for backward compatibility.
            end_ms = None
        else:
            end_ms = _to_ms(end_ts)

        # CCXT 'since' is inclusive start; we page forward
        if end_ms is None:
            since_ms = None
        else:
            since_ms = end_ms - lookback_days * 86_400_000

        out: List[Dict[str, Any]] = []

        # Try native endpoint via CCXT (Bybit supports fetchOpenInterestHistory in some builds)
        # Fall back to empty if unavailable; downstream should fail-closed.
        if not hasattr(self.ex, "fetch_open_interest_history"):
            return out

        # Paging loop
        cursor = since_ms
        while True:
            rows = await self.ex.fetch_open_interest_history(
                symbol,
                timeframe="5m",
                since=cursor,
                limit=1000,
            )
            if not rows:
                break

            # rows are dicts
            out.extend(rows)

            # Advance cursor by tf_ms past last timestamp
            last_ts = rows[-1].get("timestamp")
            if last_ts is None:
                break
            cursor = int(last_ts) + tf_ms

            # Stop when we reached end_ms
            if end_ms is not None and cursor > end_ms:
                break

            # Safety
            if len(out) > 50000:
                break

        # Enforce as-of: drop anything beyond end_ts
        if end_ts is not None:
            end_ms = _to_ms(end_ts)
            out = [r for r in out if r.get("timestamp") is not None and int(r["timestamp"]) <= end_ms]

        out.sort(key=lambda d: d["timestamp"])
        return out

    async def fetch_oi_funding_series_5m(
        self,
        symbol: str,
        *,
        lookback_days: Optional[int] = None,
        lookback_oi_days: int = 8,
        lookback_fr_days: int = 8,
        end_ts: Optional[pd.Timestamp] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch open-interest (5m) and funding-rate (event-based) history up to end_ts (as-of),
        and return DataFrames indexed by UTC timestamps.

        Supports both:
        - lookback_days=...
        - lookback_oi_days=..., lookback_fr_days=...
        """
        if lookback_days is not None:
            lookback_oi_days = int(lookback_days)
            lookback_fr_days = int(lookback_days)

        lookback_oi_days = int(lookback_oi_days)
        lookback_fr_days = int(lookback_fr_days)

        end_ts = _ensure_utc_ts(end_ts)

        oi_rows = await self.fetch_open_interest_history_5m(
            symbol,
            lookback_days=lookback_oi_days,
            end_ts=end_ts,
        )
        fr_rows = await self.fetch_funding_rate_history(
            symbol,
            lookback_days=lookback_fr_days,
            end_ts=end_ts,
        )

        # Build OI dataframe
        oi_df = pd.DataFrame(oi_rows or [])
        if oi_df.empty:
            oi_df = pd.DataFrame({"open_interest": pd.Series(dtype="float64")})
            oi_df.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        else:
            if "timestamp" in oi_df.columns:
                oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"], unit="ms", utc=True)
                oi_df = oi_df.set_index("timestamp")

            if "openInterest" in oi_df.columns and "open_interest" not in oi_df.columns:
                oi_df = oi_df.rename(columns={"openInterest": "open_interest"})

            if "open_interest" not in oi_df.columns:
                oi_df = pd.DataFrame({"open_interest": pd.Series(dtype="float64")})
                oi_df.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
            else:
                oi_df["open_interest"] = pd.to_numeric(oi_df["open_interest"], errors="coerce").astype(float)
                oi_df = oi_df[["open_interest"]].sort_index()
                oi_df = oi_df[~oi_df.index.duplicated(keep="last")]

        # Build funding dataframe
        fr_df = pd.DataFrame(fr_rows or [])
        if fr_df.empty:
            fr_df = pd.DataFrame({"funding_rate": pd.Series(dtype="float64")})
            fr_df.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        else:
            if "timestamp" in fr_df.columns:
                fr_df["timestamp"] = pd.to_datetime(fr_df["timestamp"], unit="ms", utc=True)
                fr_df = fr_df.set_index("timestamp")

            if "fundingRate" in fr_df.columns and "funding_rate" not in fr_df.columns:
                fr_df = fr_df.rename(columns={"fundingRate": "funding_rate"})

            if "funding_rate" not in fr_df.columns:
                fr_df = pd.DataFrame({"funding_rate": pd.Series(dtype="float64")})
                fr_df.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
            else:
                fr_df["funding_rate"] = pd.to_numeric(fr_df["funding_rate"], errors="coerce").astype(float)
                fr_df = fr_df[["funding_rate"]].sort_index()
                fr_df = fr_df[~fr_df.index.duplicated(keep="last")]

        # Enforce as-of cutoff
        if end_ts is not None:
            oi_df = oi_df.loc[:end_ts]
            fr_df = fr_df.loc[:end_ts]

        return oi_df, fr_df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
    async def fetch_funding_rate_history(
        self,
        symbol: str,
        *,
        lookback_days: int = 7,
        end_ts: Optional[pd.Timestamp] = None,
        asof_ts: Optional[pd.Timestamp] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Fetch funding-rate history up to end_ts (as-of). Returns list of dicts with:
          - timestamp (ms)
          - fundingRate or funding_rate
        Compatibility:
          - accepts end_ts=...
          - accepts asof_ts=... alias
        """
        import pandas as pd  # local import to avoid any import-cycle surprises

        if end_ts is None and asof_ts is not None:
            end_ts = asof_ts

        end_ts = _ensure_utc_ts(end_ts)
        lookback_days = int(lookback_days)
        if lookback_days <= 0:
            raise ValueError("lookback_days must be > 0")

        # CCXT bybit: fetchFundingRateHistory(symbol, since, limit, params)
        if end_ts is None:
            since_ms = None
            end_ms = None
        else:
            end_ms = _to_ms(end_ts)
            since_ms = end_ms - lookback_days * 86_400_000

        out: List[Dict[str, Any]] = []

        # Some CCXT builds use fetch_funding_rate_history; others use fetchFundingRateHistory naming under the hood.
        if hasattr(self.ex, "fetch_funding_rate_history"):
            rows = await self.ex.fetch_funding_rate_history(symbol, since=since_ms)
            out = list(rows or [])
        elif hasattr(self.ex, "fetch_funding_rate_history"):
            rows = await self.ex.fetch_funding_rate_history(symbol, since=since_ms)
            out = list(rows or [])
        else:
            # No support: fail-closed upstream by returning empty
            return out

        # Enforce as-of cutoff
        if end_ms is not None:
            out = [r for r in out if r.get("timestamp") is not None and int(r["timestamp"]) <= int(end_ms)]

        out.sort(key=lambda d: d.get("timestamp", 0))
        return out
