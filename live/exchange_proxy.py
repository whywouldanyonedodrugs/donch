# live/exchange_proxy.py
"""
A resilient proxy wrapper for the CCXT exchange object.

It uses the 'tenacity' library to automatically retry API calls that fail
due to temporary, recoverable network issues.
"""
import pandas as pd
import math
from datetime import timedelta
import logging
import asyncio
from functools import wraps
import ccxt.async_support as ccxt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)

_TIMEFRAME_MS_CACHE: dict[str, int] = {}

# Define the specific, temporary errors we want to retry on.
# We should NOT retry on things like "Invalid API Key" or "Insufficient Funds".
RETRYABLE_EXCEPTIONS = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RequestTimeout,
    ccxt.DDoSProtection,
)

class ExchangeProxy:
    """
    Wraps a ccxt exchange instance to provide automatic retries on network errors.
    """
    def __init__(self, exchange: ccxt.Exchange):
        self._exchange = exchange

    @property
    def markets(self):
        """Pass through to the underlying exchange's markets property."""
        return self._exchange.markets

    def _end_ms(self, end_ts: Optional[pd.Timestamp]) -> int:
        """
        Convert a tz-aware pd.Timestamp to ms since epoch.
        If end_ts is None, fall back to exchange.milliseconds().
        """
        if end_ts is None:
            return self._exchange.milliseconds()
        if not isinstance(end_ts, pd.Timestamp):
            end_ts = pd.Timestamp(end_ts)
        if end_ts.tz is None:
            # enforce UTC if caller passed naive
            end_ts = end_ts.tz_localize("UTC")
        return int(end_ts.timestamp() * 1000)

    async def _bybit_resolve_symbol_and_category(self, symbol: str) -> Tuple[str, str, str]:
        """
        Returns (unified_symbol, id_symbol, category) for Bybit.
        unified_symbol: ccxt unified, e.g. 'BTC/USDT:USDT'
        id_symbol: exchange id, e.g. 'BTCUSDT'
        category: 'linear' or 'inverse' (default 'linear')
        If not Bybit or markets not available, returns best-effort fallbacks.
        """
        ex = self._exchange
        unified = symbol
        sym_id = symbol
        category = "linear"

        if getattr(ex, "id", None) != "bybit":
            return unified, sym_id, category

        # best-effort markets load
        try:
            if not getattr(ex, "markets", None):
                await ex.load_markets()
        except Exception:
            return unified, sym_id, category

        m = None
        try:
            if "/" in symbol:
                m = ex.market(symbol)
            else:
                m = (getattr(ex, "markets_by_id", {}) or {}).get(symbol)
        except Exception:
            m = None

        if isinstance(m, dict):
            unified = m.get("symbol") or unified
            sym_id = m.get("id") or sym_id
            if m.get("inverse"):
                category = "inverse"
            elif m.get("linear"):
                category = "linear"

        return unified, sym_id, category



    def __getattr__(self, name):
        """
        Intercepts any call to a method that doesn't exist on the Proxy,
        retrieves it from the underlying exchange object, and wraps it
        in our retry logic.
        """
        original_attr = getattr(self._exchange, name)

        if not callable(original_attr):
            return original_attr

        @wraps(original_attr)
        def wrapper(*args, **kwargs):
            # Define the retry decorator dynamically
            retry_decorator = retry(
                wait=wait_exponential(multiplier=1, min=2, max=30),
                stop=stop_after_attempt(5),
                retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
                before_sleep=lambda state: LOG.warning(
                    "Retrying API call %s due to %s. Attempt #%d",
                    name, state.outcome.exception(), state.attempt_number
                )
            )

            if asyncio.iscoroutinefunction(original_attr):
                # Apply decorator to an async function
                @retry_decorator
                async def async_call():
                    return await original_attr(*args, **kwargs)
                return async_call()
            else:
                # Apply decorator to a sync function
                @retry_decorator
                def sync_call():
                    return original_attr(*args, **kwargs)
                return sync_call()
        # Cache the newly created wrapper function on the instance.
        # The next call to this method will use the cached version
        # instead of triggering __getattr__ again.
        setattr(self, name, wrapper)
        return wrapper

    # ─────────────────────────────────────────────────────────────────────
    # Open Interest & Funding History helpers (Bybit V5 + CCXT unified)
    # ─────────────────────────────────────────────────────────────────────
    async def fetch_open_interest_history_5m(
        self,
        symbol: str,
        lookback_days: int = 7,
        end_ts: Optional[pd.Timestamp] = None,
        category: str = "linear",
        limit: int = 200,
    ) -> list[dict]:
        """
        Fetch Bybit V5 open-interest history at 5min resolution.

        Returns list of dicts:
          {"timestamp": <ms int>, "openInterest": <float>}

        Semantics:
        - Uses startTime/endTime window derived from end_ts and lookback_days.
        - Paginates using nextPageCursor until exhausted.
        - Filters returned timestamps to <= end_ts (no lookahead).
        """
        ex = self.ex
        if ex is None:
            raise RuntimeError("ExchangeProxy.ex is None")

        if end_ts is None:
            end_ts = pd.Timestamp.now(tz="UTC").floor("5min")
        else:
            end_ts = pd.Timestamp(end_ts)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            else:
                end_ts = end_ts.tz_convert("UTC")
            end_ts = end_ts.floor("5min")

        if lookback_days <= 0:
            return []

        start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)

        # Prefer Bybit V5 raw endpoint (documented params: category/symbol/intervalTime/startTime/endTime/limit/cursor)
        # https://bybit-exchange.github.io/docs/v5/market/open-interest :contentReference[oaicite:1]{index=1}
        if not hasattr(ex, "publicGetV5MarketOpenInterest"):
            # Fall back to unified method if present
            if getattr(ex, "has", {}).get("fetchOpenInterestHistory"):
                # Try to resolve CCXT symbol if markets are loaded
                try:
                    if not getattr(ex, "markets", None):
                        await ex.load_markets()
                    sym_ccxt = symbol
                    if hasattr(ex, "markets_by_id") and symbol in ex.markets_by_id:
                        sym_ccxt = ex.markets_by_id[symbol]["symbol"]
                except Exception:
                    sym_ccxt = symbol

                rows = await ex.fetch_open_interest_history(
                    sym_ccxt,
                    timeframe="5m",
                    since=start_ms,
                    limit=limit,
                    params={"category": category},
                )
                out: list[dict] = []
                for r in rows or []:
                    ts = r.get("timestamp")
                    oi = r.get("openInterest") or r.get("open_interest") or r.get("openInterestValue")
                    if ts is None or oi is None:
                        continue
                    ts = int(ts)
                    if ts <= end_ms:
                        out.append({"timestamp": ts, "openInterest": float(oi)})
                out.sort(key=lambda x: x["timestamp"])
                return out

            raise NotImplementedError(
                "No supported open-interest method (missing publicGetV5MarketOpenInterest and fetchOpenInterestHistory)"
            )

        cursor: Optional[str] = None
        out: list[dict] = []

        while True:
            params = {
                "category": category,
                "symbol": symbol,
                "intervalTime": "5min",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": int(limit),
            }
            if cursor:
                params["cursor"] = cursor

            resp = await ex.publicGetV5MarketOpenInterest(params)
            if (resp or {}).get("retCode") not in (0, "0", None):
                raise RuntimeError(f"Bybit open-interest retCode={resp.get('retCode')} retMsg={resp.get('retMsg')}")

            result = (resp or {}).get("result", {}) or {}
            lst = result.get("list", []) or []
            next_cursor = result.get("nextPageCursor", "") or ""

            if not lst:
                break

            for r in lst:
                try:
                    ts_ms = int(r["timestamp"])
                    if ts_ms > end_ms:
                        continue
                    oi = float(r["openInterest"])
                    out.append({"timestamp": ts_ms, "openInterest": oi})
                except Exception:
                    continue

            if not next_cursor:
                break
            cursor = next_cursor

        out.sort(key=lambda x: x["timestamp"])
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

        # Normalize end_ts
        if end_ts is not None:
            end_ts = pd.to_datetime(end_ts, utc=True)
            # Defensive: if caller passed naive ts, force UTC
            if getattr(end_ts, "tzinfo", None) is None:
                end_ts = end_ts.tz_localize("UTC")

        # Fetch raw history
        oi_rows = await self.fetch_open_interest_history_5m(symbol, lookback_days=lookback_days)
        fr_rows = await self.fetch_funding_rate_history(symbol, lookback_days=lookback_days)

        # Build OI dataframe
        oi_df = pd.DataFrame(oi_rows or [])
        if oi_df.empty:
            oi_df = pd.DataFrame({"open_interest": pd.Series(dtype="float64")})
            oi_df.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        else:
            if "timestamp" in oi_df.columns:
                oi_df["timestamp"] = pd.to_datetime(oi_df["timestamp"], unit="ms", utc=True)
                oi_df = oi_df.set_index("timestamp")

            # Normalize column name(s)
            if "openInterest" in oi_df.columns and "open_interest" not in oi_df.columns:
                oi_df = oi_df.rename(columns={"openInterest": "open_interest"})

            if "open_interest" not in oi_df.columns:
                # Fail-closed by returning empty; downstream will treat as missing/stale
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

        # Truncate to end_ts to preserve "as-of" semantics (prevents staleness masking)
        if end_ts is not None:
            oi_df = oi_df.loc[:end_ts]
            fr_df = fr_df.loc[:end_ts]

        return oi_df, fr_df



    async def fetch_funding_rate_history(self, symbol: str, *, lookback_days: int = 7) -> list[dict]:
        """
        Return [{"timestamp": ms, "fundingRate": float}, ...], newest last.
        """
        ex = self._exchange

        # Try CCXT unified first
        if getattr(ex, "has", {}).get("fetchFundingRateHistory"):
            rows = await ex.fetchFundingRateHistory(symbol)
            if not rows:
                return []
            out = []
            if isinstance(rows[0], dict):
                for r in rows:
                    ts = int(r.get("timestamp") or r.get("time") or 0)
                    fr = r.get("fundingRate") or r.get("rate")
                    try: fr = float(fr)
                    except Exception: fr = None
                    if ts and fr is not None:
                        out.append({"timestamp": ts, "fundingRate": fr})
            else:
                for ts, rate, *_ in rows:
                    out.append({"timestamp": int(ts), "fundingRate": float(rate)})
            out.sort(key=lambda x: x["timestamp"])
            return out

        # Fallback: Bybit V5 public endpoint
        if ex.id == "bybit" and hasattr(ex, "publicGetV5MarketHistoryFundRate"):
            now_ms = ex.milliseconds()
            start_ms = now_ms - lookback_days * 24 * 60 * 60 * 1000
            cursor = None
            result: list[dict] = []
            while True:


                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "endTime": end_ms,
                    "limit": 200,
                }




                if cursor:
                    params["cursor"] = cursor
                resp = await ex.publicGetV5MarketHistoryFundRate(params)
                lst = (((resp or {}).get("result") or {}).get("list")) or []
                for row in lst:
                    ts = int(row.get("fundingRateTimestamp") or row.get("timestamp") or 0)
                    fr = row.get("fundingRate")
                    try: fr = float(fr)
                    except Exception: fr = None
                    if ts and fr is not None:
                        result.append({"timestamp": ts, "fundingRate": fr})
                cursor = (((resp or {}).get("result") or {}).get("nextPageCursor")) or ""
                if not cursor:
                    break
                await asyncio.sleep(0.1)
            result.sort(key=lambda x: x["timestamp"])
            return result

        return []



    async def close(self):
        """Gracefully close the underlying exchange connection."""
        await self._exchange.close()

# ---------------------------------------------------------------------------
# utils: fetch_ohlcv_paginated
# ---------------------------------------------------------------------------

def _timeframe_ms(tf: str) -> int:
    """
    Return the duration of one candle in **milliseconds** for a ccxt timeframe
    string (e.g. '5m', '1h', '4h').  Memoised for speed.
    """
    if tf in _TIMEFRAME_MS_CACHE:
        return _TIMEFRAME_MS_CACHE[tf]

    unit = tf[-1]
    value = int(tf[:-1])
    if unit == "m":
        ms = value * 60_000
    elif unit == "h":
        ms = value * 60 * 60_000
    elif unit == "d":
        ms = value * 24 * 60 * 60_000
    else:
        raise ValueError(f"Unsupported timeframe: {tf}")
    _TIMEFRAME_MS_CACHE[tf] = ms
    return ms


async def fetch_ohlcv_paginated(
    exchange,
    symbol: str,
    timeframe: str,
    wanted: int,
    *,
    since: int | None = None,
    max_batch: int = 200,
    sleep_sec: float = 0.05,
) -> list[list]:
    """
    Fetch **wanted** historical candles even when the exchange caps `limit`
    (Bybit v5 returns 200 rows max for TF < 1h).

    Returns a list **oldest → newest** compatible with the ccxt `fetch_ohlcv`
    format.  Works for any timeframe and any exchange that obeys `since`
    semantics (most do).

    - Uses `since` going *backwards* from `since or now`.
    - Stops when `wanted` rows have been collected OR the exchange sends
      fewer than `max_batch` rows (meaning you hit listing date).
    """
    all_rows: list[list] = []
    tf_ms = _timeframe_ms(timeframe)
    now = exchange.milliseconds()

    # if since not given, start from "now" rounded down to nearest candle
    cursor = since or (now - (now % tf_ms))

    while len(all_rows) < wanted:
        batch_limit = min(max_batch, wanted - len(all_rows))
        rows = await exchange.fetch_ohlcv(
            symbol, timeframe, since=cursor - tf_ms * batch_limit, limit=batch_limit
        )
        if not rows:
            break  # no more history

        # When fetching with `since`, Bybit returns newest→oldest; reverse
        rows.reverse()
        # Drop the *newest* row if it is the same timestamp as last append
        if all_rows and rows and rows[-1][0] >= all_rows[0][0]:
            rows = rows[:-1]
        all_rows = rows + all_rows

        if len(rows) < batch_limit:
            break  # hit listing date
        cursor = rows[0][0]  # oldest timestamp in this batch
        await asyncio.sleep(sleep_sec)  # be gentle with rate‑limits

    return all_rows[-wanted:] if len(all_rows) >= wanted else all_rows