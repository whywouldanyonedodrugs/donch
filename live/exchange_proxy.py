# live/exchange_proxy.py
"""
A resilient proxy wrapper for the CCXT exchange object.

It uses the 'tenacity' library to automatically retry API calls that fail
due to temporary, recoverable network issues.
"""

import math
from datetime import timedelta
import logging
import asyncio
from functools import wraps
import ccxt.async_support as ccxt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
    async def fetch_open_interest_history_5m(self, symbol: str, *, lookback_days: int = 7) -> list[dict]:
        """
        Return a list of dicts: [{"timestamp": ms, "openInterest": float}, ...] on a 5-minute grid,
        newest last. Robust to both CCXT unified and direct Bybit V5 endpoints.
        """
        ex = self._exchange
        interval = "5m"  # CCXT timeframe
        intervalTime = "5min"  # Bybit V5 param

        # Try CCXT unified first
        if getattr(ex, "has", {}).get("fetchOpenInterestHistory"):
            # CCXT returns newest-first; we normalize to newest-last
            rows = await ex.fetchOpenInterestHistory(symbol, timeframe=interval)
            if not rows:
                return []
            # rows may be list of dicts or list of [ts, oi]
            out = []
            if isinstance(rows[0], dict):
                for r in rows:
                    ts = int(r.get("timestamp") or r.get("time") or r.get("datetime") or 0)
                    oi = r.get("openInterest") or r.get("open_interest") or r.get("value") or r.get("openInterestValue")
                    try: oi = float(oi)
                    except Exception: oi = None
                    if ts and oi is not None:
                        out.append({"timestamp": ts, "openInterest": oi})
            else:
                for ts, oi, *_ in rows:
                    out.append({"timestamp": int(ts), "openInterest": float(oi)})
            out.sort(key=lambda x: x["timestamp"])
            return out

        # Fallback: direct Bybit V5 public endpoint through CCXT (async)
        if ex.id == "bybit" and hasattr(ex, "publicGetV5MarketOpenInterest"):
            now_ms = ex.milliseconds()
            start_ms = now_ms - lookback_days * 24 * 60 * 60 * 1000
            cursor = None
            result: list[dict] = []
            while True:
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "intervalTime": intervalTime,
                    "startTime": start_ms,
                    "endTime": now_ms,
                    "limit": 200,
                }
                if cursor:
                    params["cursor"] = cursor
                resp = await ex.publicGetV5MarketOpenInterest(params)
                lst = (((resp or {}).get("result") or {}).get("list")) or []
                for row in lst:
                    ts = int(row.get("timestamp") or 0)
                    oi = row.get("openInterest")
                    try: oi = float(oi)
                    except Exception: oi = None
                    if ts and oi is not None:
                        result.append({"timestamp": ts, "openInterest": oi})
                cursor = (((resp or {}).get("result") or {}).get("nextPageCursor")) or ""
                if not cursor:
                    break
                await asyncio.sleep(0.1)
            result.sort(key=lambda x: x["timestamp"])
            return result

        # Unsupported exchange
        return []

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
                    "startTime": start_ms,
                    "endTime": now_ms,
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