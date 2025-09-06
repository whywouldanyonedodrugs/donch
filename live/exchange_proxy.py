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

    async def fetch_open_interest_history(self, symbol: str, timeframe: str = "5m",
                                          since: int | None = None, limit: int | None = None, params: dict | None = None):
        """
        Unified wrapper over ccxt.fetchOpenInterestHistory.
        - Bybit expects 'period' like '5min', '15min', '1h' in params; ccxt handles mapping for v5.
        - We still translate common tf strings to vendor flavors for robustness.
        Returns: list of { 'timestamp': ms, 'openInterest': float, ... }
        """
        ex = self._exchange
        if not hasattr(ex, "fetchOpenInterestHistory"):
            return []

        # Map common tfs to Bybit period strings (ccxt usually handles this; keep as fallback)
        tf_map = {"1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
                  "1h": "1h", "4h": "4h", "1d": "1d"}
        period = tf_map.get(str(timeframe), timeframe)

        p = dict(params or {})
        # For Bybit v5, ccxt may expect {'interval': '5', 'unit': 'min'} or 'period': '5min'; ccxt normalizes.
        # Provide both keys harmlessly; the API will ignore extras.
        p.setdefault("period", period)
        try:
            out = await ex.fetchOpenInterestHistory(symbol, timeframe=timeframe, since=since, limit=limit, params=p)
            # Some ccxt versions return a dict with 'data'
            if isinstance(out, dict) and "data" in out:
                out = out["data"]
            return out or []
        except Exception:
            return []

    async def fetch_funding_rate_history(self, symbol: str,
                                         since: int | None = None, limit: int | None = None, params: dict | None = None):
        """
        Unified wrapper over ccxt.fetchFundingRateHistory.
        Returns: list of { 'timestamp': ms, 'fundingRate': float, ... }
        Note: venues typically store settlement points; we forward-fill to 5m later.
        """
        ex = self._exchange
        if not hasattr(ex, "fetchFundingRateHistory"):
            return []
        p = dict(params or {})
        try:
            # fetchFundingRateHistory is often limited (e.g., ~200 rows); ExchangeProxy's callers may page externally.
            out = await ex.fetchFundingRateHistory(symbol, since=since, limit=limit, params=p)
            if isinstance(out, dict) and "data" in out:
                out = out["data"]
            return out or []
        except Exception:
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