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


RETRYABLE_EXCEPTIONS = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RequestTimeout,
    ccxt.DDoSProtection,
)

class ExchangeProxy:


    def __init__(self, exchange: ccxt.Exchange):
        self._exchange = exchange

    @property
    def markets(self):

        return self._exchange.markets
    @property
    def ex(self):

        return self._exchange

    def _end_ms(self, end_ts: Optional[pd.Timestamp]) -> int:


        if end_ts is None:
            return self._exchange.milliseconds()
        if not isinstance(end_ts, pd.Timestamp):
            end_ts = pd.Timestamp(end_ts)
        if end_ts.tz is None:

            end_ts = end_ts.tz_localize("UTC")
        return int(end_ts.timestamp() * 1000)

    async def _bybit_resolve_symbol_and_category(self, symbol: str) -> Tuple[str, str, str]:


        ex = self._exchange
        unified = symbol
        sym_id = symbol
        category = "linear"

        if getattr(ex, "id", None) != "bybit":
            return unified, sym_id, category


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


        original_attr = getattr(self._exchange, name)

        if not callable(original_attr):
            return original_attr

        @wraps(original_attr)
        def wrapper(*args, **kwargs):

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

                @retry_decorator
                async def async_call():
                    return await original_attr(*args, **kwargs)
                return async_call()
            else:

                @retry_decorator
                def sync_call():
                    return original_attr(*args, **kwargs)
                return sync_call()


        setattr(self, name, wrapper)
        return wrapper


    async def fetch_open_interest_history_5m(
        self,
        symbol: str,
        lookback_days: int = 7,
        *,
        end_ts: "Optional[pd.Timestamp]" = None,
        limit: int = 200,
    ) -> "List[Dict[str, Any]]":


        if lookback_days is None:
            raise ValueError("lookback_days must not be None")

        ex = self._exchange


        fn = getattr(ex, "publicGetV5MarketOpenInterest", None)
        if fn is None:
            return []


        end_ms = self._end_ms(end_ts)
        start_ms = end_ms - int(lookback_days) * 24 * 60 * 60 * 1000


        try:

            unified_symbol, sym_id, category = await self._bybit_resolve_symbol_and_category(symbol)
        except Exception:
            sym_id, category = symbol, "linear"

        step_ms = 5 * 60 * 1000
        page_span_ms = limit * step_ms

        out: list[dict] = []
        seen_ts: set[int] = set()

        page_end_ms = end_ms
        while page_end_ms > start_ms:
            page_start_ms = max(start_ms, page_end_ms - page_span_ms)

            params = {
                "category": category,
                "symbol": sym_id,
                "intervalTime": "5min",
                "startTime": int(page_start_ms),
                "endTime": int(page_end_ms),
                "limit": int(limit),
            }

            resp = await fn(params)
            result = (resp or {}).get("result") or {}
            lst = result.get("list") or []
            if not lst:
                break


            oldest_ms = None
            for row in lst:
                try:
                    ts = int(row.get("timestamp") or 0)
                    oi = float(row.get("openInterest")) if row.get("openInterest") is not None else None
                except Exception:
                    continue

                if ts <= 0 or oi is None:
                    continue
                if ts < start_ms or ts > end_ms:
                    continue

                if ts not in seen_ts:
                    out.append({"timestamp": ts, "openInterest": oi})
                    seen_ts.add(ts)

                if oldest_ms is None or ts < oldest_ms:
                    oldest_ms = ts

            if oldest_ms is None:
                break
            if oldest_ms <= start_ms:
                break


            page_end_ms = oldest_ms - 1


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


        if lookback_days is not None:
            lookback_oi_days = int(lookback_days)
            lookback_fr_days = int(lookback_days)

        lookback_oi_days = int(lookback_oi_days)
        lookback_fr_days = int(lookback_fr_days)


        if end_ts is not None:
            end_ts = pd.to_datetime(end_ts, utc=True)

            if getattr(end_ts, "tzinfo", None) is None:
                end_ts = end_ts.tz_localize("UTC")


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


        if end_ts is not None:
            oi_df = oi_df.loc[:end_ts]
            fr_df = fr_df.loc[:end_ts]

        return oi_df, fr_df

    async def fetch_funding_rate_history(
        self,
        symbol: str,
        *,
        lookback_days: int = 7,
        end_ts=None,
        asof_ts=None,
        **kwargs,
    ):


        import pandas as pd

        if end_ts is None and asof_ts is not None:
            end_ts = asof_ts

        if lookback_days is None:
            raise ValueError("lookback_days must not be None")


        if end_ts is None:
            end_ms = self.ex.milliseconds()
        else:
            if isinstance(end_ts, pd.Timestamp):
                ts = end_ts
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                end_ms = int(ts.timestamp() * 1000)
            else:

                end_ms = int(end_ts)

        start_ms = end_ms - int(lookback_days) * 24 * 60 * 60 * 1000


        if self.ex.has.get("fetchFundingRateHistory"):


            return await self.ex.fetch_funding_rate_history(symbol, since=start_ms)


        if getattr(self.ex, "id", "") == "bybit":
            return await self.ex.public_get_v5_market_funding_history({
                "symbol": symbol.replace("/", ""),
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 200,
            })

        raise NotImplementedError("Funding rate history not supported on this exchange adapter")

    async def close(self):

        await self._exchange.close()


def _timeframe_ms(tf: str) -> int:


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


    all_rows: list[list] = []
    tf_ms = _timeframe_ms(timeframe)
    now = exchange.milliseconds()


    cursor = since or (now - (now % tf_ms))

    while len(all_rows) < wanted:
        batch_limit = min(max_batch, wanted - len(all_rows))
        rows = await exchange.fetch_ohlcv(
            symbol, timeframe, since=cursor - tf_ms * batch_limit, limit=batch_limit
        )
        if not rows:
            break


        rows.reverse()

        if all_rows and rows and rows[-1][0] >= all_rows[0][0]:
            rows = rows[:-1]
        all_rows = rows + all_rows

        if len(rows) < batch_limit:
            break
        cursor = rows[0][0]
        await asyncio.sleep(sleep_sec)

    return all_rows[-wanted:] if len(all_rows) >= wanted else all_rows
