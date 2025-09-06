"""
oi_funding.py  —  Async helpers to fetch/align OI & funding to a 5m grid and
                   compute the 13 OI+Funding features used by the meta-model.

- Uses ExchangeProxy wrappers (fetch_open_interest_history / fetch_funding_rate_history)
- Reindexes to the bot’s base timeframe (5m) and forward-fills funding
- Computes exactly the 13 numeric features listed in the training spec
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


# ---- constants for 5-minute bars (keep parity with training) ----
WIN_1H, WIN_4H, WIN_1D, WIN_3D, WIN_7D = 12, 48, 288, 864, 2016


def _ceil_to_5m(ts: datetime) -> datetime:
    """Round up to next 5-minute boundary (UTC)."""
    minute = (ts.minute // 5) * 5
    base = ts.replace(second=0, microsecond=0, minute=minute, tzinfo=timezone.utc)
    if base < ts:
        base = base + timedelta(minutes=5)
    return base


def _mk_5m_index(start_utc: datetime, end_utc: datetime) -> pd.DatetimeIndex:
    start_utc = start_utc.replace(second=0, microsecond=0, tzinfo=timezone.utc)
    start_utc = _ceil_to_5m(start_utc - timedelta(minutes=5))
    end_utc   = end_utc.replace(second=0, microsecond=0, tzinfo=timezone.utc)
    end_utc   = end_utc.replace(minute=(end_utc.minute // 5) * 5)
    return pd.date_range(start=start_utc, end=end_utc, freq="5min", tz="UTC")


async def fetch_series_5m(
    exchange, symbol: str, lookback_days_oi: int = 7, lookback_days_funding: int = 7
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (oi_series, funding_series) reindexed to a common 5m UTC grid.

    - OI: fetched directly at 5m (paged if needed) when the venue exposes it;
          otherwise nearest-available step is upsampled with ffill.
    - Funding: fetched from funding history (settlement / discrete records),
               then forward-filled to the 5m grid (training parity).
    """
    now = datetime.now(timezone.utc)
    since_oi  = now - timedelta(days=max(1, lookback_days_oi) + 1)       # pad 1 day
    since_fr  = now - timedelta(days=max(1, lookback_days_funding) + 1)  # pad 1 day

    # Build target 5m grid
    idx5 = _mk_5m_index(since_oi, now)

    # 1) Open Interest history → 5m series
    oi_hist = await exchange.fetch_open_interest_history(
        symbol=symbol,
        timeframe="5m",   # ExchangeProxy handles vendor mapping
        since=int(since_oi.timestamp() * 1000),
        limit=None,       # paged inside ExchangeProxy
        params=None,
    )

    if isinstance(oi_hist, list) and oi_hist:
        # ccxt format: [{timestamp, openInterest, ...}, ...]
        oits = pd.DataFrame(oi_hist)[["timestamp", "openInterest"]]
        oits["timestamp"] = pd.to_datetime(oits["timestamp"], unit="ms", utc=True)
        oi = oits.set_index("timestamp").sort_index()["openInterest"].astype("float64")
    else:
        oi = pd.Series(index=idx5, dtype="float64")  # empty series fallback

    # Reindex OI to 5m grid (ffill to avoid gaps for pct_change windows)
    oi_5m = oi.reindex(idx5).ffill()

    # 2) Funding history → forward-fill to 5m
    fr_hist = await exchange.fetch_funding_rate_history(
        symbol=symbol,
        since=int(since_fr.timestamp() * 1000),
        limit=None,        # paged inside ExchangeProxy
        params=None,
    )
    if isinstance(fr_hist, list) and fr_hist:
        frdf = pd.DataFrame(fr_hist)[["timestamp", "fundingRate"]]
        frdf["timestamp"] = pd.to_datetime(frdf["timestamp"], unit="ms", utc=True)
        fr = frdf.set_index("timestamp").sort_index()["fundingRate"].astype("float64")
    else:
        fr = pd.Series(index=idx5, dtype="float64")

    # Up-sample to 5m and forward-fill (training parity)
    fr_5m = fr.reindex(idx5).ffill()

    return oi_5m, fr_5m


def compute_oi_funding_features(
    df5: pd.DataFrame,
    oi_5m: pd.Series,
    fr_5m: pd.Series,
) -> Dict[str, Any]:
    """
    Compute the 13 OI + funding features on the aligned 5m grid.

    df5 must contain columns: ['close','volume'] aligned to oi_5m / fr_5m.
    NaNs are left as-is; LightGBM handles them.
    """
    # Guard: ensure aligned index
    df = pd.DataFrame(index=df5.index.copy())
    df["close"] = df5["close"].astype("float64")
    df["volume"] = df5["volume"].astype("float64")
    df["oi"] = oi_5m.reindex(df.index).astype("float64")
    df["fr"] = fr_5m.reindex(df.index).astype("float64")

    # Funding already forward-filled; keep raw level
    # 1) OI level & notional
    oi_level = df["oi"].iloc[-1]
    oi_notional_est = df["oi"].iloc[-1] * df["close"].iloc[-1]

    # 2) Percent changes (explicit no fill_method)
    oi_pct_1h = df["oi"].pct_change(WIN_1H)
    oi_pct_4h = df["oi"].pct_change(WIN_4H)
    oi_pct_1d = df["oi"].pct_change(WIN_1D)

    # 3) 7d OI z-score
    oi_mean_7d = df["oi"].rolling(WIN_7D, min_periods=WIN_1D).mean()
    oi_std_7d  = df["oi"].rolling(WIN_7D, min_periods=WIN_1D).std()
    oi_z_7d    = (df["oi"] - oi_mean_7d) / (oi_std_7d + 1e-12)

    # 4) OI change normalized by 1h turnover (volume proxy)
    vol_1h = df["volume"].rolling(WIN_1H).sum()
    oi_chg_norm_vol_1h = (df["oi"] - df["oi"].shift(WIN_1H)) / (vol_1h + 1e-9)

    # 5) OI–price interaction (sign)
    ret_1h = df["close"].pct_change(WIN_1H)
    oi_price_div_1h = np.sign(ret_1h) * oi_pct_1h

    # 6) Funding transforms (already on 5m, ffilled)
    funding_rate = df["fr"]
    funding_abs = funding_rate.abs()
    fr_mean_7d = funding_rate.rolling(WIN_7D, min_periods=WIN_1D).mean()
    fr_std_7d  = funding_rate.rolling(WIN_7D, min_periods=WIN_1D).std()
    funding_z_7d = (funding_rate - fr_mean_7d) / (fr_std_7d + 1e-12)
    funding_rollsum_3d = funding_rate.rolling(WIN_3D, min_periods=WIN_1D).sum()

    # 7) Funding × OI interaction
    funding_oi_div = funding_z_7d * oi_z_7d

    # Output: the LAST available values on the aligned grid
    out = {
        "oi_level": float(oi_level) if np.isfinite(oi_level) else np.nan,
        "oi_notional_est": float(oi_notional_est) if np.isfinite(oi_notional_est) else np.nan,
        "oi_pct_1h": float(oi_pct_1h.iloc[-1]) if np.isfinite(oi_pct_1h.iloc[-1]) else np.nan,
        "oi_pct_4h": float(oi_pct_4h.iloc[-1]) if np.isfinite(oi_pct_4h.iloc[-1]) else np.nan,
        "oi_pct_1d": float(oi_pct_1d.iloc[-1]) if np.isfinite(oi_pct_1d.iloc[-1]) else np.nan,
        "oi_z_7d": float(oi_z_7d.iloc[-1]) if np.isfinite(oi_z_7d.iloc[-1]) else np.nan,
        "oi_chg_norm_vol_1h": float(oi_chg_norm_vol_1h.iloc[-1]) if np.isfinite(oi_chg_norm_vol_1h.iloc[-1]) else np.nan,
        "oi_price_div_1h": float(oi_price_div_1h.iloc[-1]) if np.isfinite(oi_price_div_1h.iloc[-1]) else np.nan,
        "funding_rate": float(funding_rate.iloc[-1]) if np.isfinite(funding_rate.iloc[-1]) else np.nan,
        "funding_abs": float(funding_abs.iloc[-1]) if np.isfinite(funding_abs.iloc[-1]) else np.nan,
        "funding_z_7d": float(funding_z_7d.iloc[-1]) if np.isfinite(funding_z_7d.iloc[-1]) else np.nan,
        "funding_rollsum_3d": float(funding_rollsum_3d.iloc[-1]) if np.isfinite(funding_rollsum_3d.iloc[-1]) else np.nan,
        "funding_oi_div": float(funding_oi_div.iloc[-1]) if np.isfinite(funding_oi_div.iloc[-1]) else np.nan,
    }
    return out
