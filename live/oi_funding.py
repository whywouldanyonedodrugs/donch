# live/oi_funding.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

WIN_1H, WIN_4H, WIN_1D, WIN_3D, WIN_7D = 12, 48, 288, 864, 2016

def _as_df(items: list[dict], ts_key: str, val_key: str) -> pd.DataFrame:
    """
    Accepts list of dicts or list of lists; returns DataFrame with UTC ms index
    and a single value column (float).
    """
    if not items:
        return pd.DataFrame(columns=[val_key])
    # list of dicts
    if isinstance(items[0], dict):
        df = pd.DataFrame(items)
        # normalize possible variants of keys
        if ts_key not in df.columns:
            for k in ("time", "timestamp", "fundingRateTimestamp"):
                if k in df.columns:
                    ts_key = k; break
        if val_key not in df.columns:
            # open interest possible aliases
            alt = ["open_interest", "value", "openInterestValue", "oi", "rate", "funding_rate"]
            for k in alt:
                if k in df.columns:
                    val_key = k; break
        df = df[[ts_key, val_key]].copy()
        df[ts_key] = pd.to_numeric(df[ts_key], errors="coerce").astype("Int64")
        df[val_key] = pd.to_numeric(df[val_key], errors="coerce")
    else:
        # list of lists: assume [timestamp, value, ...]
        df = pd.DataFrame(items, columns=[ts_key, val_key])
        df[ts_key] = pd.to_numeric(df[ts_key], errors="coerce").astype("Int64")
        df[val_key] = pd.to_numeric(df[val_key], errors="coerce")
    df.dropna(subset=[ts_key], inplace=True)
    df.set_index(pd.to_datetime(df[ts_key].astype("int64"), unit="ms", utc=True), inplace=True)
    df = df.drop(columns=[ts_key]).sort_index()
    return df

async def fetch_series_5m(exchange, symbol: str, lookback_oi_days: int = 7, lookback_fr_days: int = 7) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (oi_series_5m, funding_series_5m) aligned to 5m UTC timestamps.
    """
    oi_hist = await exchange.fetch_open_interest_history_5m(symbol, lookback_days=lookback_oi_days)
    fr_hist = await exchange.fetch_funding_rate_history(symbol, lookback_days=lookback_fr_days)

    oi_df = _as_df(oi_hist, "timestamp", "openInterest")
    fr_df = _as_df(fr_hist, "timestamp", "fundingRate")

    # 5-minute grid index covering the union of inputs
    if not oi_df.empty:
        start = oi_df.index.min()
    else:
        start = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=lookback_oi_days)
    end = pd.Timestamp.utcnow().tz_localize("UTC")
    idx5 = pd.date_range(start=start.floor("5min"), end=end.floor("5min"), freq="5min", tz="UTC")

    oi5 = oi_df.reindex(idx5)["openInterest"]
    fr5 = fr_df.reindex(idx5)["fundingRate"]

    # Forward-fill funding to every 5m bar (as per design)
    fr5 = fr5.ffill()

    return oi5, fr5

def compute_oi_funding_features(df5: pd.DataFrame, oi5: pd.Series, fr5: pd.Series, *, allow_nans: bool = True) -> Dict[str, float]:
    """
    Compute the 13 new OI/Funding features on a 5m grid for the **last bar** of df5.
    If allow_nans is False, NaNs are replaced with 0.0 (to satisfy strict parity).
    """
    # align all to df5 index (5m UTC)
    oi = oi5.reindex(df5.index)
    fr = fr5.reindex(df5.index)

    close = df5["close"].astype(float)
    volume = df5.get("volume", pd.Series(index=df5.index, dtype=float)).astype(float)

    # 1-2) levels
    oi_level        = oi
    oi_notional_est = oi * close

    # 3-5) pct changes
    oi_pct_1h = oi.pct_change(WIN_1H, fill_method=None)
    oi_pct_4h = oi.pct_change(WIN_4H, fill_method=None)
    oi_pct_1d = oi.pct_change(WIN_1D, fill_method=None)

    # 6) OI z-score (7d)
    oi_mean_7d = oi.rolling(WIN_7D, min_periods=WIN_1D).mean()
    oi_std_7d  = oi.rolling(WIN_7D, min_periods=WIN_1D).std()
    oi_z_7d    = (oi - oi_mean_7d) / (oi_std_7d + 1e-12)

    # 7) ΔOI normalized by recent turnover (1h)
    vol_1h = volume.rolling(WIN_1H).sum()
    oi_chg_norm_vol_1h = (oi - oi.shift(WIN_1H)) / (vol_1h + 1e-9)

    # 8) OI–price interaction
    ret_1h          = close.pct_change(WIN_1H, fill_method=None)
    oi_price_div_1h = np.sign(ret_1h) * oi_pct_1h

    # 9-12) Funding transforms
    funding_rate       = fr
    funding_abs        = fr.abs()
    fr_mean_7d         = fr.rolling(WIN_7D, min_periods=WIN_1D).mean()
    fr_std_7d          = fr.rolling(WIN_7D, min_periods=WIN_1D).std()
    funding_z_7d       = (fr - fr_mean_7d) / (fr_std_7d + 1e-12)
    funding_rollsum_3d = fr.rolling(WIN_3D, min_periods=WIN_1D).sum()

    # 13) Interaction
    funding_oi_div = funding_z_7d * oi_z_7d

    # last bar snapshot
    fields = {
        "oi_level":            oi_level.iloc[-1],
        "oi_notional_est":     oi_notional_est.iloc[-1],
        "oi_pct_1h":           oi_pct_1h.iloc[-1],
        "oi_pct_4h":           oi_pct_4h.iloc[-1],
        "oi_pct_1d":           oi_pct_1d.iloc[-1],
        "oi_z_7d":             oi_z_7d.iloc[-1],
        "oi_chg_norm_vol_1h":  oi_chg_norm_vol_1h.iloc[-1],
        "oi_price_div_1h":     oi_price_div_1h.iloc[-1],
        "funding_rate":        funding_rate.iloc[-1],
        "funding_abs":         funding_abs.iloc[-1],
        "funding_z_7d":        funding_z_7d.iloc[-1],
        "funding_rollsum_3d":  funding_rollsum_3d.iloc[-1],
        "funding_oi_div":      funding_oi_div.iloc[-1],
    }
    if not allow_nans:
        for k, v in list(fields.items()):
            if v is None or not np.isfinite(float(v)):
                fields[k] = 0.0
    return fields
