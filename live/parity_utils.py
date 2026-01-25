# live/parity_utils.py
import pandas as pd
import numpy as np
from . import indicators as ta

def _norm_tf(tf: str) -> str:
    """
    Normalize timeframe strings to pandas-safe aliases.
    Prevents '15m' being interpreted as 15 months.
    """
    tf = str(tf).strip()
    mapping = {
        "1m": "1min", "3m": "3min", "5m": "5min",
        "15m": "15min", "30m": "30min",
        "1h": "1h", "2h": "2h", "4h": "4h",
        "1d": "1D", "1D": "1D",
    }
    if tf in mapping:
        return mapping[tf]

    if tf.endswith("m") and not tf.endswith("min"):
        return tf[:-1] + "min"

    if tf.lower().endswith("d"):
        return tf.upper()

    return tf

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Standard resampling (Label='left', Closed='left').
    Used for Entry Quality features (ATR, RSI, ADX, etc.) and days_since_prev_break.
    """
    if df is None or df.empty:
        return df

    freq = _norm_tf(timeframe)

    # Explicit left/left to match offline/scout behavior
    res = df.resample(freq, label="left", closed="left")
    agg = res.agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return agg.dropna()


def resample_ohlcv_robust(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Robust resampling (Label='right', Closed='right').
    Matches offline 'scout' usage for Regime detection and Donchian 'completed days' logic.
    """
    if df.empty:
        return df

    freq = _norm_tf(timeframe)
    res = df.resample(freq, label="right", closed="right")
    agg = res.agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return agg.dropna()

def map_to_left_index(target_index: pd.DatetimeIndex, source_series: pd.Series) -> pd.Series:
    """
    Forward-fill source_series onto target_index.
    Matches offline indicators.map_to_left_index.
    """
    return source_series.reindex(target_index, method="ffill")

def donchian_upper_days_no_lookahead(high_5m: pd.Series, n_days: int) -> pd.Series:
    """
    Daily Donchian upper on *completed* days only.
    Matches offline scout.py logic (Right/Right + dropna + shift).

    Returns a pd.Series indexed by high_5m.index (wrapping the numpy array from scout logic).
    """
    daily_high = high_5m.resample("1D", label="right", closed="right").max().dropna()
    don_daily = daily_high.rolling(n_days, min_periods=n_days).max().shift(1)

    # Map to 5m bars by day start, then ffill through the day.
    keyed = don_daily.reindex(high_5m.index.floor("D"))
    arr = keyed.ffill().to_numpy(dtype=float)
    return pd.Series(arr, index=high_5m.index)
