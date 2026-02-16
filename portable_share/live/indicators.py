from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated.*")


if not hasattr(np, "NaN"):
    np.NaN = np.nan


try:
    import talib
    _HAS_TA = True
except ImportError:
    try:
        import pandas_ta as pta
    except ImportError as exc:
        raise ImportError("Neither TA‑Lib nor pandas_ta is installed.\n"
                          "Run `pip install ta-lib‑binary` (Windows) or `pip install pandas_ta`."
                          ) from exc
    _HAS_TA = False


__all__ = ["ema", "atr", "rsi", "macd", "bollinger", "lbr_310", "adx"]


def sma(x: pd.Series, length: int) -> pd.Series:


    length = int(length)
    if length <= 0:
        raise ValueError("length must be > 0")
    s = pd.to_numeric(x, errors="coerce").astype(float)
    return s.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    if _HAS_TA:
        return pd.Series(talib.EMA(series, timeperiod=span), index=series.index)
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    if _HAS_TA:
        return pd.Series(talib.ATR(df["high"], df["low"], df["close"], timeperiod=period), index=df.index)
    atr_series = pta.atr(high=df["high"], low=df["low"], close=df["close"], length=period)
    if atr_series is None:
        return pd.Series(dtype='float64', index=df.index)
    return atr_series


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if _HAS_TA:
        return pd.Series(talib.RSI(series, timeperiod=period), index=series.index)
    rsi_series = pta.rsi(series, length=period)
    if rsi_series is None:
        return pd.Series(dtype='float64', index=series.index)
    return rsi_series

def macd(series: pd.Series) -> pd.DataFrame:


    if _HAS_TA:
        macd, sig, hist = talib.MACD(series)
        return pd.DataFrame(
            {"macd": macd, "signal": sig, "hist": hist}, index=series.index
        )

    df_raw = pta.macd(series)
    mapping = {}
    for col in df_raw.columns:
        if "MACDh" in col or "hist" in col: mapping[col] = "hist"
        elif "MACDs" in col: mapping[col] = "signal"
        elif "MACD" in col: mapping[col] = "macd"
    df = df_raw.rename(columns=mapping)

    if "hist" not in df.columns: df["hist"] = df["macd"] - df["signal"]
    if "macd" not in df.columns: df["macd"] = np.nan
    if "signal" not in df.columns: df["signal"] = np.nan

    return df[["macd", "signal", "hist"]]


def bollinger(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    if _HAS_TA:
        upper, mid, lower = talib.BBANDS(series, timeperiod=length, nbdevup=std, nbdevdn=std)
        return pd.DataFrame({"upper": upper, "mid": mid, "lower": lower}, index=series.index)

    bbands = pta.bbands(series, length=length, std=std)

    if bbands is None or bbands.empty:
        return pd.DataFrame(columns=["upper", "mid", "lower"], index=series.index)

    upper_col = [col for col in bbands.columns if 'BBU' in col.upper()]
    mid_col = [col for col in bbands.columns if 'BBM' in col.upper()]
    lower_col = [col for col in bbands.columns if 'BBL' in col.upper()]

    if not (upper_col and mid_col and lower_col):
         return pd.DataFrame(columns=["upper", "mid", "lower"], index=series.index)

    return pd.DataFrame({
        "upper": bbands[upper_col[0]],
        "mid": bbands[mid_col[0]],
        "lower": bbands[lower_col[0]],
    }, index=series.index)


def lbr_310(series: pd.Series) -> pd.Series:

    return series.rolling(3).mean() - series.rolling(10).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:


    if len(df) < period:
        return pd.Series(dtype='float64', index=df.index)

    if _HAS_TA:
        adx_series = talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)
        return pd.Series(adx_series, index=df.index)


    adx_df = pta.adx(high=df["high"], low=df["low"], close=df["close"], length=period)
    if adx_df is None or adx_df.empty:
        return pd.Series(dtype='float64', index=df.index)


    adx_col = [col for col in adx_df.columns if 'ADX' in col.upper()]
    if not adx_col:
        return pd.Series(dtype='float64', index=df.index)

    return adx_df[adx_col[0]]

def vwap_stack_features(df: pd.DataFrame, lookback_bars: int = 12, band_pct: float = 0.004):


    out0 = {"vwap_frac_in_band": 0.0, "vwap_expansion_pct": 0.0, "vwap_slope_pph": 0.0}

    if df is None or df.empty:
        return out0


    df = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n = len(df)

    if n < lookback_bars + 2:
        return out0


    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].fillna(0.0)


    tpv = (tp * vol).rolling(lookback_bars, min_periods=lookback_bars).sum()
    vv = vol.rolling(lookback_bars, min_periods=lookback_bars).sum()


    if not np.isfinite(vv.iloc[-1]) or vv.iloc[-1] == 0:
        return out0


    cur_vwap = (tpv / vv).iloc[-1]
    cur_close = float(df["close"].iloc[-1])
    expansion = abs(cur_close / cur_vwap - 1.0) if np.isfinite(cur_vwap) and cur_vwap != 0 else 0.0


    prior_slice = slice(-lookback_bars - 1, -1)

    rvwap = (tpv / vv)
    prior_vwap = rvwap.iloc[prior_slice].to_numpy()
    prior_close = df["close"].iloc[prior_slice].to_numpy()


    m = np.isfinite(prior_vwap) & np.isfinite(prior_close)
    prior_vwap = prior_vwap[m]
    prior_close = prior_close[m]

    if prior_vwap.size == 0 or prior_close.size == 0:
        return out0

    band_hi = prior_vwap * (1.0 + band_pct)
    band_lo = prior_vwap * (1.0 - band_pct)
    in_band = (prior_close >= band_lo) & (prior_close <= band_hi)
    frac = float(in_band.mean())


    k = int(min(lookback_bars, 12))
    vsub = rvwap.iloc[-k:].to_numpy()
    if k >= 2 and np.isfinite(vsub[0]) and vsub[0] != 0 and np.isfinite(vsub[-1]):
        slope = (vsub[-1] - vsub[0]) / vsub[0]
        slope_pph = float(slope * (60 / 5) / k)
    else:
        slope_pph = 0.0

    return {
        "vwap_frac_in_band": float(max(0.0, min(1.0, frac))),
        "vwap_expansion_pct": float(expansion),
        "vwap_slope_pph": float(slope_pph),
    }
