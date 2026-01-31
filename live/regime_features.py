from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from . import indicators as ta


# Offline label mapping: keep stable codes
REGIME_CODE_MAP = {
    "BULL_LOW_VOL": 0,
    "BULL_HIGH_VOL": 1,
    "BEAR_LOW_VOL": 2,
    "BEAR_HIGH_VOL": 3,
}


@dataclass(frozen=True)
class RegimeConfig:
    # If True, assume the input OHLCV is already right-labeled (bar close timestamps).
    # If False, a shift may be applied for some datasets.
    assume_right_labeled_input: bool = True


def _ensure_utc_ts(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError("Invalid timestamp")
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def drop_incomplete_last_bar(df: pd.DataFrame, tf: str, asof_ts: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Drop the last (possibly still forming) bar from df.

    Backward/forward compatible:
      - Older call-sites: drop_incomplete_last_bar(df, tf)
      - Newer call-sites: drop_incomplete_last_bar(df, tf, asof_ts)
    """
    if df is None or df.empty:
        return df

    out = df.sort_index()
    if asof_ts is None:
        # Conservative: always drop the last row (treat it as potentially incomplete)
        return out.iloc[:-1] if len(out) > 1 else out.iloc[0:0]

    asof_ts = _ensure_utc_ts(asof_ts)

    # Determine bar length by tf
    if tf.endswith("m"):
        minutes = int(tf[:-1])
        bar = pd.Timedelta(minutes=minutes)
    elif tf.endswith("h"):
        hours = int(tf[:-1])
        bar = pd.Timedelta(hours=hours)
    elif tf.endswith("d"):
        days = int(tf[:-1])
        bar = pd.Timedelta(days=days)
    else:
        raise ValueError(f"Unsupported tf={tf!r}")

    # If the last bar close is > asof_ts, it is definitely incomplete relative to as-of
    # Otherwise, if asof_ts is inside the interval (last_close - bar, last_close], treat as incomplete.
    last_close = out.index[-1]
    if last_close > asof_ts:
        return out.loc[:asof_ts].iloc[:-1] if len(out.loc[:asof_ts]) > 1 else out.loc[:asof_ts].iloc[0:0]

    if (last_close - bar) < asof_ts <= last_close:
        return out.iloc[:-1] if len(out) > 1 else out.iloc[0:0]

    # Otherwise keep all rows
    return out


def compute_daily_regime_series(
    df_daily: pd.DataFrame,
    *,
    ma_period: int = 100,
    atr_period: int = 20,
    atr_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Deterministic daily regime:
      - Trend: BULL/BEAR based on close relative to MA +/- ATR*mult envelope
      - Vol: LOW_VOL/HIGH_VOL from a 2-state MarkovRegression on returns variance, with p(low-vol)
    Returns a dataframe with columns including:
      trend_regime, vol_regime, vol_prob_low, regime, regime_code
    """
    if df_daily is None or df_daily.empty:
        raise ValueError("df_daily is empty")

    df = df_daily.sort_index().copy()
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)

    tma = close.rolling(int(ma_period), min_periods=max(5, int(ma_period) // 5)).mean()

    ohlc_cols = ["open", "high", "low", "close"]
    if all(c in df.columns for c in ohlc_cols):
        atr = ta.atr(df[ohlc_cols], int(atr_period))
    else:
        atr = ta.atr(df[["high", "low", "close"]], int(atr_period))

    upper = tma + float(atr_mult) * atr
    lower = tma - float(atr_mult) * atr

    trend = pd.Series(index=df.index, dtype="object")
    trend.loc[close > upper] = "BULL"
    trend.loc[close < lower] = "BEAR"
    trend = trend.ffill().fillna("BEAR")

    # Vol regime via MarkovRegression on daily returns
    r = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 50:
        # Not enough data to fit: degrade deterministically
        vol_prob_low = pd.Series(0.5, index=df.index, dtype="float64")
        vol_regime = pd.Series("UNKNOWN", index=df.index, dtype="object")
    else:
        model = MarkovRegression(r, k_regimes=2, trend="c", switching_variance=True)
        res = model.fit(disp=False, maxiter=200)

        # smoothed marginal probabilities for regime 0/1
        p = res.smoothed_marginal_probabilities
        # Pick low-vol regime as the one with smaller variance
        sig2 = res.params.filter(like="sigma2").values
        if len(sig2) >= 2:
            low_idx = int(np.argmin(sig2))
        else:
            low_idx = 0

        p_low = p[low_idx].reindex(df.index).ffill()
        vol_prob_low = p_low.astype(float)

        # Hard label with 0.5 threshold
        vol_regime = pd.Series(index=df.index, dtype="object")
        vol_regime.loc[vol_prob_low >= 0.5] = "LOW_VOL"
        vol_regime.loc[vol_prob_low < 0.5] = "HIGH_VOL"
        vol_regime = vol_regime.fillna("UNKNOWN")

    regime = trend.astype(str) + "_" + vol_regime.astype(str)
    regime_code = regime.map(REGIME_CODE_MAP).astype("float64")

    out = pd.DataFrame(
        {
            "close": close,
            "tma": tma,
            "atr": atr,
            "upper": upper,
            "lower": lower,
            "trend_regime": trend,
            "vol_regime": vol_regime,
            "vol_prob_low": vol_prob_low,
            "regime": regime,
            "regime_code": regime_code,
        },
        index=df.index,
    )
    return out


def _asof_row(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
    ts = _ensure_utc_ts(ts)
    df = df.sort_index()
    df2 = df.loc[:ts]
    if df2.empty:
        raise ValueError(f"No rows at or before ts={ts}")
    return df2.iloc[-1]


def compute_daily_regime_snapshot(
    df_daily: pd.DataFrame,
    decision_ts: Optional[pd.Timestamp] = None,
    *,
    asof_ts: Optional[pd.Timestamp] = None,
    ma_period: int = 100,
    atr_period: int = 20,
    atr_mult: float = 2.0,
    cfg: Optional[RegimeConfig] = None,
) -> Dict[str, Any]:
    """
    Snapshot as-of a decision timestamp.

    Compatibility:
      - Newer call-sites pass asof_ts=...
      - Older call-sites pass decision_ts as positional arg
    """
    if decision_ts is None:
        decision_ts = asof_ts
    if decision_ts is None:
        raise TypeError("compute_daily_regime_snapshot requires decision_ts or asof_ts")

    cfg = cfg or RegimeConfig()
    decision_ts = _ensure_utc_ts(decision_ts)

    # Optional shift hook (kept for parity toggles)
    if cfg.assume_right_labeled_input:
        df_fit = df_daily.copy()
    else:
        df_fit = df_daily.copy()

    ser = compute_daily_regime_series(
        df_fit,
        ma_period=ma_period,
        atr_period=atr_period,
        atr_mult=atr_mult,
    )

    row = _asof_row(ser, decision_ts)

    daily_regime_str = str(row["regime"])
    regime_code = REGIME_CODE_MAP.get(daily_regime_str)
    vol_prob_low = float(row["vol_prob_low"]) if pd.notna(row["vol_prob_low"]) else np.nan

    return {
        # Keys expected by live/live_trader.py
        "trend_regime_1d": str(row["trend_regime"]),
        "vol_regime_1d": str(row["vol_regime"]),
        "vol_prob_low_1d": vol_prob_low,
        "daily_regime_str_1d": daily_regime_str,
        "regime_code_1d": int(regime_code) if regime_code is not None else None,
        # Back-compat / extra
        "regime_code": int(regime_code) if regime_code is not None else None,
        "vol_prob_low": vol_prob_low,
    }


def compute_markov4h_series(
    df_4h: pd.DataFrame,
    *,
    ewma_alpha: float = 0.2,
    maxiter: int = 200,
) -> pd.DataFrame:
    if df_4h is None or df_4h.empty:
        raise ValueError("df_4h is empty")

    df = df_4h.sort_index().copy()
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)

    r = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 80:
        p_up = pd.Series(0.5, index=df.index, dtype="float64")
        state = pd.Series(0, index=df.index, dtype="int64")
    else:
        model = MarkovRegression(r, k_regimes=2, trend="c", switching_variance=True)
        res = model.fit(disp=False, maxiter=int(maxiter))

        p = res.smoothed_marginal_probabilities
        # Define “up” state as the regime with higher mean return (approx via params)
        # If not available, default to 0
        mu = res.params.filter(like="const").values
        if len(mu) >= 2:
            up_idx = int(np.argmax(mu))
        else:
            up_idx = 0

        p_up_raw = p[up_idx].reindex(df.index).ffill().astype(float)
        p_up = p_up_raw.ewm(alpha=float(ewma_alpha), adjust=False).mean()

        state = (p_up >= 0.5).astype("int64")

    return pd.DataFrame({"close": close, "markov_prob_up": p_up, "markov_state": state}, index=df.index)


def compute_markov4h_snapshot(
    df_4h: pd.DataFrame,
    decision_ts: Optional[pd.Timestamp] = None,
    *,
    asof_ts: Optional[pd.Timestamp] = None,
    ewma_alpha: Optional[float] = None,
    alpha: Optional[float] = None,
    maxiter: int = 200,
) -> Dict[str, Any]:
    """
    Snapshot Markov 4h probability as-of decision timestamp.

    Compatibility:
      - Newer call-sites pass asof_ts=...
      - Some call-sites pass alpha=... (alias of ewma_alpha)
    """
    if decision_ts is None:
        decision_ts = asof_ts
    if decision_ts is None:
        raise TypeError("compute_markov4h_snapshot requires decision_ts or asof_ts")

    decision_ts = _ensure_utc_ts(decision_ts)

    if ewma_alpha is None and alpha is not None:
        ewma_alpha = float(alpha)
    if ewma_alpha is None:
        ewma_alpha = 0.2

    ser = compute_markov4h_series(df_4h, ewma_alpha=float(ewma_alpha), maxiter=int(maxiter))
    row = _asof_row(ser, decision_ts)

    return {
        "markov_prob_up_4h": float(row["markov_prob_up"]) if pd.notna(row["markov_prob_up"]) else np.nan,
        "markov_state_4h": int(row["markov_state"]) if pd.notna(row["markov_state"]) else 0,
    }
