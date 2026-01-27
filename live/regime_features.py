"""
regime_features.py

Deterministic, "as-of" regime feature computations for live trading.

Constraints:
- All outputs are computed strictly as-of an explicit `asof_ts` (UTC, tz-aware).
- No wall-clock time is used.
- Bars are assumed timestamped at bar OPEN (CCXT convention). The currently-forming bar is excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import indicators as ta


REGIME_CODE_MAP: Dict[str, int] = {
    "BEAR_HIGH_VOL": 0,
    "BEAR_LOW_VOL": 1,
    "BULL_HIGH_VOL": 2,
    "BULL_LOW_VOL": 3,
}


def _to_utc_ts(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError("asof_ts is not a valid timestamp")
    return ts


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by timestamps")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def triangular_moving_average(series: pd.Series, period: int) -> pd.Series:
    series = series.astype(float)
    return (
        series.rolling(window=period, min_periods=period).mean()
        .rolling(window=period, min_periods=period).mean()
    )


def drop_incomplete_last_bar(df: pd.DataFrame, tf: str, asof_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Exclude bars whose OPEN timestamp is >= floor(tf) of asof_ts.
    This removes the currently-forming bar under CCXT open-timestamp convention.
    """
    if df is None or df.empty:
        return df

    asof_ts = _to_utc_ts(asof_ts)
    df = _ensure_utc_index(df)

    if tf.lower() in ("1d", "1day", "d", "day"):
        floor = asof_ts.floor("D")
    elif tf.lower() in ("4h", "4hour", "h4"):
        floor = asof_ts.floor("4h")
    else:
        raise ValueError(f"Unsupported tf for drop_incomplete_last_bar: {tf}")

    # strict exclusion: anything opening at/after the current bucket is incomplete
    return df.loc[df.index < floor].copy()


def compute_daily_regime_snapshot(
    df_daily: pd.DataFrame,
    asof_ts: pd.Timestamp,
    ma_period: int,
    atr_period: int,
    atr_mult: float,
) -> Dict[str, object]:
    """
    Canonical daily combined regime:
      - Trend: TMA(close, ma_period) with Keltner bands using ATR(atr_period) * atr_mult.
      - Vol: 2-state MarkovRegression on daily pct returns with smoothed probs;
             identify low-vol state by weighted variance; vol_prob_low is that state's prob.

    Returns snapshot for the latest fully-closed daily bar as-of `asof_ts`:
      trend_regime_1d, vol_regime_1d, vol_prob_low_1d, daily_regime_str_1d, regime_code_1d
    """
    asof_ts = _to_utc_ts(asof_ts)
    df_daily = _ensure_utc_index(df_daily)
    df_use = drop_incomplete_last_bar(df_daily, "1d", asof_ts)

    if df_use is None or df_use.empty:
        raise ValueError("No daily bars available as-of asof_ts")

    # Require required columns
    for c in ("open", "high", "low", "close"):
        if c not in df_use.columns:
            raise ValueError(f"df_daily missing required column: {c}")

    close = df_use["close"].astype(float)

    # Vol regime (Markov on pct returns)
    ret = close.pct_change().dropna()
    vol_regime = pd.Series(index=df_use.index, data="UNKNOWN", dtype="object")
    vol_prob_low = pd.Series(index=df_use.index, data=np.nan, dtype=float)

    if len(ret) < 50:
        # Too little for stable Markov; fail closed at snapshot extraction time
        pass
    else:
        model = sm.tsa.MarkovRegression(ret, k_regimes=2, switching_variance=True, trend="c")
        results = model.fit(disp=False, maxiter=200)

        probs = [results.smoothed_marginal_probabilities[i] for i in range(2)]

        r = ret.reindex(probs[0].index)
        var_est = []
        for p in probs:
            w = p.values
            denom = float(np.sum(w))
            if denom <= 1e-12:
                var_est.append(np.inf)
                continue
            mu = float(np.sum(w * r.values) / denom)
            var = float(np.sum(w * (r.values - mu) ** 2) / denom)
            var_est.append(var)

        low_idx = int(np.argmin(var_est))
        low_prob = probs[low_idx].clip(0.0, 1.0)

        vol_reg = np.where(low_prob > 0.5, "LOW_VOL", "HIGH_VOL")
        vol_regime.loc[ret.index] = vol_reg
        vol_prob_low.loc[ret.index] = low_prob.values

    # Trend regime (TMA + ATR bands)
    tma = triangular_moving_average(close, int(ma_period))
    atr_d = ta.atr(df_use, int(atr_period)).reindex(df_use.index, method="ffill")

    upper = tma + float(atr_mult) * atr_d
    lower = tma - float(atr_mult) * atr_d

    trend = pd.Series(index=df_use.index, dtype="object")
    trend.loc[close > upper] = "BULL"
    trend.loc[close < lower] = "BEAR"
    trend = trend.ffill().bfill()

    # Snapshot at last fully-closed day
    last_idx = df_use.index[-1]
    tr_last = trend.loc[last_idx]
    vr_last = vol_regime.loc[last_idx]
    vpl_last = vol_prob_low.loc[last_idx]

    combined = f"{str(tr_last)}_{str(vr_last)}"
    code = REGIME_CODE_MAP.get(combined, None)

    if code is None or not np.isfinite(float(vpl_last)):
        raise ValueError(f"Daily regime snapshot not fully defined (combined={combined}, vol_prob={vpl_last})")

    return {
        "trend_regime_1d": str(tr_last),
        "vol_regime_1d": str(vr_last),
        "vol_prob_low_1d": float(vpl_last),
        "daily_regime_str_1d": str(combined),
        "regime_code_1d": int(code),
    }


def compute_markov4h_snapshot(
    df4h: pd.DataFrame,
    asof_ts: pd.Timestamp,
    alpha: float,
) -> Dict[str, object]:
    """
    Canonical 4h Markov:
      - 2-state MarkovRegression on log returns.
      - FILTERED (past-only) marginal probabilities.
      - Identify UP state by higher weighted mean return.
      - EWM smoothing with alpha.
    Snapshot is last available probability as-of the last fully-closed 4h bar.
    """
    asof_ts = _to_utc_ts(asof_ts)
    df4h = _ensure_utc_index(df4h)
    df_use = drop_incomplete_last_bar(df4h, "4h", asof_ts)

    if df_use is None or df_use.empty or len(df_use) < 60:
        raise ValueError("Insufficient 4h bars for Markov snapshot")

    if "close" not in df_use.columns:
        raise ValueError("df4h missing required column: close")

    close = df_use["close"].astype(float)
    ret = np.log(close).diff().dropna()
    if len(ret) < 50:
        raise ValueError("Insufficient 4h returns for Markov snapshot")

    mod = sm.tsa.MarkovRegression(ret, k_regimes=2, switching_variance=True, trend="c")
    res = mod.fit(disp=False, maxiter=200)

    filt_probs = [res.filtered_marginal_probabilities[i] for i in range(2)]

    means = []
    for p in filt_probs:
        w = p.values
        r = ret.reindex(p.index).values
        denom = max(float(np.sum(w)), 1e-12)
        mu = float(np.sum(w * r) / denom)
        means.append(mu)

    up_idx = int(np.argmax(means))
    prob_up = filt_probs[up_idx].clip(0.0, 1.0)
    prob_up = prob_up.ewm(alpha=float(alpha), adjust=False).mean().clip(0.0, 1.0)

    last_prob = float(prob_up.iloc[-1])
    last_state = int(last_prob > 0.5)

    return {
        "markov_prob_up_4h": float(last_prob),
        "markov_state_4h": int(last_state),
    }
