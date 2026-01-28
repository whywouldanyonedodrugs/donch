from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

# IMPORTANT: indicators is inside the live package on LIVE
from . import indicators as ta


# -----------------------------
# Helpers
# -----------------------------

def _ensure_utc_ts(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts, utc=True)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _ensure_dt_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Ensure df is indexed by a tz-aware UTC DatetimeIndex, sorted ascending.
    Accepts either:
      - df already indexed by DatetimeIndex, or
      - df has a timestamp column.
    """
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        idx = pd.to_datetime(out.index, utc=True, errors="coerce")
        out.index = idx
    elif ts_col in out.columns:
        idx = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
        out = out.drop(columns=[ts_col])
        out.index = idx
    else:
        raise AssertionError(f"Expected DatetimeIndex or '{ts_col}' column.")

    out = out[~out.index.isna()].sort_index()
    return out


def _asof_row(df: pd.DataFrame, asof_ts: pd.Timestamp) -> pd.Series:
    """
    Deterministic as-of: last row with index <= asof_ts.
    """
    asof_ts = _ensure_utc_ts(asof_ts)
    if df.empty:
        raise AssertionError("Series is empty.")
    pos = df.index.searchsorted(asof_ts, side="right") - 1
    if pos < 0:
        raise AssertionError(f"No rows <= asof_ts={asof_ts} (min={df.index.min()})")
    return df.iloc[int(pos)]


def triangular_moving_average(series: pd.Series, period: int) -> pd.Series:
    """
    OFFLINE parity:
      TMA = SMA(SMA(close, period), period)
    """
    p = int(period)
    return series.rolling(p).mean().rolling(p).mean()


def _atr_compat(ohlc: pd.DataFrame, period: int) -> pd.Series:
    """
    LIVE has had atr signature drift ('length' vs 'period').
    Use a compat wrapper to avoid failures.
    """
    try:
        return ta.atr(ohlc, length=int(period))  # pandas_ta-like
    except TypeError:
        return ta.atr(ohlc, period=int(period))  # local implementation-like


# -----------------------------
# Daily regime (trend + vol)
# -----------------------------

@dataclass(frozen=True)
class DailyRegimeConfig:
    ma_period: int = 200
    atr_period: int = 20
    atr_mult: float = 2.0
    maxiter: int = 200


def compute_daily_regime_series(df_daily: pd.DataFrame, cfg: DailyRegimeConfig) -> pd.DataFrame:
    """
    Match OFFLINE regime_detector.compute_daily_combined_regime() semantics,
    assuming df_daily already represents 1D OHLC with the OFFLINE bin labeling.

    - Trend:
        TMA(close, ma_period) +/- atr_mult * ATR(atr_period)
        trend = BULL if close > upper, BEAR if close < lower, then ffill/bfill
    - Volatility:
        MarkovRegression on daily pct returns, k=2, switching_variance=True, trend='c'
        Uses smoothed_marginal_probabilities (two-sided).
        Low-vol state chosen by prob-weighted variance under smoothed probs.
        vol_prob_low = smoothed prob of low-var state
        vol_regime = LOW_VOL if vol_prob_low > 0.5 else HIGH_VOL (on return rows only)
    - regime_code mapping:
        BEAR_HIGH_VOL=0, BEAR_LOW_VOL=1, BULL_HIGH_VOL=2, BULL_LOW_VOL=3
    """
    df = _ensure_dt_index(df_daily)

    req = {"open", "high", "low", "close"}
    missing = sorted(req - set(df.columns))
    if missing:
        raise AssertionError(f"Daily OHLC missing required columns: {missing}")

    # Ensure numeric close
    close = df["close"].astype(float)

    # Trend intermediates
    tma = triangular_moving_average(close, int(cfg.ma_period))

    ohlc = df[["open", "high", "low", "close"]].astype(float)
    atr = _atr_compat(ohlc, int(cfg.atr_period))
    # Align + ffill like OFFLINE does via reindex(..., method="ffill")
    atr = atr.reindex(df.index, method="ffill")

    upper = tma + float(cfg.atr_mult) * atr
    lower = tma - float(cfg.atr_mult) * atr

    trend = pd.Series(index=df.index, dtype="object")
    trend[close > upper] = "BULL"
    trend[close < lower] = "BEAR"
    # OFFLINE: ffill().bfill()
    trend = trend.ffill().bfill()

    # Volatility regime intermediates
    ret = close.pct_change()
    r = ret.dropna()

    vol_regime = pd.Series("UNKNOWN", index=df.index, dtype="object")
    vol_prob_low = pd.Series(np.nan, index=df.index, dtype=float)

    if not r.empty:
        model = sm.tsa.MarkovRegression(r, k_regimes=2, switching_variance=True, trend="c")
        res = model.fit(disp=False, maxiter=int(cfg.maxiter))

        probs = [res.smoothed_marginal_probabilities[i].clip(0, 1) for i in range(2)]
        rr = r.reindex(probs[0].index)

        var_est = []
        for p in probs:
            w = p.values
            denom = float(np.sum(w))
            if denom <= 1e-12:
                var_est.append(np.inf)
                continue
            mu = float(np.sum(w * rr.values) / denom)
            var = float(np.sum(w * (rr.values - mu) ** 2) / denom)
            var_est.append(var)

        low_idx = int(np.argmin(var_est))
        low_prob = probs[low_idx].clip(0, 1)

        vol_prob_low.loc[low_prob.index] = low_prob.values
        vol_regime.loc[low_prob.index] = np.where(low_prob.values > 0.5, "LOW_VOL", "HIGH_VOL")

    regime = (trend.fillna("NA") + "_" + vol_regime.fillna("UNKNOWN")).astype(str)
    code_map = {"BEAR_HIGH_VOL": 0, "BEAR_LOW_VOL": 1, "BULL_HIGH_VOL": 2, "BULL_LOW_VOL": 3}
    regime_code = regime.map(code_map).astype("Int64")

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
    out.index.name = "timestamp"
    return out


def compute_daily_regime_snapshot(
    df_daily: pd.DataFrame,
    decision_ts: pd.Timestamp,
    *,
    ma_period: int = 200,
    atr_period: int = 20,
    atr_mult: float = 2.0,
    maxiter: int = 200,
) -> Dict[str, Any]:
    """
    Convenience wrapper (NOT used by the optimized test; kept for callers).
    Computes full series then as-of lookup.

    OFFLINE mapping is effectively day-based (ts.floor('D')), but since daily index
    is at 00:00 it is equivalent to as-of decision_ts for any intraday time.
    """
    cfg = DailyRegimeConfig(ma_period=int(ma_period), atr_period=int(atr_period), atr_mult=float(atr_mult), maxiter=int(maxiter))
    series = compute_daily_regime_series(df_daily, cfg)
    row = _asof_row(series, decision_ts)
    return {
        "regime_code_1d": int(row["regime_code"]) if pd.notna(row["regime_code"]) else None,
        "vol_prob_low_1d": float(row["vol_prob_low"]) if pd.notna(row["vol_prob_low"]) else None,
    }


# -----------------------------
# 4h Markov regime
# -----------------------------

@dataclass(frozen=True)
class Markov4hConfig:
    ewma_alpha: float = 0.2
    maxiter: int = 200


def compute_markov4h_series(df_4h: pd.DataFrame, cfg: Markov4hConfig) -> pd.DataFrame:
    """
    Match OFFLINE regime_detector.compute_markov_regime_4h() semantics,
    assuming df_4h already represents 4h OHLC with the OFFLINE bin labeling.

    - returns: log(close).diff()
    - MarkovRegression: k=2, switching_variance=True, trend='c', maxiter=200
    - probabilities: filtered_marginal_probabilities (past-only)
    - UP state: regime with higher prob-weighted mean return under filtered probs
    - prob_up: filtered prob of UP state, then EWMA(alpha, adjust=False)
    - state_up: (prob_up > 0.5).astype(int)
    """
    df = _ensure_dt_index(df_4h)

    if "close" not in df.columns:
        raise AssertionError("4h OHLC missing required column: close")

    close = df["close"].astype(float)
    ret = np.log(close).diff()
    r = ret.dropna()
    if r.empty:
        raise AssertionError("4h log return series is empty")

    model = sm.tsa.MarkovRegression(r, k_regimes=2, switching_variance=True, trend="c")
    res = model.fit(disp=False, maxiter=int(cfg.maxiter))

    fp = [res.filtered_marginal_probabilities[i].clip(0, 1) for i in range(2)]
    rr = r.reindex(fp[0].index)

    means = []
    for p in fp:
        w = p.values
        denom = float(np.sum(w))
        if denom <= 1e-12:
            means.append(-np.inf)
            continue
        mu = float(np.sum(w * rr.values) / denom)
        means.append(mu)

    up_idx = int(np.argmax(means))

    prob_up_raw = fp[up_idx].clip(0, 1)
    prob_up = prob_up_raw.ewm(alpha=float(cfg.ewma_alpha), adjust=False).mean().clip(0, 1)
    state_up = (prob_up > 0.5).astype(int)

    out = pd.DataFrame(
        {
            "close": close,
            "logret": ret,
            "prob_up": prob_up,
            "state_up": state_up,
        },
        index=df.index,
    )
    out.index.name = "timestamp"
    return out


def compute_markov4h_snapshot(
    df_4h: pd.DataFrame,
    decision_ts: pd.Timestamp,
    *,
    ewma_alpha: Optional[float] = None,
    alpha: Optional[float] = None,
    maxiter: int = 200,
) -> Dict[str, Any]:
    """
    Convenience wrapper with backward-compat keyword handling:
      - accept ewma_alpha=... (preferred)
      - accept alpha=... (legacy)

    Computes full series then as-of lookup.
    """
    if ewma_alpha is None and alpha is not None:
        ewma_alpha = alpha
    if ewma_alpha is None:
        ewma_alpha = 0.2

    cfg = Markov4hConfig(ewma_alpha=float(ewma_alpha), maxiter=int(maxiter))
    series = compute_markov4h_series(df_4h, cfg)
    row = _asof_row(series, decision_ts)
    return {
        "markov_prob_up_4h": float(row["prob_up"]),
        "markov_state_4h": int(row["state_up"]),
    }
