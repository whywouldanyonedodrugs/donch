from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

import indicators as ta


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class DailyRegimeConfig:
    ma_period: int = 200
    atr_period: int = 20
    atr_mult: float = 2.0
    maxiter: int = 200
    min_obs: int = 80  # keep consistent with your tests/tooling; offline code effectively needs "enough" data


@dataclass(frozen=True)
class Markov4hConfig:
    maxiter: int = 200
    ewma_alpha: float = 0.2
    min_obs: int = 80


# ----------------------------
# Internal helpers
# ----------------------------

def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    else:
        df = df.copy()
        df.index = df.index.tz_convert("UTC")
    return df


def triangular_moving_average(series: pd.Series, period: int) -> pd.Series:
    # TMA = SMA(SMA(price, period), period)
    return series.rolling(period).mean().rolling(period).mean()


def _asof_row(df: pd.DataFrame, asof_ts: pd.Timestamp) -> Optional[pd.Series]:
    if df.empty:
        return None
    ts = pd.to_datetime(asof_ts, utc=True, errors="coerce")
    if ts is pd.NaT:
        return None
    sub = df.loc[:ts]
    if sub.empty:
        return None
    return sub.iloc[-1]


# ----------------------------
# Caches (fit-once per input window)
# ----------------------------

_DAILY_SERIES_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}
_MARKOV4H_SERIES_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}


def _daily_cache_key(df_daily: pd.DataFrame, cfg: DailyRegimeConfig) -> Tuple[Any, ...]:
    # Key by identity + shape + bounds + params (good enough for tests/live snapshots)
    return (
        id(df_daily),
        int(df_daily.shape[0]),
        str(df_daily.index[0]),
        str(df_daily.index[-1]),
        int(cfg.ma_period),
        int(cfg.atr_period),
        float(cfg.atr_mult),
        int(cfg.maxiter),
        int(cfg.min_obs),
    )


def _markov4h_cache_key(df4h: pd.DataFrame, cfg: Markov4hConfig) -> Tuple[Any, ...]:
    return (
        id(df4h),
        int(df4h.shape[0]),
        str(df4h.index[0]),
        str(df4h.index[-1]),
        int(cfg.maxiter),
        float(cfg.ewma_alpha),
        int(cfg.min_obs),
    )


# ----------------------------
# Daily regime (trend + daily vol Markov)
# Offline semantics:
# - model fit ONCE over full window
# - volatility uses smoothed probs
# - low-vol state chosen by weighted variance under smoothed probs
# - as-of lookup is done by selecting the last daily row <= decision ts
# ----------------------------

def compute_daily_regime_series(df_daily: pd.DataFrame, cfg: DailyRegimeConfig) -> pd.DataFrame:
    df = _ensure_utc_index(df_daily)

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df_daily missing columns: {sorted(missing)}")

    close = df["close"].astype(float)

    # --- Trend intermediates ---
    tma = triangular_moving_average(close, int(cfg.ma_period))
    atr = ta.atr(df[["open", "high", "low", "close"]], length=int(cfg.atr_period))
    atr = atr.reindex(df.index, method="ffill")

    upper = tma + float(cfg.atr_mult) * atr
    lower = tma - float(cfg.atr_mult) * atr

    trend = pd.Series(index=df.index, dtype="object")
    trend[close > upper] = "BULL"
    trend[close < lower] = "BEAR"
    trend = trend.ffill().bfill()

    # --- Volatility regime (global fit; smoothed probs) ---
    ret = close.pct_change().dropna()
    vol_regime = pd.Series("UNKNOWN", index=df.index, dtype="object")
    vol_prob_low = pd.Series(np.nan, index=df.index, dtype=float)

    if ret.shape[0] >= int(cfg.min_obs):
        model = sm.tsa.MarkovRegression(ret, k_regimes=2, switching_variance=True, trend="c")
        res = model.fit(disp=False, maxiter=int(cfg.maxiter))

        probs = [res.smoothed_marginal_probabilities[i].clip(0, 1) for i in range(2)]
        r = ret.reindex(probs[0].index)

        var_est = []
        for p in probs:
            w = p.values
            denom = np.sum(w)
            if denom <= 0:
                var_est.append(np.inf)
                continue
            mu = np.sum(w * r.values) / denom
            var = np.sum(w * (r.values - mu) ** 2) / denom
            var_est.append(var)

        low_idx = int(np.argmin(var_est))
        low_prob = probs[low_idx].clip(0, 1)

        vol_prob_low.loc[low_prob.index] = low_prob.values
        vol_regime.loc[low_prob.index] = np.where(low_prob > 0.5, "LOW_VOL", "HIGH_VOL")
    else:
        # Not enough observations; keep UNKNOWN / NaN
        pass

    regime = (trend.fillna("NA") + "_" + vol_regime.fillna("UNKNOWN")).astype(str)
    code_map = {"BEAR_HIGH_VOL": 0, "BEAR_LOW_VOL": 1, "BULL_HIGH_VOL": 2, "BULL_LOW_VOL": 3}
    regime_code = regime.map(code_map).astype("Int64")

    out = pd.DataFrame(
        {
            "trend_regime_1d": trend.astype("object"),
            "vol_regime_1d": vol_regime.astype("object"),
            "vol_prob_low_1d": vol_prob_low.astype(float),
            "regime_1d": regime.astype(str),
            "regime_code_1d": regime_code,
        },
        index=df.index,
    )
    out.index.name = "timestamp"
    return out


def compute_daily_regime_snapshot(
    df_daily: pd.DataFrame,
    asof_ts: pd.Timestamp,
    ma_period: int,
    atr_period: int,
    atr_mult: float,
    maxiter: int = 200,
    min_obs: int = 80,
) -> Dict[str, object]:
    cfg = DailyRegimeConfig(
        ma_period=int(ma_period),
        atr_period=int(atr_period),
        atr_mult=float(atr_mult),
        maxiter=int(maxiter),
        min_obs=int(min_obs),
    )

    df_daily = _ensure_utc_index(df_daily)
    key = _daily_cache_key(df_daily, cfg)
    if key not in _DAILY_SERIES_CACHE:
        _DAILY_SERIES_CACHE[key] = compute_daily_regime_series(df_daily, cfg)

    series = _DAILY_SERIES_CACHE[key]
    row = _asof_row(series, asof_ts)
    if row is None:
        return {
            "trend_regime_1d": None,
            "vol_regime_1d": None,
            "vol_prob_low_1d": np.nan,
            "regime_1d": None,
            "regime_code_1d": None,
        }

    regime_code = row.get("regime_code_1d")
    vol_prob_low = row.get("vol_prob_low_1d")

    return {
        "trend_regime_1d": row.get("trend_regime_1d"),
        "vol_regime_1d": row.get("vol_regime_1d"),
        "vol_prob_low_1d": float(vol_prob_low) if pd.notna(vol_prob_low) else np.nan,
        "regime_1d": row.get("regime_1d"),
        "regime_code_1d": int(regime_code) if pd.notna(regime_code) else None,
    }


# ----------------------------
# 4h Markov regime
# Offline semantics:
# - log returns
# - model fit ONCE over full window
# - filtered probs (past-only)
# - UP state chosen by higher prob-weighted mean return (filtered probs)
# - EWMA smoothing on prob_up (adjust=False)
# - as-of lookup via last row <= decision ts
# ----------------------------

def compute_markov4h_series(df4h: pd.DataFrame, cfg: Markov4hConfig) -> pd.DataFrame:
    df = _ensure_utc_index(df4h)

    if "close" not in df.columns:
        raise ValueError("df4h missing required column: close")

    close = df["close"].astype(float)
    ret = np.log(close).diff().dropna()
    if ret.shape[0] < int(cfg.min_obs):
        out = pd.DataFrame(columns=["markov_prob_up_4h", "markov_state_4h"], index=pd.DatetimeIndex([], tz="UTC"))
        out.index.name = "timestamp"
        return out

    mod = sm.tsa.MarkovRegression(ret, k_regimes=2, switching_variance=True, trend="c")
    res = mod.fit(disp=False, maxiter=int(cfg.maxiter))

    fp = [res.filtered_marginal_probabilities[i].clip(0, 1) for i in range(2)]

    means = []
    for p in fp:
        w = p.values
        r = ret.reindex(p.index).values
        denom = max(np.sum(w), 1e-12)
        mu = np.sum(w * r) / denom
        means.append(mu)

    up_idx = int(np.argmax(means))

    prob_up_raw = fp[up_idx].clip(0, 1)
    prob_up = prob_up_raw.ewm(alpha=float(cfg.ewma_alpha), adjust=False).mean().clip(0, 1)
    state_up = (prob_up > 0.5).astype(int)

    out = pd.DataFrame(
        {"markov_prob_up_4h": prob_up.astype(float), "markov_state_4h": state_up.astype(int)},
        index=prob_up.index,
    )
    out.index.name = "timestamp"
    return out


def compute_markov4h_snapshot(
    df4h: pd.DataFrame,
    asof_ts: pd.Timestamp,
    ewma_alpha: float = 0.2,
    maxiter: int = 200,
    min_obs: int = 80,
) -> Dict[str, object]:
    cfg = Markov4hConfig(maxiter=int(maxiter), ewma_alpha=float(ewma_alpha), min_obs=int(min_obs))

    df4h = _ensure_utc_index(df4h)
    key = _markov4h_cache_key(df4h, cfg)
    if key not in _MARKOV4H_SERIES_CACHE:
        _MARKOV4H_SERIES_CACHE[key] = compute_markov4h_series(df4h, cfg)

    series = _MARKOV4H_SERIES_CACHE[key]
    row = _asof_row(series, asof_ts)
    if row is None:
        return {"markov_prob_up_4h": np.nan, "markov_state_4h": None}

    p = row.get("markov_prob_up_4h")
    s = row.get("markov_state_4h")
    return {
        "markov_prob_up_4h": float(p) if pd.notna(p) else np.nan,
        "markov_state_4h": int(s) if pd.notna(s) else None,
    }
