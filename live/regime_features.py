from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import indicators as ta


def _tf_to_timedelta(tf: str) -> pd.Timedelta:
    tf = str(tf).strip().lower()
    if tf.endswith("m"):
        return pd.Timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return pd.Timedelta(hours=int(tf[:-1]))
    if tf.endswith("d"):
        return pd.Timedelta(days=int(tf[:-1]))
    raise ValueError(f"Unsupported timeframe: {tf}")

def drop_incomplete_last_bar(df: pd.DataFrame, tf: str, asof_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Drop the last row if it corresponds to a bar that is not fully completed as-of asof_ts.

    Works whether the index is bar START time or bar CLOSE time, as long as:
      last_index + tf_delta > asof_ts  => last bar is still forming.
    """
    if df is None or len(df) == 0:
        return df

    asof_ts = pd.Timestamp(asof_ts)
    if asof_ts.tzinfo is None:
        asof_ts = asof_ts.tz_localize("UTC")
    else:
        asof_ts = asof_ts.tz_convert("UTC")

    idx = pd.to_datetime(df.index, utc=True)
    last_ts = idx[-1]
    delta = _tf_to_timedelta(tf)

    if last_ts + delta > asof_ts:
        return df.iloc[:-1]

    return df


def _ensure_utc_ts(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts, utc=True)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _ensure_dt_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
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
    asof_ts = _ensure_utc_ts(asof_ts)
    if df.empty:
        raise AssertionError("Series is empty.")
    pos = df.index.searchsorted(asof_ts, side="right") - 1
    if pos < 0:
        raise AssertionError(f"No rows <= asof_ts={asof_ts} (min={df.index.min()})")
    return df.iloc[int(pos)]


def triangular_moving_average(series: pd.Series, period: int) -> pd.Series:
    p = int(period)
    return series.rolling(p).mean().rolling(p).mean()


def _atr_compat(ohlc: pd.DataFrame, period: int) -> pd.Series:
    try:
        return ta.atr(ohlc, length=int(period))
    except TypeError:
        return ta.atr(ohlc, period=int(period))


@dataclass(frozen=True)
class DailyRegimeConfig:
    ma_period: int = 200
    atr_period: int = 20
    atr_mult: float = 2.0
    maxiter: int = 200
    # LIVE fixtures are bar-close labeled (right). OFFLINE fit series are left-labeled.
    assume_right_labeled_input: bool = True


@dataclass(frozen=True)
class Markov4hConfig:
    ewma_alpha: float = 0.2
    maxiter: int = 200
    assume_right_labeled_input: bool = True


def compute_daily_regime_series(df_daily: pd.DataFrame, cfg: DailyRegimeConfig) -> pd.DataFrame:
    df = _ensure_dt_index(df_daily)

    req = {"open", "high", "low", "close"}
    missing = sorted(req - set(df.columns))
    if missing:
        raise AssertionError(f"Daily OHLC missing required columns: {missing}")

    shift = pd.Timedelta("1D")
    if cfg.assume_right_labeled_input:
        df_fit = df.copy()
        df_fit.index = df_fit.index - shift
    else:
        df_fit = df

    close = df_fit["close"].astype(float)

    tma = triangular_moving_average(close, int(cfg.ma_period))

    ohlc = df_fit[["open", "high", "low", "close"]].astype(float)
    atr = _atr_compat(ohlc, int(cfg.atr_period))
    atr = atr.reindex(df_fit.index, method="ffill")

    upper = tma + float(cfg.atr_mult) * atr
    lower = tma - float(cfg.atr_mult) * atr

    trend = pd.Series(index=df_fit.index, dtype="object")
    trend[close > upper] = "BULL"
    trend[close < lower] = "BEAR"
    trend = trend.ffill().bfill()

    ret = close.pct_change()
    r = ret.dropna()

    vol_regime = pd.Series("UNKNOWN", index=df_fit.index, dtype="object")
    vol_prob_low = pd.Series(np.nan, index=df_fit.index, dtype=float)

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
        index=df_fit.index,
    )
    out.index.name = "timestamp"

    # shift outputs back to bar-close labels
    if cfg.assume_right_labeled_input:
        out = out.copy()
        out.index = out.index + shift
    return out


def compute_markov4h_series(df_4h: pd.DataFrame, cfg: Markov4hConfig) -> pd.DataFrame:
    df = _ensure_dt_index(df_4h)
    if "close" not in df.columns:
        raise AssertionError("4h OHLC missing required column: close")

    shift = pd.Timedelta("4h")
    if cfg.assume_right_labeled_input:
        df_fit = df.copy()
        df_fit.index = df_fit.index - shift
    else:
        df_fit = df

    close = df_fit["close"].astype(float)
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
        {"close": close, "logret": ret, "prob_up": prob_up, "state_up": state_up},
        index=df_fit.index,
    )
    out.index.name = "timestamp"

    if cfg.assume_right_labeled_input:
        out = out.copy()
        out.index = out.index + shift
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
    cfg = DailyRegimeConfig(
        ma_period=int(ma_period),
        atr_period=int(atr_period),
        atr_mult=float(atr_mult),
        maxiter=int(maxiter),
        assume_right_labeled_input=True,
    )
    series = compute_daily_regime_series(df_daily, cfg)
    row = _asof_row(series, decision_ts)
    return {
        "regime_code_1d": int(row["regime_code"]) if pd.notna(row["regime_code"]) else None,
        "vol_prob_low_1d": float(row["vol_prob_low"]) if pd.notna(row["vol_prob_low"]) else None,
    }


def compute_markov4h_snapshot(
    df_4h: pd.DataFrame,
    decision_ts: pd.Timestamp,
    *,
    ewma_alpha: Optional[float] = None,
    alpha: Optional[float] = None,
    maxiter: int = 200,
) -> Dict[str, Any]:
    if ewma_alpha is None and alpha is not None:
        ewma_alpha = alpha
    if ewma_alpha is None:
        ewma_alpha = 0.2

    cfg = Markov4hConfig(
        ewma_alpha=float(ewma_alpha),
        maxiter=int(maxiter),
        assume_right_labeled_input=True,
    )
    series = compute_markov4h_series(df_4h, cfg)
    row = _asof_row(series, decision_ts)
    return {
        "markov_prob_up_4h": float(row["prob_up"]),
        "markov_state_4h": int(row["state_up"]),
    }
