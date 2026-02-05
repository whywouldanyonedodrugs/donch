from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from . import indicators as ta


# Offline mapping (per offline team):
# {"BEAR_HIGH_VOL":0, "BEAR_LOW_VOL":1, "BULL_HIGH_VOL":2, "BULL_LOW_VOL":3}
REGIME_CODE_MAP = {
    "BEAR_HIGH_VOL": 0,
    "BEAR_LOW_VOL": 1,
    "BULL_HIGH_VOL": 2,
    "BULL_LOW_VOL": 3,
}


@dataclass(frozen=True)
class RegimeConfig:
    # Match offline defaults (per offline team note)
    ma_period: int = 200
    atr_period: int = 20
    atr_mult: float = 2.0
    markov4h_prob_ewma_alpha: float = 0.2


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    else:
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _tf_to_timedelta(tf: str) -> pd.Timedelta:
    tf = str(tf).strip().lower()
    if tf.endswith("m"):
        return pd.Timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return pd.Timedelta(hours=int(tf[:-1]))
    if tf.endswith("d"):
        return pd.Timedelta(days=int(tf[:-1]))
    if tf in ("1d", "1D"):
        return pd.Timedelta(days=1)
    raise ValueError(f"Unsupported tf={tf}")


def drop_incomplete_last_bar(df: pd.DataFrame, tf: str, asof_ts: Any) -> pd.DataFrame:
    """
    Drop the last bar if it's incomplete as-of asof_ts.

    IMPORTANT: In this repo, CCXT OHLCV timestamps are BAR OPEN TIME (left-labeled).
    Therefore, a bar at index t covers [t, t+bar). It is complete iff (t+bar) <= asof_ts.
    """
    if df is None or df.empty:
        return df
    df = _ensure_utc_index(df)

    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize("UTC")
    else:
        asof = asof.tz_convert("UTC")

    bar = _tf_to_timedelta(tf)
    last_open = df.index[-1]

    # If the last bar's end is after asof, it's still forming -> drop it.
    if (last_open + bar) > asof:
        return df.iloc[:-1]
    return df


def _triangular_moving_average(close: pd.Series, period: int) -> pd.Series:
    # Offline: TMA = SMA(SMA(close, ma_period), ma_period)
    sma1 = close.rolling(period, min_periods=period).mean()
    sma2 = sma1.rolling(period, min_periods=period).mean()
    return sma2


def compute_daily_regime_series(
    daily_ohlc: pd.DataFrame,
    *,
    ma_period: int = 200,
    atr_period: int = 20,
    atr_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Offline-aligned daily combined regime:
      - Trend: TMA + ATR channel; BULL if close>upper, BEAR if close<lower, ffill+bfill
      - Vol: 2-state Markov switching variance on pct returns; use SMOOTHED probs
      - vol_prob_low = P(low-variance state); LOW_VOL if vol_prob_low>0.5 else HIGH_VOL
      - regime_code mapping: BEAR_HIGH_VOL=0, BEAR_LOW_VOL=1, BULL_HIGH_VOL=2, BULL_LOW_VOL=3
    """
    if daily_ohlc is None or daily_ohlc.empty:
        raise ValueError("compute_daily_regime_series: empty daily_ohlc")

    df = _ensure_utc_index(daily_ohlc)
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"compute_daily_regime_series: missing column {c}")

    close = df["close"].astype(float)

    tma = _triangular_moving_average(close, int(ma_period))
    atr = ta.atr(df[["open", "high", "low", "close"]], length=int(atr_period)).astype(float)

    upper = tma + float(atr_mult) * atr
    lower = tma - float(atr_mult) * atr

    trend = pd.Series(index=df.index, dtype=object)
    trend[close > upper] = "BULL"
    trend[close < lower] = "BEAR"
    trend = trend.ffill().bfill()

    ret = close.pct_change().dropna()
    if len(ret) < 50:
        raise ValueError(f"compute_daily_regime_series: insufficient returns for Markov fit (n={len(ret)})")

    mod = MarkovRegression(ret, k_regimes=2, switching_variance=True, trend="c")
    res = mod.fit(disp=False)

    # Identify low-vol state by smaller sigma2 estimate (robust extraction)
    sigma2_items = []
    for k, v in res.params.items():
        ks = str(k)
        if "sigma2" in ks:
            # common keys look like 'sigma2[0]' / 'sigma2[1]'
            sigma2_items.append((ks, float(v)))
    if len(sigma2_items) < 2:
        # Fallback: assume state 0 is low-vol if we can't extract reliably
        low_state = 0
    else:
        # Prefer bracketed numeric ordering if present
        def _state_from_key(ks: str) -> Optional[int]:
            lb = ks.find("[")
            rb = ks.find("]")
            if lb >= 0 and rb > lb:
                try:
                    return int(ks[lb + 1 : rb])
                except Exception:
                    return None
            return None

        parsed = []
        for ks, v in sigma2_items:
            st = _state_from_key(ks)
            if st is not None:
                parsed.append((st, v))
        if len(parsed) >= 2:
            parsed = sorted(parsed, key=lambda x: x[0])
            low_state = int(min(parsed, key=lambda x: x[1])[0])
        else:
            # Just take the smallest sigma2 among the first two items
            low_state = int(np.argmin([sigma2_items[0][1], sigma2_items[1][1]]))

    p0 = res.smoothed_marginal_probabilities[0]
    p1 = res.smoothed_marginal_probabilities[1]
    vol_prob_low = (p0 if low_state == 0 else p1).reindex(df.index).ffill()

    vol_regime = np.where(vol_prob_low > 0.5, "LOW_VOL", "HIGH_VOL")
    regime_name = trend.astype(str) + "_" + pd.Series(vol_regime, index=df.index).astype(str)
    regime_code = regime_name.map(REGIME_CODE_MAP).astype(float)

    out = pd.DataFrame(
        {
            "regime_name": regime_name,
            "regime_code": regime_code,
            "vol_prob_low": vol_prob_low.astype(float),
            "tma": tma.astype(float),
            "atr": atr.astype(float),
            "upper": upper.astype(float),
            "lower": lower.astype(float),
        },
        index=df.index,
    )
    out = out.dropna(subset=["regime_code", "vol_prob_low"])
    return out


def compute_daily_regime_snapshot(daily_ohlc: pd.DataFrame, *, asof_ts: Any, cfg: Optional[RegimeConfig] = None) -> Dict[str, Any]:
    cfg = cfg or RegimeConfig()
    df = _ensure_utc_index(daily_ohlc)
    df = drop_incomplete_last_bar(df, "1d", asof_ts)

    ser = compute_daily_regime_series(
        df,
        ma_period=cfg.ma_period,
        atr_period=cfg.atr_period,
        atr_mult=cfg.atr_mult,
    )
    if ser.empty:
        raise ValueError("compute_daily_regime_snapshot: empty regime series after drop")

    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize("UTC")
    else:
        asof = asof.tz_convert("UTC")

    ser2 = ser.loc[:asof]
    if ser2.empty:
        raise ValueError(f"compute_daily_regime_snapshot: no row asof={asof} (min={ser.index.min()} max={ser.index.max()})")

    row = ser2.iloc[-1]
    return {
        "regime_code_1d": int(row["regime_code"]),
        "vol_prob_low_1d": float(row["vol_prob_low"]),
    }


def compute_markov4h_series(
    ohlc_4h: pd.DataFrame,
    *,
    prob_ewma_alpha: float = 0.2,
) -> pd.DataFrame:
    """
    Offline-aligned 4h Markov regime:
      - returns: log returns
      - MarkovRegression(k=2, switching_variance=True, trend="c")
      - probabilities: FILTERED (past-only)
      - "up" state: higher weighted mean return under filtered probs
      - prob_up: EWMA(alpha, adjust=False)
      - state_up = int(prob_up > 0.5)
    """
    if ohlc_4h is None or ohlc_4h.empty:
        raise ValueError("compute_markov4h_series: empty ohlc_4h")

    df = _ensure_utc_index(ohlc_4h)
    if "close" not in df.columns:
        raise ValueError("compute_markov4h_series: missing close")

    close = df["close"].astype(float)
    ret = np.log(close).diff().dropna()
    if len(ret) < 80:
        raise ValueError(f"compute_markov4h_series: insufficient returns for Markov fit (n={len(ret)})")

    mod = MarkovRegression(ret, k_regimes=2, switching_variance=True, trend="c")
    res = mod.fit(disp=False)

    p0 = res.filtered_marginal_probabilities[0]
    p1 = res.filtered_marginal_probabilities[1]

    # Weighted mean returns under filtered probs
    m0 = float((ret * p0).sum() / max(1e-12, float(p0.sum())))
    m1 = float((ret * p1).sum() / max(1e-12, float(p1.sum())))
    up_state = 0 if m0 > m1 else 1

    prob_up_raw = (p0 if up_state == 0 else p1).reindex(df.index).ffill()
    prob_up = prob_up_raw.ewm(alpha=float(prob_ewma_alpha), adjust=False).mean()

    state_up = (prob_up > 0.5).astype(int)

    out = pd.DataFrame(
        {
            "markov_prob_up": prob_up.astype(float),
            "markov_state_up": state_up.astype(int),
            "markov_prob_up_raw": prob_up_raw.astype(float),
        },
        index=df.index,
    )
    out = out.dropna(subset=["markov_prob_up"])
    return out


def compute_markov4h_snapshot(ohlc_4h: pd.DataFrame, *, asof_ts: Any, cfg: Optional[RegimeConfig] = None) -> Dict[str, Any]:
    cfg = cfg or RegimeConfig()
    df = _ensure_utc_index(ohlc_4h)
    df = drop_incomplete_last_bar(df, "4h", asof_ts)

    ser = compute_markov4h_series(df, prob_ewma_alpha=cfg.markov4h_prob_ewma_alpha)
    if ser.empty:
        raise ValueError("compute_markov4h_snapshot: empty series after drop")

    asof = pd.Timestamp(asof_ts)
    if asof.tzinfo is None:
        asof = asof.tz_localize("UTC")
    else:
        asof = asof.tz_convert("UTC")

    ser2 = ser.loc[:asof]
    if ser2.empty:
        raise ValueError(f"compute_markov4h_snapshot: no row asof={asof} (min={ser.index.min()} max={ser.index.max()})")

    row = ser2.iloc[-1]
    return {
        "markov_state_4h": int(row["markov_state_up"]),
        "markov_prob_up_4h": float(row["markov_prob_up"]),
    }
