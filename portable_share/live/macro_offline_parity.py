from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:


    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        return out.sort_index()

    if "timestamp" in df.columns:
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.set_index("timestamp").sort_index()
        return out

    raise ValueError("df must have a DatetimeIndex or a 'timestamp' column")


def _asof_value(s: pd.Series, ts: pd.Timestamp) -> float:


    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)

    if s.index.tz is None:

        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    else:
        ts = ts.tz_localize(s.index.tz) if ts.tz is None else ts.tz_convert(s.index.tz)

    s2 = s.loc[:ts]
    if len(s2) == 0:
        return float("nan")
    v = s2.iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _infer_bars_per_day(idx: pd.DatetimeIndex) -> int:


    if len(idx) < 10:
        return 288
    diffs = idx.to_series().diff().dropna()
    med = diffs.median()
    if pd.isna(med) or med <= pd.Timedelta(0):
        return 288
    bpd = int(round(pd.Timedelta(days=1) / med))
    return bpd if bpd > 0 else 288


def _resample_ohlcv_like_offline(df5: pd.DataFrame, rule: str) -> pd.DataFrame:


    try:
        from live.indicators import resample_ohlcv
        return resample_ohlcv(df5, rule)
    except Exception:

        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        if "turnover" in df5.columns:
            agg["turnover"] = "sum"
        out = df5.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
        return out


def _atr_like_offline(df: pd.DataFrame, length: int) -> pd.Series:


    try:
        from live.indicators import atr
        return atr(df, length)
    except Exception:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        alpha = 1.0 / float(length)
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
        return atr


def map_to_left_index(target_index: pd.DatetimeIndex, s: pd.Series) -> pd.Series:


    return s.reindex(target_index, method="ffill")


def compute_asset_macd_4h_features(df5: pd.DataFrame, decision_ts: pd.Timestamp) -> Dict[str, float]:


    df5 = _ensure_dt_index(df5)
    df5_upto = df5.loc[:decision_ts]
    if len(df5_upto) == 0:
        return {
            "asset_macd_line_4h": float("nan"),
            "asset_macd_signal_4h": float("nan"),
            "asset_macd_hist_4h": float("nan"),
            "asset_macd_slope_4h": float("nan"),
        }

    df4h = _resample_ohlcv_like_offline(df5_upto, "4h")
    c4 = df4h["close"].astype(float)

    ema_fast = c4.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = c4.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - macd_signal
    macd_slope = macd_hist.diff(3)

    line_5 = map_to_left_index(df5_upto.index, macd_line)
    sig_5 = map_to_left_index(df5_upto.index, macd_signal)
    hist_5 = map_to_left_index(df5_upto.index, macd_hist)
    slope_5 = map_to_left_index(df5_upto.index, macd_slope)

    return {
        "asset_macd_line_4h": _asof_value(line_5, decision_ts),
        "asset_macd_signal_4h": _asof_value(sig_5, decision_ts),
        "asset_macd_hist_4h": _asof_value(hist_5, decision_ts),
        "asset_macd_slope_4h": _asof_value(slope_5, decision_ts),
    }


def compute_entry_quality_required_features(
    df5: pd.DataFrame, decision_ts: pd.Timestamp, cfg: Any
) -> Dict[str, float]:


    df5 = _ensure_dt_index(df5)
    df5_upto = df5.loc[:decision_ts]
    if len(df5_upto) == 0:
        return {
            "prior_1d_ret": float("nan"),
            "rv_3d": float("nan"),
            "consolidation_range_atr": float("nan"),
            "days_since_prev_break": float("nan"),
            "don_break_level": float("nan"),
            "don_dist_atr": float("nan"),
            "gap_from_1d_ma": float("nan"),
        }

    idx = df5_upto.index
    bpd = _infer_bars_per_day(idx)

    close = df5_upto["close"].astype(float)
    high = df5_upto["high"].astype(float)
    low = df5_upto["low"].astype(float)


    prior_1d_ret = close / close.shift(bpd) - 1.0


    logret = np.log(close).diff()
    rv_3d = logret.rolling(window=3 * bpd, min_periods=3 * bpd).std()


    atr_len = int(getattr(cfg, "ATR_LEN", 14))
    df1h = _resample_ohlcv_like_offline(df5_upto, "1h")
    atr1h = _atr_like_offline(df1h, atr_len)
    atr1h_5m = map_to_left_index(idx, atr1h)


    win_bars = getattr(cfg, "PULLBACK_WINDOW_BARS", None)
    if win_bars is None:

        win_bars = 12
    win_bars = int(win_bars)

    cons_range = high.rolling(win_bars, min_periods=win_bars).max() - low.rolling(
        win_bars, min_periods=win_bars
    ).min()
    consolidation_range_atr = cons_range / atr1h_5m.replace(0.0, np.nan)


    don_n_days = int(getattr(cfg, "DON_N_DAYS", 20))
    daily_high = high.resample("1D").max()
    don_daily = daily_high.rolling(don_n_days, min_periods=don_n_days).max().shift(1)
    don_5m = don_daily.reindex(idx, method="ffill")

    touch = high >= don_5m


    last_touch_ts = pd.Series(pd.NaT, index=idx, dtype=idx.dtype)
    if touch.any():
        last_touch_ts.loc[touch] = idx[touch.to_numpy()]
    last_touch_ts = last_touch_ts.ffill()


    idx_s = idx.to_series()
    days_since_prev_break = (idx_s - last_touch_ts).dt.total_seconds() / 86400.0


    daily_close = close.resample("1D").last()
    don_upper = daily_high.rolling(don_n_days, min_periods=don_n_days).max().shift(1)

    don_effect = don_upper.copy()
    don_effect.index = don_effect.index + pd.Timedelta(days=1)
    don_break_level_5m = don_effect.reindex(idx, method="ffill")


    don_dist_atr = (close - don_break_level_5m) / atr1h_5m.replace(0.0, np.nan)


    ma1d = close.rolling(window=bpd, min_periods=bpd).mean()
    gap_from_1d_ma = (close - ma1d) / atr1h_5m.replace(0.0, np.nan)

    return {
        "prior_1d_ret": _asof_value(prior_1d_ret, decision_ts),
        "rv_3d": _asof_value(rv_3d, decision_ts),
        "consolidation_range_atr": _asof_value(consolidation_range_atr, decision_ts),
        "days_since_prev_break": _asof_value(days_since_prev_break, decision_ts),
        "don_break_level": _asof_value(don_break_level_5m, decision_ts),
        "don_dist_atr": _asof_value(don_dist_atr, decision_ts),
        "gap_from_1d_ma": _asof_value(gap_from_1d_ma, decision_ts),
    }


def compute_cross_asset_daily_context(
    df5: pd.DataFrame, decision_ts: pd.Timestamp, *, prefix: str
) -> Dict[str, float]:


    df5 = _ensure_dt_index(df5)
    df5_upto = df5.loc[:decision_ts]
    if len(df5_upto) == 0:
        return {
            f"{prefix}_vol_regime_level": float("nan"),
            f"{prefix}_trend_slope": float("nan"),
        }

    df1d = _resample_ohlcv_like_offline(df5_upto, "1D")
    c1d = df1d["close"].astype(float)

    atr_daily = _atr_like_offline(df1d, 20)
    atr_pct_1d = atr_daily / c1d.replace(0.0, np.nan)

    baseline = atr_pct_1d.expanding(min_periods=50).median()
    vol_regime_level = (atr_pct_1d / baseline).shift(1)

    ma20 = c1d.rolling(20, min_periods=20).mean()
    ma50 = c1d.rolling(50, min_periods=50).mean()
    trend_slope = (ma20 - ma50).diff().shift(1)

    vol_5m = map_to_left_index(df5_upto.index, vol_regime_level)
    slope_5m = map_to_left_index(df5_upto.index, trend_slope)

    return {
        f"{prefix}_vol_regime_level": _asof_value(vol_5m, decision_ts),
        f"{prefix}_trend_slope": _asof_value(slope_5m, decision_ts),
    }


def load_regime_thresholds(bundle_dir: Path) -> Dict[str, Any]:
    rp = bundle_dir / "regimes_report.json"
    if not rp.exists():
        return {}
    try:
        d = json.loads(rp.read_text())
        return d.get("thresholds", {}) or {}
    except Exception:
        return {}
