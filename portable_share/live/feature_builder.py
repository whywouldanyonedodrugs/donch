from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional

from . import indicators as ta
from .indicators import vwap_stack_features
from .macro_offline_parity import compute_entry_quality_required_features

def _resample_ohlcv(df5: pd.DataFrame, rule: str) -> pd.DataFrame:


    o = df5["open"].resample(rule, label="right", closed="right").first()
    h = df5["high"].resample(rule, label="right", closed="right").max()
    l = df5["low"].resample(rule, label="right", closed="right").min()
    c = df5["close"].resample(rule, label="right", closed="right").last()
    v = df5["volume"].resample(rule, label="right", closed="right").sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    out = out.dropna()
    return out


def _last_ts_asof(df: pd.DataFrame, decision_ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    sub = df.loc[df.index <= decision_ts]
    if sub.empty:
        return None
    return sub.index[-1]

def _prebreak_congestion_atr_15m(
    df5: pd.DataFrame,
    decision_ts: pd.Timestamp,
    atr_len: int,
    lookback_bars_15m: int,
) -> Optional[float]:


    if df5 is None or df5.empty:
        return None

    df15 = _resample_ohlcv(df5, "15min")
    ts15 = _last_ts_asof(df15, decision_ts)
    if ts15 is None:
        return None

    win = df15.loc[:ts15].tail(int(lookback_bars_15m))
    if win.empty:
        return None

    rng = float(win["high"].max() - win["low"].min())
    if not np.isfinite(rng):
        return None

    try:
        atr15 = ta.atr(df15[["open", "high", "low", "close"]], int(atr_len))
        a = float(atr15.loc[ts15])
    except Exception:
        return None

    if (not np.isfinite(a)) or a == 0.0:
        return None

    return float(rng / a)

class FeatureBuilder:


    def __init__(self, cfg: dict, logger=None):
        self.cfg = cfg or {}
        self.log = logger

    def _days_since_last_break(self, break_series: pd.Series, asof_day: pd.Timestamp) -> float:


        try:
            s = break_series.loc[:asof_day]
            if s.empty:
                return float("nan")
            idx = s[s.astype(bool)].index
            if len(idx) == 0:
                return float("nan")
            last_break = idx[-1]
            return float((asof_day - last_break).days)
        except Exception:
            return float("nan")

    def compute_entry_quality_features(self, df5: pd.DataFrame, decision_ts: pd.Timestamp) -> Dict[str, float]:


        out: Dict[str, float] = {}

        if df5 is None or df5.empty:
            return out


        df5 = df5.sort_index()
        df5 = df5.loc[df5.index <= decision_ts]
        if df5.empty:
            return out

        atr_len = int(self.cfg.get("ATR_LEN", 14))
        rsi_len = int(self.cfg.get("RSI_LEN", 14))
        don_n_days = int(self.cfg.get("DON_N_DAYS", 20))
        gap_ma_len = int(self.cfg.get("GAP_1D_MA_LEN", 20))


        df1d = _resample_ohlcv(df5, "1D")
        ts_1d = _last_ts_asof(df1d, decision_ts)
        if ts_1d is not None:
            asof_day = ts_1d


            don_upper = df1d["high"].rolling(don_n_days, min_periods=don_n_days).max().shift(1)
            atr_1d = ta.atr(df1d[["open", "high", "low", "close"]], atr_len)

            try:
                don_level = float(don_upper.loc[asof_day])
            except Exception:
                don_level = float("nan")

            try:
                close_1d = float(df1d["close"].loc[asof_day])
            except Exception:
                close_1d = float("nan")

            try:
                atrv = float(atr_1d.loc[asof_day])
            except Exception:
                atrv = float("nan")

            out["don_break_level"] = don_level


            if np.isfinite(close_1d) and np.isfinite(don_level) and np.isfinite(atrv) and atrv != 0.0:
                out["don_dist_atr"] = float((close_1d - don_level) / atrv)
            else:
                out["don_dist_atr"] = float("nan")


            break_series = (df1d["close"] > don_upper).astype(float)
            out["days_since_prev_break"] = self._days_since_last_break(break_series, asof_day)


            ma = ta.sma(df1d["close"], gap_ma_len)
            try:
                ma_val = float(ma.loc[asof_day])
                if np.isfinite(close_1d) and np.isfinite(ma_val) and ma_val != 0.0:
                    out["gap_from_1d_ma"] = float((close_1d - ma_val) / ma_val)
                else:
                    out["gap_from_1d_ma"] = float("nan")
            except Exception:
                out["gap_from_1d_ma"] = float("nan")


        req_eq = compute_entry_quality_required_features(df5, decision_ts, cfg=self.cfg)
        out.update(req_eq)


        try:

            cong_len = int(self.cfg.get("PREBREAK_CONGESTION_N15", 12))
            cong = _prebreak_congestion_atr_15m(df5, decision_ts=decision_ts, atr_len=atr_len, lookback_bars_15m=cong_len)
            out["congestion_range_atr"] = float(cong) if cong is not None else float("nan")
        except Exception:
            out["congestion_range_atr"] = float("nan")


        out["prebreak_congestion"] = float(out.get("congestion_range_atr", float("nan")))


        try:
            vwap_lb = int(self.cfg.get("VWAP_LOOKBACK_BARS", 12))
            vwap_band = float(self.cfg.get("VWAP_BAND_PCT", 0.004))
            df5_cut = df5.loc[:decision_ts]
            vwap = vwap_stack_features(df5_cut, lookback_bars=vwap_lb, band_pct=vwap_band)
            for k, v in vwap.items():
                out[k] = float(v) if v is not None else float("nan")
        except Exception:
            pass


        try:
            df15 = _resample_ohlcv(df5, "15min")
            ts15 = _last_ts_asof(df15, decision_ts)
            if ts15 is not None:
                rsi15 = ta.rsi(df15["close"], rsi_len)
                out["asset_rsi_15m"] = float(rsi15.loc[ts15])
            else:
                out["asset_rsi_15m"] = float("nan")
        except Exception:
            out["asset_rsi_15m"] = float("nan")


        try:
            df4h = _resample_ohlcv(df5, "4h")
            ts4h = _last_ts_asof(df4h, decision_ts)
            if ts4h is not None:
                rsi4h = ta.rsi(df4h["close"], rsi_len)
                out["asset_rsi_4h"] = float(rsi4h.loc[ts4h])
            else:
                out["asset_rsi_4h"] = float("nan")
        except Exception:
            out["asset_rsi_4h"] = float("nan")


        try:
            df1h = _resample_ohlcv(df5, "1h")
            macd1 = ta.macd(df1h["close"])
            m1 = macd1.loc[:decision_ts].tail(2)
            if len(m1) >= 1:
                last = m1.iloc[-1]
                out["asset_macd_line_1h"] = float(last["macd"])
                out["asset_macd_signal_1h"] = float(last["signal"])
                out["asset_macd_hist_1h"] = float(last["hist"])
                if len(m1) >= 2:
                    prev = m1.iloc[-2]
                    out["asset_macd_slope_1h"] = float(last["hist"] - prev["hist"])
                else:
                    out["asset_macd_slope_1h"] = float("nan")
            else:
                out["asset_macd_line_1h"] = float("nan")
                out["asset_macd_signal_1h"] = float("nan")
                out["asset_macd_hist_1h"] = float("nan")
                out["asset_macd_slope_1h"] = float("nan")
        except Exception:
            out["asset_macd_line_1h"] = float("nan")
            out["asset_macd_signal_1h"] = float("nan")
            out["asset_macd_hist_1h"] = float("nan")
            out["asset_macd_slope_1h"] = float("nan")


        try:
            df4h = _resample_ohlcv(df5, "4h")
            macd4 = ta.macd(df4h["close"])
            m4 = macd4.loc[:decision_ts].tail(2)
            if len(m4) >= 1:
                last = m4.iloc[-1]
                out["asset_macd_line_4h"] = float(last["macd"])
                out["asset_macd_signal_4h"] = float(last["signal"])
                out["asset_macd_hist_4h"] = float(last["hist"])
                if len(m4) >= 2:
                    prev = m4.iloc[-2]
                    out["asset_macd_slope_4h"] = float(last["hist"] - prev["hist"])
                else:
                    out["asset_macd_slope_4h"] = float("nan")
            else:
                out["asset_macd_line_4h"] = float("nan")
                out["asset_macd_signal_4h"] = float("nan")
                out["asset_macd_hist_4h"] = float("nan")
                out["asset_macd_slope_4h"] = float("nan")
        except Exception:
            out["asset_macd_line_4h"] = float("nan")
            out["asset_macd_signal_4h"] = float("nan")
            out["asset_macd_hist_4h"] = float("nan")
            out["asset_macd_slope_4h"] = float("nan")


        try:
            df1h = _resample_ohlcv(df5, "1h")
            ts1h = _last_ts_asof(df1h, decision_ts)
            if ts1h is not None:
                atr1 = ta.atr(df1h[["open", "high", "low", "close"]], atr_len)
                c1 = float(df1h["close"].loc[ts1h])
                a1 = float(atr1.loc[ts1h])
                out["asset_vol_1h"] = float((a1 / c1) * 100.0) if np.isfinite(a1) and np.isfinite(c1) and c1 != 0.0 else float("nan")
            else:
                out["asset_vol_1h"] = float("nan")
        except Exception:
            out["asset_vol_1h"] = float("nan")

        try:
            df4h = _resample_ohlcv(df5, "4h")
            ts4h = _last_ts_asof(df4h, decision_ts)
            if ts4h is not None:
                atr4 = ta.atr(df4h[["open", "high", "low", "close"]], atr_len)
                c4 = float(df4h["close"].loc[ts4h])
                a4 = float(atr4.loc[ts4h])
                out["asset_vol_4h"] = float((a4 / c4) * 100.0) if np.isfinite(a4) and np.isfinite(c4) and c4 != 0.0 else float("nan")
            else:
                out["asset_vol_4h"] = float("nan")
        except Exception:
            out["asset_vol_4h"] = float("nan")

        return out
