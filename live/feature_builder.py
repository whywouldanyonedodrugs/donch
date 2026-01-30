# live/feature_builder.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from live.regime_truth import macro_regimes_asof

from . import indicators as ta
from .parity_utils import resample_ohlcv, map_to_left_index, donchian_upper_days_no_lookahead

from live.regimes_report import load_regime_thresholds
from live.oi_funding import oi_funding_features_at_decision, StaleDerivativesDataError


LOG = logging.getLogger("feature_builder")


class FeatureBuilder:
    """
    Strict-parity feature builder.
    Computes specific feature families using exact offline semantics.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.atr_len = int(cfg.get("ATR_LEN", 14))
        self.rsi_len = int(cfg.get("RSI_LEN", 14))
        self.adx_len = int(cfg.get("ADX_LEN", 14))
        self.vol_lookback = int(cfg.get("VOL_LOOKBACK_DAYS", 30))
        self.don_days = int(cfg.get("DON_N_DAYS", 20))
        self.pullback_win = int(cfg.get("PULLBACK_WINDOW_BARS", 12))

        # thresholds cache (bundle-specific; keyed by meta_dir)
        self._thr_meta_dir: Optional[Path] = None
        self._thr_obj = None

    def _get_regime_thresholds(self, meta_dir: Path):
        meta_dir = Path(meta_dir).resolve()
        if self._thr_meta_dir is None or self._thr_obj is None or meta_dir != self._thr_meta_dir:
            self._thr_obj = load_regime_thresholds(meta_dir)
            self._thr_meta_dir = meta_dir
        return self._thr_obj

    def compute_entry_quality_features(self, df5: pd.DataFrame, decision_ts: pd.Timestamp) -> Dict[str, float]:
        """
        Computes 'Entry Quality' features for the bar at decision_ts.
        Matches scout.py logic exactly (rolling windows, min_periods, NaNs).
        """
        out: Dict[str, float] = {}

        if df5.empty or decision_ts not in df5.index:
            return out

        bars_per_day = 288

        # --- 1) 1h Context (ATR, RSI, ADX) ---
        df1h = resample_ohlcv(df5, "1h")
        if not df1h.empty:
            atr1h_s = ta.atr(df1h, self.atr_len)
            rsi1h_s = ta.rsi(df1h["close"], self.rsi_len)
            adx1h_s = ta.adx(df1h, self.adx_len)

            atr1h_5m = map_to_left_index(df5.index, atr1h_s)
            rsi1h_5m = map_to_left_index(df5.index, rsi1h_s)
            adx1h_5m = map_to_left_index(df5.index, adx1h_s)

            out["atr_1h"] = float(atr1h_5m.loc[decision_ts])
            out["rsi_1h"] = float(rsi1h_5m.loc[decision_ts])
            out["adx_1h"] = float(adx1h_5m.loc[decision_ts])

            close_now = float(df5.loc[decision_ts, "close"])
            out["atr_pct"] = (out["atr_1h"] / close_now) if close_now > 0 else 0.0
        else:
            out["atr_1h"] = np.nan
            out["rsi_1h"] = np.nan
            out["adx_1h"] = np.nan
            out["atr_pct"] = np.nan

        # --- 2) Volume Multiple ---
        vol_lb_bars = bars_per_day * self.vol_lookback
        min_periods = max(5, vol_lb_bars // 10)
        vol_med = df5["volume"].rolling(vol_lb_bars, min_periods=min_periods).median()
        vol_mult_s = df5["volume"] / vol_med.replace(0.0, np.nan)
        out["vol_mult"] = float(vol_mult_s.loc[decision_ts])

        # --- 3) Days Since Prev Break (Left/Left daily highs) ---
        daily_high = df5["high"].resample("1D", label="left", closed="left").max().dropna()
        don_daily = daily_high.rolling(self.don_days, min_periods=self.don_days).max().shift(1)
        don_5m = don_daily.reindex(df5.index, method="ffill")

        touch = df5["high"] >= don_5m
        touch_upto = touch.loc[:decision_ts]
        if touch_upto.any():
            last_touch_ts = touch_upto[touch_upto].index[-1]
            delta = (decision_ts - last_touch_ts).total_seconds() / 86400.0
            out["days_since_prev_break"] = float(delta)
        else:
            out["days_since_prev_break"] = np.nan

        # --- 4) Consolidation Range ATR ---
        high_win = df5["high"].rolling(self.pullback_win).max()
        low_win = df5["low"].rolling(self.pullback_win).min()
        cons_range = high_win - low_win

        val = cons_range.loc[decision_ts]
        if "atr1h_5m" in locals():
            atr_series = atr1h_5m
        else:
            atr_series = pd.Series(np.nan, index=df5.index)

        atr_safe = atr_series.replace(0.0, np.nan)
        if np.isfinite(atr_safe.loc[decision_ts]):
            out["consolidation_range_atr"] = float(val / atr_safe.loc[decision_ts])
        else:
            out["consolidation_range_atr"] = np.nan

        # --- 5) Prior 1d Return ---
        prior_ret_s = df5["close"] / df5["close"].shift(bars_per_day) - 1.0
        out["prior_1d_ret"] = float(prior_ret_s.loc[decision_ts])

        # --- 6) Realized Vol (3d), explicit ddof=1 ---
        rv_win = 3 * bars_per_day
        log_ret = np.log(df5["close"]).diff()
        rv_3d_s = log_ret.rolling(rv_win).std(ddof=1)
        out["rv_3d"] = float(rv_3d_s.loc[decision_ts])

        # --- 7) Donch Break Level (Right/Right completed days) ---
        don_upper_s = donchian_upper_days_no_lookahead(df5["high"], self.don_days)
        out["don_break_level"] = float(don_upper_s.loc[decision_ts])
        out["don_break_len"] = float(self.don_days)

        # --- 8) Donch Dist ATR ---
        if np.isfinite(out.get("don_break_level", np.nan)) and np.isfinite(out.get("atr_1h", np.nan)) and out["atr_1h"] > 0:
            close_now = float(df5.loc[decision_ts, "close"])
            out["don_dist_atr"] = (close_now - out["don_break_level"]) / out["atr_1h"]
        else:
            out["don_dist_atr"] = np.nan

        return out

    def compute_oi_funding_features(
        self,
        df5: pd.DataFrame,
        decision_ts: pd.Timestamp,
        *,
        meta_dir: Path,
    ) -> Dict[str, float]:
        """
        Computes OI/Funding features at decision_ts with strict as-of semantics.
        Fail-closed on missing required raw cols or stale feeds by raising.
        """
        thr = self._get_regime_thresholds(Path(meta_dir))

        # Default staleness: 8h unless configured; use None to disable.
        max_age_raw = self.cfg.get("DERIV_MAX_AGE", "8h")
        staleness_max_age = pd.Timedelta(max_age_raw) if max_age_raw else None

        try:
            return oi_funding_features_at_decision(
                df5,
                decision_ts,
                thresholds=thr,
                staleness_max_age=staleness_max_age,
            )
        except StaleDerivativesDataError:
            # Upstream must treat as safe-mode/no trade
            raise
        except KeyError:
            # Missing required raw -> fail-closed (upstream safe-mode/no trade)
            raise

    def compute_features_for_decision(
        self,
        df5: pd.DataFrame,
        decision_ts: pd.Timestamp,
        *,
        meta_dir: Path,
        include_entry_quality: bool = True,
        include_oi_funding: bool = True,
    ) -> Dict[str, float]:
        """
        Aggregates feature families into a single feature dict for scoring.
        This method is the intended integration point for Sprint 2.2.

        It does NOT use wall-clock time. Everything is as-of decision_ts.
        """
        out: Dict[str, float] = {}

        if include_entry_quality:
            out.update(self.compute_entry_quality_features(df5, decision_ts))

        if include_oi_funding:
            out.update(self.compute_oi_funding_features(df5, decision_ts, meta_dir=meta_dir))

        return out
