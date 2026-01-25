# tests/test_parity.py
import unittest
import pandas as pd
import numpy as np
from live.feature_builder import FeatureBuilder
from live.parity_utils import donchian_upper_days_no_lookahead, resample_ohlcv, map_to_left_index, _norm_tf
from live import indicators as ta

class TestParity(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range(start="2024-01-01", periods=288 * 40, freq="5min", tz="UTC")
        n = len(dates)

        np.random.seed(42)
        returns = np.random.normal(0, 0.001, n)
        price = 100 * np.exp(np.cumsum(returns))

        self.df5 = pd.DataFrame({
            "open": price,
            "high": price * 1.001,
            "low":  price * 0.999,
            "close": price,
            "volume": np.random.randint(100, 1000, n).astype(float),
        }, index=dates)

        self.cfg = {
            "ATR_LEN": 14, "RSI_LEN": 14, "ADX_LEN": 14,
            "VOL_LOOKBACK_DAYS": 30, "DON_N_DAYS": 20,
            "PULLBACK_WINDOW_BARS": 12,
        }
        self.builder = FeatureBuilder(self.cfg)

    def test_norm_tf(self):
        self.assertEqual(_norm_tf("15m"), "15min")
        self.assertEqual(_norm_tf("1h"), "1h")
        self.assertEqual(_norm_tf("1d"), "1D")
        self.assertEqual(_norm_tf("1D"), "1D")

    def test_donchian_helper_parity(self):
        highs = self.df5["high"]

        daily = highs.resample("1D", label="right", closed="right").max().dropna()
        don_daily = daily.rolling(20, min_periods=20).max().shift(1)

        floored_idx = highs.index.floor("D")
        don_5m_offline = don_daily.reindex(floored_idx).ffill()
        don_5m_offline_series = pd.Series(don_5m_offline.values, index=highs.index)

        don_5m_live = donchian_upper_days_no_lookahead(highs, 20)

        pd.testing.assert_series_equal(
            don_5m_offline_series.dropna(),
            don_5m_live.dropna(),
            check_names=False
        )

    def test_entry_quality_parity_all_fields(self):
        decision_ts = self.df5.index[-1]
        live_feats = self.builder.compute_entry_quality_features(self.df5, decision_ts)

        # Offline replication

        # 1h context
        df1h = resample_ohlcv(self.df5, "1h")
        atr1h = map_to_left_index(self.df5.index, ta.atr(df1h, 14))
        rsi1h = map_to_left_index(self.df5.index, ta.rsi(df1h["close"], 14))
        adx1h = map_to_left_index(self.df5.index, ta.adx(df1h, 14))

        atr_1h_off = float(atr1h.loc[decision_ts])
        rsi_1h_off = float(rsi1h.loc[decision_ts])
        adx_1h_off = float(adx1h.loc[decision_ts])

        close_now = float(self.df5.loc[decision_ts, "close"])
        atr_pct_off = (atr_1h_off / close_now) if close_now > 0 else 0.0

        self.assertAlmostEqual(live_feats["atr_1h"], atr_1h_off, places=10)
        self.assertAlmostEqual(live_feats["rsi_1h"], rsi_1h_off, places=10)
        self.assertAlmostEqual(live_feats["adx_1h"], adx_1h_off, places=10)
        self.assertAlmostEqual(live_feats["atr_pct"], atr_pct_off, places=12)

        # Vol mult
        vol = self.df5["volume"]
        lb = 288 * 30
        med = vol.rolling(lb, min_periods=max(5, lb // 10)).median()
        vol_mult_off = float((vol / med.replace(0, np.nan)).loc[decision_ts])
        self.assertAlmostEqual(live_feats["vol_mult"], vol_mult_off, places=10)

        # days_since_prev_break (Left/Left daily highs)
        daily_high = self.df5["high"].resample("1D", label="left", closed="left").max().dropna()
        don_daily = daily_high.rolling(20, min_periods=20).max().shift(1)
        don_5m = don_daily.reindex(self.df5.index, method="ffill")
        touch = self.df5["high"] >= don_5m
        touch_upto = touch.loc[:decision_ts]
        if touch_upto.any():
            last_touch_ts = touch_upto[touch_upto].index[-1]
            days_since_off = float((decision_ts - last_touch_ts).total_seconds() / 86400.0)
        else:
            days_since_off = np.nan

        if np.isnan(days_since_off):
            self.assertTrue(np.isnan(live_feats["days_since_prev_break"]))
        else:
            self.assertAlmostEqual(live_feats["days_since_prev_break"], days_since_off, places=12)

        # Consolidation range ATR
        high_win = self.df5["high"].rolling(12).max()
        low_win = self.df5["low"].rolling(12).min()
        cons_range = high_win - low_win
        cons_off = float((cons_range / atr1h.replace(0, np.nan)).loc[decision_ts])
        self.assertAlmostEqual(live_feats["consolidation_range_atr"], cons_off, places=10)

        # Prior 1d ret
        prior_ret_off = float((self.df5["close"] / self.df5["close"].shift(288) - 1.0).loc[decision_ts])
        self.assertAlmostEqual(live_feats["prior_1d_ret"], prior_ret_off, places=12)

        # RV 3d (ddof=1)
        log_ret = np.log(self.df5["close"]).diff()
        rv_3d_off = float(log_ret.rolling(3 * 288).std(ddof=1).loc[decision_ts])
        self.assertAlmostEqual(live_feats["rv_3d"], rv_3d_off, places=12)

        # Donch break level + len (Right/Right)
        don_s = donchian_upper_days_no_lookahead(self.df5["high"], 20)
        don_level_off = float(don_s.loc[decision_ts])
        self.assertAlmostEqual(live_feats["don_break_level"], don_level_off, places=12)
        self.assertAlmostEqual(live_feats["don_break_len"], 20.0, places=12)

        # Don dist ATR
        don_dist_off = (close_now - don_level_off) / atr_1h_off if atr_1h_off > 0 else np.nan
        if np.isnan(don_dist_off):
            self.assertTrue(np.isnan(live_feats["don_dist_atr"]))
        else:
            self.assertAlmostEqual(live_feats["don_dist_atr"], don_dist_off, places=12)

if __name__ == "__main__":
    unittest.main()
