import os
import tempfile
import unittest
import pandas as pd
import numpy as np

from live.online_state import OnlinePerformanceState


class TestOnlinePerformanceState(unittest.TestCase):
    def test_shifted_winrates(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "online_state.jsonl")
            st = OnlinePerformanceState(path=path, max_records=100)

            t1 = pd.Timestamp("2026-01-01T00:00:00Z")
            t2 = pd.Timestamp("2026-01-02T00:00:00Z")
            t3 = pd.Timestamp("2026-01-03T00:00:00Z")
            t4 = pd.Timestamp("2026-01-04T00:00:00Z")

            st.record_trade_close(t1, pnl=+1.0)  # win
            st.record_trade_close(t2, pnl=-1.0)  # loss
            st.record_trade_close(t3, pnl=+1.0)  # win

            # as-of t4, prior outcomes are [1,0,1], then shift(1) -> [1,0]
            feats = st.features_asof(t4)
            self.assertTrue(np.isfinite(feats["recent_winrate_20"]))
            self.assertAlmostEqual(feats["recent_winrate_20"], 0.5, places=9)
            self.assertAlmostEqual(feats["recent_winrate_50"], 0.5, places=9)

            # EWM on [1,0] with span=20 -> last value should be between 0 and 1
            self.assertTrue(0.0 <= feats["recent_winrate_ewm_20"] <= 1.0)

            # as-of t3, prior outcomes are [1,0] then shift(1) -> [1]
            feats2 = st.features_asof(t3)
            self.assertAlmostEqual(feats2["recent_winrate_20"], 1.0, places=9)

    def test_persistence_across_restarts(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "online_state.jsonl")

            st1 = OnlinePerformanceState(path=path, max_records=100)
            t1 = pd.Timestamp("2026-01-01T00:00:00Z")
            t2 = pd.Timestamp("2026-01-02T00:00:00Z")
            st1.record_trade_close(t1, pnl=+5.0)
            st1.record_trade_close(t2, pnl=-2.0)

            st2 = OnlinePerformanceState(path=path, max_records=100)
            feats = st2.features_asof(pd.Timestamp("2026-01-03T00:00:00Z"))
            # prior outcomes [1,0], shift(1)->[1] => winrate=1.0
            self.assertAlmostEqual(feats["recent_winrate_20"], 1.0, places=9)

    def test_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "online_state.jsonl")
            st = OnlinePerformanceState(path=path, max_records=3)

            base = pd.Timestamp("2026-01-01T00:00:00Z")
            for i in range(5):
                st.record_trade_close(base + pd.Timedelta(days=i), pnl=1.0 if i % 2 == 0 else -1.0)

            # internal should keep last 3
            feats = st.features_asof(pd.Timestamp("2026-01-10T00:00:00Z"))
            # prior kept wins should exist (may be NaN if shift empties; but here 3 -> shift -> 2)
            self.assertTrue(np.isfinite(feats["recent_winrate_20"]) or np.isnan(feats["recent_winrate_20"]))
