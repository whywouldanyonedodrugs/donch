import unittest
import pandas as pd

from live.live_trader import create_entry_cid


class TestClientOrderId(unittest.TestCase):
    def test_entry_cid_stable(self):
        ts = pd.Timestamp("2026-01-30T12:35:00Z")
        c1 = create_entry_cid(symbol="BTCUSDT", decision_ts=ts, side="LONG", tag="ENTRY")
        c2 = create_entry_cid(symbol="BTCUSDT", decision_ts=ts, side="LONG", tag="ENTRY")
        self.assertEqual(c1, c2)
        self.assertLessEqual(len(c1), 36)

    def test_entry_cid_changes_with_inputs(self):
        ts1 = pd.Timestamp("2026-01-30T12:35:00Z")
        ts2 = pd.Timestamp("2026-01-30T12:40:00Z")
        c1 = create_entry_cid(symbol="BTCUSDT", decision_ts=ts1, side="LONG", tag="ENTRY")
        c2 = create_entry_cid(symbol="BTCUSDT", decision_ts=ts2, side="LONG", tag="ENTRY")
        c3 = create_entry_cid(symbol="BTCUSDT", decision_ts=ts1, side="SHORT", tag="ENTRY")
        self.assertNotEqual(c1, c2)
        self.assertNotEqual(c1, c3)
