import unittest

from live.live_trader import LiveTrader


class TestRiskOnFormula(unittest.TestCase):
    def _mk_trader(self) -> LiveTrader:
        trader = LiveTrader.__new__(LiveTrader)
        trader.regime_thresholds = {
            "btc_vol_hi": 1.5,
            "crowd_z_high": 1.0,
            "crowd_z_low": -1.0,
        }
        return trader

    def test_risk_on_requires_strict_positive_trend(self) -> None:
        trader = self._mk_trader()

        # slope == 0 should be risk_off under strict backtest parity.
        meta0 = {
            "regime_up": 1.0,
            "btcusdt_trend_slope": 0.0,
            "btcusdt_vol_regime_level": 1.0,
        }
        LiveTrader._augment_meta_with_regime_sets(trader, meta0)
        self.assertEqual(float(meta0.get("risk_on", -1.0)), 0.0)
        self.assertEqual(float(meta0.get("risk_on_1", -1.0)), 0.0)

        # slope > 0 and vol below threshold => risk_on.
        meta1 = {
            "regime_up": 1.0,
            "btcusdt_trend_slope": 0.01,
            "btcusdt_vol_regime_level": 1.49,
        }
        LiveTrader._augment_meta_with_regime_sets(trader, meta1)
        self.assertEqual(float(meta1.get("risk_on", -1.0)), 1.0)
        self.assertEqual(float(meta1.get("risk_on_1", -1.0)), 1.0)

        # vol must be strictly below btc_vol_hi.
        meta2 = {
            "regime_up": 1.0,
            "btcusdt_trend_slope": 0.01,
            "btcusdt_vol_regime_level": 1.5,
        }
        LiveTrader._augment_meta_with_regime_sets(trader, meta2)
        self.assertEqual(float(meta2.get("risk_on", -1.0)), 0.0)

    def test_risk_on_uses_generic_btc_keys_fallback(self) -> None:
        trader = self._mk_trader()
        meta = {
            "regime_up": 1.0,
            "btc_trend_slope": 0.05,
            "btc_vol_regime_level": 1.0,
        }
        LiveTrader._augment_meta_with_regime_sets(trader, meta)
        self.assertEqual(float(meta.get("risk_on", -1.0)), 1.0)
        self.assertEqual(float(meta.get("risk_on_1", -1.0)), 1.0)


if __name__ == "__main__":
    unittest.main()
