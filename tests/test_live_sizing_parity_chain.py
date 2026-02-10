import unittest

from live.live_trader import LiveTrader


class TestLiveSizingParityChain(unittest.TestCase):
    def _mk_trader(self) -> LiveTrader:
        trader = LiveTrader.__new__(LiveTrader)
        trader.cfg = {
            "META_SIZING_ENABLED": True,
            "META_SIZING_P0": 0.40,
            "META_SIZING_P1": 0.90,
            "META_SIZING_MIN": 1.00,
            "META_SIZING_MAX": 2.00,
            "DYN_MACD_HIST_THRESH": 0.0,
            "REGIME_DOWNSIZE_MULT": 0.5,
            "SIZE_MIN_CAP": 0.0,
            "SIZE_MAX_CAP": 10.0,
        }
        return trader

    def test_dyn_size_multiplier_uses_linear_map_then_eth_down_mult(self) -> None:
        trader = self._mk_trader()
        # p=0.65 maps to 1.50 on [0.40..0.90] -> [1.0..2.0], then *0.5 on negative hist.
        out = LiveTrader._dyn_size_multiplier(trader, 0.65, -0.1)
        self.assertAlmostEqual(out, 0.75, places=8)

    def test_dyn_size_multiplier_clamps_to_caps(self) -> None:
        trader = self._mk_trader()
        trader.cfg["SIZE_MIN_CAP"] = 1.20
        trader.cfg["SIZE_MAX_CAP"] = 1.30
        out = LiveTrader._dyn_size_multiplier(trader, 0.65, 0.1)
        self.assertAlmostEqual(out, 1.30, places=8)

