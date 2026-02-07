import unittest

from live.live_trader import LiveTrader


class TestLiveMetaProbeKnob(unittest.TestCase):
    def _mk_trader(self, probe: float) -> LiveTrader:
        trader = LiveTrader.__new__(LiveTrader)
        trader.cfg = {
            "META_SIZING_ENABLED": True,
            "META_RISK_OFF_PROBE_ENABLED": True,
            "META_SIZING_FAIL_CLOSED": False,
            "RISK_OFF_PROBE_MULT": probe,
        }
        trader._meta_size_mult = lambda _p: 0.80
        trader._meta_bins = [(0.0, 1.0, 0.80)]
        trader._meta_curve_x = None
        trader._meta_curve_y = None
        return trader

    def test_risk_off_probe_knob_changes_meta_multiplier(self) -> None:
        t1 = self._mk_trader(0.01)
        t2 = self._mk_trader(0.25)

        m1 = LiveTrader._resolve_meta_multiplier(t1, 0.70, 0.0)
        m2 = LiveTrader._resolve_meta_multiplier(t2, 0.70, 0.0)

        self.assertAlmostEqual(m1, 0.01, places=8)
        self.assertAlmostEqual(m2, 0.25, places=8)
        self.assertNotEqual(m1, m2)

    def test_risk_on_not_probe_capped(self) -> None:
        t = self._mk_trader(0.01)
        m = LiveTrader._resolve_meta_multiplier(t, 0.70, 1.0)
        self.assertAlmostEqual(m, 0.80, places=8)

    def test_fail_closed_caps_to_probe_when_curve_missing(self) -> None:
        trader = LiveTrader.__new__(LiveTrader)
        trader.cfg = {
            "META_SIZING_ENABLED": True,
            "META_RISK_OFF_PROBE_ENABLED": False,
            "META_SIZING_FAIL_CLOSED": True,
            "RISK_OFF_PROBE_MULT": 0.03,
        }
        trader._meta_bins = None
        trader._meta_curve_x = None
        trader._meta_curve_y = None
        trader._meta_size_mult = lambda _p: 0.99

        m = LiveTrader._resolve_meta_multiplier(trader, 0.70, 1.0)
        self.assertAlmostEqual(m, 0.03, places=8)
