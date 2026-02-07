import unittest
import inspect

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

    def test_open_position_has_no_legacy_min_winprob_fallback(self) -> None:
        src = inspect.getsource(LiveTrader._open_position)
        self.assertNotIn("MIN_WINPROB_TO_TRADE", src)

    def test_open_position_meta_threshold_handles_none(self) -> None:
        src = inspect.getsource(LiveTrader._open_position)
        self.assertIn("meta_thresh_raw = self.cfg.get(\"META_PROB_THRESHOLD\", None)", src)
        self.assertNotIn("float(self.cfg.get(\"META_PROB_THRESHOLD\", 0.0))", src)

    def test_summary_report_uses_current_regime_detector(self) -> None:
        src = inspect.getsource(LiveTrader._generate_summary_report)
        self.assertIn("get_current_regime", src)
        self.assertIn("Regime (current)", src)

    def test_handle_cmd_includes_regime_command(self) -> None:
        src = inspect.getsource(LiveTrader._handle_cmd)
        self.assertIn('elif root == "/regime":', src)
        self.assertIn("get_current_regime", src)

    def test_no_meta_gate_does_not_apply_scope_veto(self) -> None:
        src = inspect.getsource(LiveTrader._scan_symbol_for_signal)
        self.assertIn("if pstar is None:", src)
        self.assertIn('reason = "no_prob_gate"', src)
        self.assertIn("if scope_gate_enabled and (not bool(scope_ok))", src)
