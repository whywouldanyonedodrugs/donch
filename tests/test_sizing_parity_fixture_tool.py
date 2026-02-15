import unittest

from tools.sizing_parity_fixture import Case, live_chain, offline_reference_chain


class TestSizingParityFixtureTool(unittest.TestCase):
    def test_live_matches_offline_reference_cash_mode(self) -> None:
        cfg = {
            "RISK_MODE": "fixed",
            "FIXED_RISK_CASH": 20.0,
            "RISK_PCT": 0.01,
            "REGIME_BLOCK_WHEN_DOWN": False,
            "REGIME_SIZE_WHEN_DOWN": 0.2,
            "META_SIZING_ENABLED": True,
            "META_SIZING_P0": 0.42,
            "META_SIZING_P1": 0.90,
            "META_SIZING_MIN": 1.0,
            "META_SIZING_MAX": 2.0,
            "DYN_MACD_HIST_THRESH": 0.0,
            "REGIME_DOWNSIZE_MULT": 1.0,
            "SIZE_MIN_CAP": 0.0,
            "SIZE_MAX_CAP": 2.0,
            "RISK_OFF_PROBE_MULT": 0.25,
            "BTC_VOL_HI": 0.75,
            "NOTIONAL_CAP_PCT_OF_EQUITY": 1.0,
            "MAX_LEVERAGE": 10.0,
        }
        c = Case(
            case_id=1,
            equity=1000.0,
            regime_up=0,
            btc_trend_slope=-0.1,
            btc_vol_regime_level=1.0,
            p_cal=0.80,
            eth_macd_hist_4h=0.2,
            risk_scale=None,
            entry=100.0,
            sl_initial=95.0,
        )
        lv = live_chain(cfg, c)
        bt = offline_reference_chain(cfg, c)
        self.assertAlmostEqual(float(lv["size_mult_final"]), float(bt["size_mult_final"]), places=12)
        self.assertAlmostEqual(float(lv["risk_usd"]), float(bt["risk_usd"]), places=12)
        self.assertAlmostEqual(float(lv["qty"]), float(bt["qty"]), places=12)

    def test_live_matches_offline_reference_percent_mode_with_risk_scale(self) -> None:
        cfg = {
            "RISK_MODE": "percent",
            "RISK_PCT": 0.01,
            "FIXED_RISK_CASH": 20.0,
            "REGIME_BLOCK_WHEN_DOWN": False,
            "REGIME_SIZE_WHEN_DOWN": 1.0,
            "META_SIZING_ENABLED": True,
            "META_SIZING_P0": 0.42,
            "META_SIZING_P1": 0.90,
            "META_SIZING_MIN": 1.0,
            "META_SIZING_MAX": 2.0,
            "DYN_MACD_HIST_THRESH": 0.0,
            "REGIME_DOWNSIZE_MULT": 1.0,
            "SIZE_MIN_CAP": 0.0,
            "SIZE_MAX_CAP": 2.0,
            "RISK_OFF_PROBE_MULT": 0.25,
            "BTC_VOL_HI": 0.75,
            "NOTIONAL_CAP_PCT_OF_EQUITY": 1.0,
            "MAX_LEVERAGE": 10.0,
        }
        c = Case(
            case_id=2,
            equity=2000.0,
            regime_up=1,
            btc_trend_slope=0.2,
            btc_vol_regime_level=0.2,
            p_cal=0.30,
            eth_macd_hist_4h=-0.3,
            risk_scale=1.7,
            entry=50.0,
            sl_initial=49.5,
        )
        lv = live_chain(cfg, c)
        bt = offline_reference_chain(cfg, c)
        self.assertAlmostEqual(float(lv["size_mult_final"]), float(bt["size_mult_final"]), places=12)
        self.assertAlmostEqual(float(lv["risk_usd"]), float(bt["risk_usd"]), places=12)
        self.assertAlmostEqual(float(lv["qty"]), float(bt["qty"]), places=12)


if __name__ == "__main__":
    unittest.main()
