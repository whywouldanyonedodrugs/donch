import unittest
from pathlib import Path

from live.regimes_report import load_regime_thresholds


class TestRegimesReport(unittest.TestCase):
    def test_thresholds_present(self) -> None:
        meta_dir = Path("results/meta_export")
        thr = load_regime_thresholds(meta_dir)

        self.assertGreater(thr.funding_neutral_eps, 0.0)
        self.assertIn(thr.oi_source, ("oi_z_7d", "oi_pct_1d"))
        self.assertLessEqual(thr.oi_q33, thr.oi_q66)
        self.assertGreater(thr.btc_vol_hi, 0.0)


if __name__ == "__main__":
    unittest.main()
