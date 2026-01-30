import unittest
import numpy as np
import pandas as pd

from live.oi_funding import add_oi_funding_features
from live.regimes_report import RegimeThresholds
from live.oi_funding import StaleDerivativesDataError


class TestOiFundingUnit(unittest.TestCase):
    def test_codes_and_s3(self):
        idx = pd.date_range("2024-01-01", periods=288 * 8, freq="5min", tz="UTC")
        close = pd.Series(100.0 + np.sin(np.arange(len(idx)) / 50.0), index=idx)

        # OI increases once per day (piecewise constant). This should produce an almost-constant oi_pct_1d,
        # but floating arithmetic makes zscore slightly non-zero; test against the definition, not 0.
        oi = pd.Series(1000.0 * (1.01 ** (np.arange(len(idx)) // 288)), index=idx)

        fr = pd.Series(0.0, index=idx)

        df = pd.DataFrame({"close": close, "open_interest": oi, "funding_rate": fr}, index=idx)

        thr = RegimeThresholds(
            funding_neutral_eps=7.908e-05,
            oi_source="oi_z_7d",
            oi_q33=0.534149823784828,
            oi_q66=1.9565849494934076,
            btc_vol_hi=0.753777980804443,
        )

        out = add_oi_funding_features(df, thresholds=thr, staleness_max_age=pd.Timedelta(days=999))
        row = out.iloc[-1]

        # compute expected oi_z_7d by the same definition
        WIN_1D = 288
        WIN_7D = 7 * 288
        oi_pct_1d = oi.pct_change(WIN_1D)
        mu = oi_pct_1d.rolling(WIN_7D, min_periods=WIN_1D).mean()
        sd = oi_pct_1d.rolling(WIN_7D, min_periods=WIN_1D).std()
        expected_oi_z = (oi_pct_1d - mu) / (sd + 1e-12)

        self.assertTrue(np.isfinite(float(row["oi_z_7d"])))
        self.assertAlmostEqual(float(row["oi_z_7d"]), float(expected_oi_z.iloc[-1]), places=12)

        # funding==0 -> regime code should be 0 under eps
        self.assertEqual(int(row["funding_regime_code"]), 0)

        # oi_source=oi_z_7d with q33>0 => tiny oi_z_7d should bucket to -1
        self.assertEqual(int(row["oi_regime_code"]), -1)

        # S3 = (funding_regime_code + 1)*3 + (oi_regime_code + 1)
        self.assertEqual(int(row["S3_funding_x_oi"]), 3)

    def test_staleness_raises(self):
        idx = pd.date_range("2024-01-01", periods=288 * 2, freq="5min", tz="UTC")
        close = pd.Series(100.0, index=idx)
        oi = pd.Series(np.nan, index=idx)
        fr = pd.Series(np.nan, index=idx)

        # only one old datapoint at the very beginning
        oi.iloc[0] = 1000.0
        fr.iloc[0] = 0.0001

        df = pd.DataFrame({"close": close, "open_interest": oi, "funding_rate": fr}, index=idx)

        thr = RegimeThresholds(
            funding_neutral_eps=7.908e-05,
            oi_source="oi_z_7d",
            oi_q33=0.534149823784828,
            oi_q66=1.9565849494934076,
            btc_vol_hi=0.753777980804443,
        )

        with self.assertRaises(StaleDerivativesDataError):
            add_oi_funding_features(df, thresholds=thr, staleness_max_age=pd.Timedelta(minutes=30))


if __name__ == "__main__":
    unittest.main()
