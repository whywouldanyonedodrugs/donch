import unittest
import numpy as np
import pandas as pd

from live.oi_funding import (
    WIN_1D,
    WIN_7D,
    add_oi_funding_features,
    oi_funding_features_at_decision,
    StaleDerivativesDataError,
)
from live.regimes_report import RegimeThresholds


class TestOiFundingUnit(unittest.TestCase):
    def _make_df(self, days: int = 8) -> pd.DataFrame:
        # 5m grid, UTC
        n = days * 288
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")

        # Close: simple monotonic (avoid zeros)
        close = pd.Series(100.0 + np.linspace(0, 1, n), index=idx)

        # Open interest: stepwise daily growth => constant oi_pct_1d after day 1
        r = 0.01  # 1% daily
        day = (np.arange(n) // 288)
        oi = 1000.0 * (1.0 + r) ** day
        oi = pd.Series(oi, index=idx)

        # Funding: constant 0 => funding_regime_code should be 0 (if eps > 0)
        fr = pd.Series(0.0, index=idx)

        return pd.DataFrame({"close": close, "open_interest": oi, "funding_rate": fr}, index=idx)

    def test_codes_and_s3(self) -> None:
        df = self._make_df(days=8)
        decision_ts = df.index[-1]

        thr = RegimeThresholds(
            funding_neutral_eps=7.908e-05,  # example
            oi_source="oi_z_7d",
            oi_q33=0.534149823784828,
            oi_q66=1.9565849494934076,
            btc_vol_hi=0.753777980804443,
        )

        out = add_oi_funding_features(df, thresholds=thr, decision_ts=decision_ts, staleness_max_age=pd.Timedelta("999d"))
        row = out.iloc[-1]

        # funding_rate=0 => in (-eps, eps) => 0
        self.assertEqual(int(row["funding_regime_code"]), 0)

        # oi_pct_1d becomes constant => oi_z_7d ~ 0 (std~0)
        self.assertTrue(abs(float(row["oi_z_7d"])) < 1e-6)

        # q33 is positive => oi_z_7d <= q33 => -1
        self.assertEqual(int(row["oi_regime_code"]), -1)

        # S3 = (0+1)*3 + (-1+1) = 3
        self.assertEqual(int(row["S3_funding_x_oi"]), 3)

    def test_staleness_raises(self) -> None:
        df = self._make_df(days=2)
        decision_ts = df.index[-1]

        thr = RegimeThresholds(
            funding_neutral_eps=1e-4,
            oi_source="oi_pct_1d",
            oi_q33=-1.0,
            oi_q66=1.0,
            btc_vol_hi=1.0,
        )

        # Make raw series stale: set last raw values far in the past and NaN afterwards
        df2 = df.copy()
        df2.loc[df2.index[-200:], "open_interest"] = np.nan
        df2.loc[df2.index[-200:], "funding_rate"] = np.nan

        with self.assertRaises(StaleDerivativesDataError):
            add_oi_funding_features(
                df2,
                thresholds=thr,
                decision_ts=decision_ts,
                staleness_max_age=pd.Timedelta("1h"),
            )


if __name__ == "__main__":
    unittest.main()
