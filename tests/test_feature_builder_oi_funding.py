import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from live.feature_builder import FeatureBuilder


class TestFeatureBuilderOiFunding(unittest.TestCase):
    def _make_df(self, days: int = 8) -> pd.DataFrame:
        n = days * 288
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        close = pd.Series(100.0 + np.linspace(0, 1, n), index=idx)

        day = (np.arange(n) // 288)
        oi = 1000.0 * (1.0 + 0.01) ** day
        oi = pd.Series(oi, index=idx)

        fr = pd.Series(0.0, index=idx)

        return pd.DataFrame({"close": close, "open_interest": oi, "funding_rate": fr, "high": close, "low": close, "open": close, "volume": 1.0}, index=idx)

    def test_builder_adds_oi_funding_keys(self) -> None:
        df = self._make_df(days=8)
        decision_ts = df.index[-1]

        with tempfile.TemporaryDirectory() as td:
            meta_dir = Path(td)
            # minimal regimes_report.json for thresholds
            regimes_report = {
                "thresholds": {
                    "funding_neutral_eps": 1e-4,
                    "oi_source": "oi_z_7d",
                    "oi_q33": 0.5,
                    "oi_q66": 2.0,
                    "btc_vol_hi": 1.0,
                }
            }
            (meta_dir / "regimes_report.json").write_text(json.dumps(regimes_report), encoding="utf-8")

            fb = FeatureBuilder({"DERIV_MAX_AGE": "999d"})
            feats = fb.compute_features_for_decision(df, decision_ts, meta_dir=meta_dir)

            for k in ["funding_rate", "funding_regime_code", "oi_level", "oi_regime_code", "S3_funding_x_oi", "est_leverage"]:
                self.assertIn(k, feats)


if __name__ == "__main__":
    unittest.main()
