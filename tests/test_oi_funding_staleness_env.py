import os
import unittest

import pandas as pd

from live.oi_funding import compute_oi_funding_features, StaleDerivativesDataError
from live.regimes_report import RegimeThresholds


class TestOiFundingStalenessEnv(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop('DONCH_OI_MAX_AGE_MIN', None)
        os.environ.pop('DONCH_FUNDING_MAX_AGE_MIN', None)

    def _make_inputs(self):
        decision_ts = pd.Timestamp('2026-01-31 10:30:00', tz='UTC')
        idx = pd.date_range(end=decision_ts, periods=50, freq='5min', tz='UTC')
        df5 = pd.DataFrame(
            {
                'open': 1.0,
                'high': 1.0,
                'low': 1.0,
                'close': 1.0,
                'volume': 100.0,
            },
            index=idx,
        )

        # OI is fresh (5m before decision)
        oi_idx = pd.DatetimeIndex([decision_ts - pd.Timedelta(minutes=5)], tz='UTC')
        oi5 = pd.Series([1000.0], index=oi_idx, name='open_interest')

        # Funding last print 08:00 (2.5h stale vs decision)
        fr_idx = pd.DatetimeIndex([pd.Timestamp('2026-01-31 08:00:00', tz='UTC')], tz='UTC')
        fr5 = pd.Series([0.0001], index=fr_idx, name='funding_rate')

        thresholds = RegimeThresholds(
            funding_neutral_eps=1e-6,
            oi_source='oi_z_7d',
            oi_q33=-0.5,
            oi_q66=0.5,
            btc_vol_hi=0.05,
        )
        return df5, oi5, fr5, thresholds

    def test_funding_staleness_raises_without_env_override(self) -> None:
        df5, oi5, fr5, thresholds = self._make_inputs()
        with self.assertRaises(StaleDerivativesDataError):
            compute_oi_funding_features(
                df5=df5,
                oi5=oi5,
                fr5=fr5,
                thresholds=thresholds,
                allow_nans=True,
                staleness_max_age=pd.Timedelta(minutes=30),
            )

    def test_funding_staleness_can_be_relaxed_via_env(self) -> None:
        df5, oi5, fr5, thresholds = self._make_inputs()
        os.environ['DONCH_FUNDING_MAX_AGE_MIN'] = '600'  # 10h
        feats = compute_oi_funding_features(
            df5=df5,
            oi5=oi5,
            fr5=fr5,
            thresholds=thresholds,
            allow_nans=True,
            staleness_max_age=pd.Timedelta(minutes=30),  # OI still effectively 30m
        )
        self.assertIn('funding_rate', feats)


if __name__ == '__main__':
    unittest.main()
