import inspect
import unittest

import live.regime_features as rf


class TestDailyRegimeNoSmoothing(unittest.TestCase):
    def test_daily_regime_uses_filtered_probabilities(self) -> None:
        src = inspect.getsource(rf.compute_daily_regime_series)
        self.assertIn("filtered_marginal_probabilities", src)
        self.assertNotIn("smoothed_marginal_probabilities", src)
