# tests/test_meta_scope.py
import unittest
import numpy as np
from live.parity_utils import eval_meta_scope

class TestMetaScope(unittest.TestCase):

    def test_risk_on_1_basic(self):
        self.assertTrue(eval_meta_scope("RISK_ON_1", {"risk_on_1": 1, "risk_on": 0})[0])
        self.assertFalse(eval_meta_scope("RISK_ON_1", {"risk_on_1": 0, "risk_on": 1})[0])

    def test_risk_on_1_floats(self):
        self.assertTrue(eval_meta_scope("RISK_ON_1", {"risk_on_1": 1.0})[0])
        self.assertFalse(eval_meta_scope("RISK_ON_1", {"risk_on_1": 0.0})[0])

    def test_risk_on_1_strings(self):
        self.assertTrue(eval_meta_scope("RISK_ON_1", {"risk_on_1": "1"})[0])
        self.assertFalse(eval_meta_scope("RISK_ON_1", {"risk_on_1": "0"})[0])
        self.assertTrue(eval_meta_scope("RISK_ON_1", {"risk_on_1": "1.0"})[0])
        self.assertFalse(eval_meta_scope("RISK_ON_1", {"risk_on_1": "0.0"})[0])

    def test_alias_fallback_to_risk_on(self):
        # Missing risk_on_1 but risk_on=1 should pass
        self.assertTrue(eval_meta_scope("RISK_ON_1", {"risk_on": 1})[0])

        # risk_on_1 present but NaN -> fallback to risk_on
        self.assertTrue(eval_meta_scope("RISK_ON_1", {"risk_on_1": np.nan, "risk_on": 1})[0])
        self.assertFalse(eval_meta_scope("RISK_ON_1", {"risk_on_1": np.nan, "risk_on": 0})[0])

    def test_none_nan_fail_closed(self):
        self.assertFalse(eval_meta_scope("RISK_ON_1", {"risk_on_1": None, "risk_on": None})[0])
        self.assertFalse(eval_meta_scope("RISK_ON_1", {"risk_on_1": np.nan})[0])
        self.assertFalse(eval_meta_scope("RISK_ON_1", {"risk_on": np.nan})[0])

    def test_unknown_scope_fails_closed(self):
        self.assertFalse(eval_meta_scope("SOME_UNKNOWN_SCOPE", {"risk_on_1": 1, "risk_on": 1})[0])

    def test_empty_scope_passes(self):
        self.assertTrue(eval_meta_scope(None, {"risk_on_1": 0})[0])
        self.assertTrue(eval_meta_scope("", {"risk_on_1": 0})[0])

if __name__ == "__main__":
    unittest.main()
