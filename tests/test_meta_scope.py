import math
import unittest
import numpy as np

from live.parity_utils import eval_meta_scope


class TestMetaScope(unittest.TestCase):
    def test_empty_scope_passes(self):
        ok, info = eval_meta_scope("", {"risk_on_1": 0})
        self.assertTrue(ok)
        self.assertIsNone(info.get("scope"))

        ok, info = eval_meta_scope(None, {"risk_on_1": 0})
        self.assertTrue(ok)
        self.assertIsNone(info.get("scope"))

    def test_unknown_scope_fails_closed(self):
        ok, info = eval_meta_scope("SOMETHING_ELSE", {"risk_on_1": 1})
        self.assertFalse(ok)
        self.assertEqual(info.get("scope"), "SOMETHING_ELSE")

    def test_risk_on_1_basic(self):
        ok, _ = eval_meta_scope("RISK_ON_1", {"risk_on_1": 1})
        self.assertTrue(ok)
        ok, _ = eval_meta_scope("RISK_ON_1", {"risk_on_1": 0})
        self.assertFalse(ok)
        ok, _ = eval_meta_scope("RISK_ON_1", {"risk_on_1": 2})
        self.assertFalse(ok)

    def test_risk_on_1_floats(self):
        ok, _ = eval_meta_scope("RISK_ON_1", {"risk_on_1": 1.0})
        self.assertTrue(ok)
        ok, _ = eval_meta_scope("RISK_ON_1", {"risk_on_1": 0.0})
        self.assertFalse(ok)

    def test_risk_on_1_strings(self):
        ok, _ = eval_meta_scope("RISK_ON_1", {"risk_on_1": "1"})
        self.assertTrue(ok)
        ok, _ = eval_meta_scope("RISK_ON_1", {"risk_on_1": "0"})
        self.assertFalse(ok)
        ok, _ = eval_meta_scope("RISK_ON_1", {"risk_on_1": "true"})
        self.assertFalse(ok)

    def test_none_nan_fail_closed(self):
        ok, _ = eval_meta_scope("RISK_ON_1", {"risk_on_1": None})
        self.assertFalse(ok)
        ok, _ = eval_meta_scope("RISK_ON_1", {"risk_on_1": float("nan")})
        self.assertFalse(ok)
        ok, _ = eval_meta_scope("RISK_ON_1", {"risk_on_1": np.nan})
        self.assertFalse(ok)

    def test_alias_fallback_to_risk_on_only_if_missing_key(self):
        # risk_on_1 key ABSENT -> fallback to risk_on
        ok, info = eval_meta_scope("RISK_ON_1", {"risk_on": 1})
        self.assertTrue(ok)
        self.assertEqual(info.get("scope_src"), "risk_on")

        ok, info = eval_meta_scope("RISK_ON_1", {"risk_on": 0})
        self.assertFalse(ok)
        self.assertEqual(info.get("scope_src"), "risk_on")

    def test_no_rowwise_fallback_when_risk_on_1_present_but_nan(self):
        # risk_on_1 key PRESENT but NaN -> treated as 0, do NOT fallback to risk_on
        ok, info = eval_meta_scope("RISK_ON_1", {"risk_on_1": np.nan, "risk_on": 1})
        self.assertFalse(ok)
        self.assertEqual(info.get("scope_src"), "risk_on_1")
        self.assertEqual(info.get("scope_val"), 0.0)

    def test_missing_both_cols_flags_error(self):
        ok, info = eval_meta_scope("RISK_ON_1", {"something_else": 1})
        self.assertFalse(ok)
        self.assertTrue(bool(info.get("missing_cols", False)))
        self.assertIsNone(info.get("scope_src"))
