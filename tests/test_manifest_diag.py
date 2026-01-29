import tempfile
import unittest
from pathlib import Path

from live.manifest_diag import (
    is_cross_asset_feature,
    is_derivatives_feature,
    scan_python_files_for_feature_refs,
)


class TestManifestDiag(unittest.TestCase):
    def test_classification(self) -> None:
        self.assertTrue(is_derivatives_feature("funding_rate_1h"))
        self.assertTrue(is_derivatives_feature("oi_level_4h"))
        self.assertTrue(is_derivatives_feature("open_interest_chg_1d"))
        self.assertFalse(is_derivatives_feature("rsi_1h"))

        self.assertTrue(is_cross_asset_feature("btc_ret_1h"))
        self.assertTrue(is_cross_asset_feature("eth_vol_1d"))
        self.assertTrue(is_cross_asset_feature("foo_btc_bar"))
        self.assertFalse(is_cross_asset_feature("volume_1h"))

    def test_scan_counts_token_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "live").mkdir(parents=True, exist_ok=True)

            py = root / "live" / "x.py"
            py.write_text(
                """
FEATURES = ["funding_rate_1h", "btc_ret_1h"]
d = {"funding_rate_1h": 1.0}
# should not count substring inside longer token:
s = "xxfunding_rate_1hyy"
""",
                encoding="utf-8",
            )

            feats = ["funding_rate_1h", "btc_ret_1h", "oi_level_4h"]
            counts = scan_python_files_for_feature_refs(root, feats, include_dirs=("live",))

            # token-boundary matches:
            # funding_rate_1h appears twice as token: in list + dict key
            self.assertEqual(counts["funding_rate_1h"], 2)
            self.assertEqual(counts["btc_ret_1h"], 1)
            self.assertEqual(counts["oi_level_4h"], 0)


if __name__ == "__main__":
    unittest.main()
