import json
import unittest
from pathlib import Path

from live.artifact_bundle import load_bundle


class TestFeatureManifestExtractsNames(unittest.TestCase):
    def test_bundle_has_feature_names(self) -> None:
        meta_dir = Path("results/meta_export")
        b = load_bundle(meta_dir, strict=True)

        self.assertIsInstance(b.feature_names, list)
        self.assertGreater(len(b.feature_names), 0)

        m = json.loads((meta_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        feats = m.get("features") or {}
        numeric = feats.get("numeric_cols") or []
        cat = feats.get("cat_cols") or []

        # Ensure manifest schema is as expected in this export
        self.assertIsInstance(numeric, list)
        self.assertIsInstance(cat, list)

        expected = list(numeric) + list(cat)
        self.assertEqual(b.feature_names, expected)
