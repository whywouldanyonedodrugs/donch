import unittest
from live.artifact_bundle import load_bundle

class TestFeatureManifestExtractsNames(unittest.TestCase):
    def test_bundle_has_feature_names(self):
        b = load_bundle("results/meta_export", strict=True)
        self.assertIsInstance(b.feature_names, list)
        self.assertGreater(len(b.feature_names), 0, "feature_names unexpectedly empty")

if __name__ == "__main__":
    unittest.main()
