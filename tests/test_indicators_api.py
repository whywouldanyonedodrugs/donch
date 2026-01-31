import unittest

class TestIndicatorsAPI(unittest.TestCase):
    def test_sma_exists(self):
        from live import indicators as ta
        self.assertTrue(hasattr(ta, "sma"))
