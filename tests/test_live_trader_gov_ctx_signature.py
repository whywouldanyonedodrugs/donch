import unittest
from pathlib import Path


class TestLiveTraderGovCtxSignature(unittest.TestCase):
    def test_gov_ctx_calls_build_oi_with_decision_ts(self) -> None:
        p = Path("live/live_trader.py")
        s = p.read_text(encoding="utf-8")

        self.assertIn(
            'btc_feats = await self._build_oi_funding_features("BTCUSDT", df_btc, decision_ts_gov)',
            s,
        )
        self.assertIn(
            'eth_feats = await self._build_oi_funding_features("ETHUSDT", df_eth, decision_ts_gov)',
            s,
        )


if __name__ == "__main__":
    unittest.main()
