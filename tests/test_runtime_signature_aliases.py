import inspect
import unittest


class TestRuntimeSignatureAliases(unittest.TestCase):
    def test_regime_features_accept_asof_ts(self) -> None:
        from live.regime_features import (
            compute_daily_regime_snapshot,
            compute_markov4h_snapshot,
            drop_incomplete_last_bar,
        )

        sig_daily = inspect.signature(compute_daily_regime_snapshot)
        sig_markov = inspect.signature(compute_markov4h_snapshot)
        sig_drop = inspect.signature(drop_incomplete_last_bar)

        self.assertIn("asof_ts", sig_daily.parameters)
        self.assertIn("asof_ts", sig_markov.parameters)
        self.assertIn("asof_ts", sig_drop.parameters)

    def test_exchange_proxy_accepts_end_ts_and_asof_ts(self) -> None:
        from live.exchange_proxy import ExchangeProxy

        sig_fr = inspect.signature(ExchangeProxy.fetch_funding_rate_history)
        self.assertIn("end_ts", sig_fr.parameters)
        self.assertIn("asof_ts", sig_fr.parameters)


if __name__ == "__main__":
    unittest.main()
