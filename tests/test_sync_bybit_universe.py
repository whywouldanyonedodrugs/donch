import unittest

from tools.sync_bybit_perps_universe import extract_linear_usdt_perp_ids


class TestSyncBybitUniverse(unittest.TestCase):
    def test_extract_filters_linear_usdt_swap_only(self) -> None:
        markets = {
            "BTC/USDT:USDT": {
                "id": "BTCUSDT",
                "swap": True,
                "linear": True,
                "contract": True,
                "active": True,
                "quote": "USDT",
                "settle": "USDT",
            },
            "ETH/USDC:USDC": {
                "id": "ETHUSDC",
                "swap": True,
                "linear": True,
                "contract": True,
                "active": True,
                "quote": "USDC",
                "settle": "USDC",
            },
            "XRP/USDT:USDT": {
                "id": "XRPUSDT",
                "swap": True,
                "linear": False,
                "contract": True,
                "active": True,
                "quote": "USDT",
                "settle": "USDT",
            },
            "DOGE/USDT:USDT": {
                "id": "DOGEUSDT",
                "swap": False,
                "linear": True,
                "contract": True,
                "active": True,
                "quote": "USDT",
                "settle": "USDT",
            },
            "OLD/USDT:USDT": {
                "id": "OLDUSDT",
                "swap": True,
                "linear": True,
                "contract": True,
                "active": False,
                "quote": "USDT",
                "settle": "USDT",
            },
        }

        out = extract_linear_usdt_perp_ids(markets)
        self.assertEqual(out, ["BTCUSDT"])

    def test_extract_can_derive_symbol_id(self) -> None:
        markets = {
            "PEPE/USDT:USDT": {
                "id": "",
                "base": "PEPE",
                "quote": "USDT",
                "swap": True,
                "linear": True,
                "contract": True,
                "active": True,
                "settle": "USDT",
            }
        }
        out = extract_linear_usdt_perp_ids(markets)
        self.assertEqual(out, ["PEPEUSDT"])


if __name__ == "__main__":
    unittest.main()
