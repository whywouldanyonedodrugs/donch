from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from artifact_bundle import load_bundle
from live.regime_truth import macro_regimes_asof


class TestDailyRegimeAndMarkovGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cls.meta_dir = repo_root / "results" / "meta_export"
        cls.golden_path = cls.meta_dir / "golden_features.parquet"
        if not cls.golden_path.exists():
            raise RuntimeError(f"Missing golden_features.parquet at {cls.golden_path}")

        cls.bundle = load_bundle(cls.meta_dir, strict=True)

        df = pd.read_parquet(cls.golden_path)
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True)
        else:
            ts = pd.to_datetime(df.index, utc=True)

        df = df.copy()
        df["timestamp"] = ts

        # reduce to one row per timestamp (macro columns are constant across symbols per timestamp)
        use = ["timestamp", "regime_code_1d", "vol_prob_low_1d", "markov_state_4h", "markov_prob_up_4h"]
        missing = [c for c in use if c not in df.columns]
        if missing:
            raise RuntimeError(f"golden missing columns: {missing}")

        df = df[use].sort_values(["timestamp"]).groupby("timestamp", as_index=False).first()
        cls.golden_macro = df

    def test_macro_regimes_match_golden_asof(self) -> None:
        g = self.golden_macro

        # Keep runtime bounded; golden contains many rows. This is an integration parity check.
        # Deterministic selection: take first N rows.
        N = 2000
        g = g.iloc[: min(N, len(g))].copy()

        mismatches = []
        prob_tol = 5e-3  # safe tol; should usually be far smaller
        for _, row in g.iterrows():
            ts = row["timestamp"]
            exp_code = int(row["regime_code_1d"])
            exp_vol = float(row["vol_prob_low_1d"])
            exp_state = int(row["markov_state_4h"])
            exp_prob = float(row["markov_prob_up_4h"])

            got = macro_regimes_asof(self.bundle, ts)

            if got["regime_code_1d"] != exp_code:
                mismatches.append(f"{ts} regime_code_1d exp={exp_code} got={got['regime_code_1d']}")
            if abs(got["vol_prob_low_1d"] - exp_vol) > prob_tol:
                mismatches.append(f"{ts} vol_prob_low_1d exp={exp_vol:.6f} got={got['vol_prob_low_1d']:.6f} tol={prob_tol}")
            if got["markov_state_4h"] != exp_state:
                mismatches.append(f"{ts} markov_state_4h exp={exp_state} got={got['markov_state_4h']}")
            if abs(got["markov_prob_up_4h"] - exp_prob) > prob_tol:
                mismatches.append(f"{ts} markov_prob_up_4h exp={exp_prob:.6f} got={got['markov_prob_up_4h']:.6f} tol={prob_tol}")

            if len(mismatches) >= 50:
                break

        if mismatches:
            raise AssertionError("Macro regime parity mismatches (showing up to 50):\n" + "\n".join(mismatches))


if __name__ == "__main__":
    unittest.main()
