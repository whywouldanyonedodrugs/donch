import os
import json
import unittest
import pandas as pd
import numpy as np


class TestRegimeSetsGolden(unittest.TestCase):
    def _load_thresholds(self) -> dict:
        report_path = os.environ.get(
            "DONCH_REGIMES_REPORT",
            "/root/apps/donch/results/meta_export/regimes_report.json",
        )
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Missing regimes_report.json at {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            j = json.load(f)

        # live code uses self.regime_thresholds dict directly; report stores thresholds under "thresholds"
        thr = dict(j.get("thresholds", {}) or {})
        # crowd thresholds may be at top-level in some exports; keep compatibility
        for k in ("crowd_z_high", "crowd_z_low", "btc_vol_hi"):
            if k in j and k not in thr:
                thr[k] = j[k]
        return thr

    def _load_golden(self) -> pd.DataFrame:
        p = os.environ.get(
            "DONCH_GOLDEN_FEATURES_PARQUET",
            "/root/apps/donch/results/meta_export/golden_features.parquet",
        )
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing golden_features.parquet at {p}")
        df = pd.read_parquet(p)
        return df

    def test_risk_on_matches_golden(self):
        # Import here so test fails loudly if package wiring is broken
        from live.live_trader import LiveTrader

        thr = self._load_thresholds()
        df = self._load_golden()

        # Required golden columns for this acceptance check
        required_cols = [
            "timestamp",
            "regime_up",
            "risk_on",
            "risk_on_1",
            "btcusdt_trend_slope",
            "btcusdt_vol_regime_level",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise AssertionError(f"golden_features.parquet missing required columns: {missing}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Deterministic sample: first/last 10 in-scope rows and first/last 10 out-of-scope rows
        in_scope = df[df["risk_on_1"].astype(float) == 1.0]
        out_scope = df[df["risk_on_1"].astype(float) == 0.0]

        sample = pd.concat(
            [
                in_scope.head(10),
                in_scope.tail(10),
                out_scope.head(10),
                out_scope.tail(10),
            ],
            axis=0,
        ).drop_duplicates(subset=["timestamp"])

        # Lightweight instance: bypass __init__
        trader = LiveTrader.__new__(LiveTrader)
        trader.regime_thresholds = thr

        # For each row: run live regime-set augment and compare risk_on/risk_on_1
        bad = []
        for _, r in sample.iterrows():
            meta_full = {
                "regime_up": float(r["regime_up"]),
                "btcusdt_trend_slope": float(r["btcusdt_trend_slope"]) if np.isfinite(r["btcusdt_trend_slope"]) else np.nan,
                "btcusdt_vol_regime_level": float(r["btcusdt_vol_regime_level"]) if np.isfinite(r["btcusdt_vol_regime_level"]) else np.nan,
            }

            # Optional S6 prereqs if present
            if "days_since_prev_break" in df.columns:
                meta_full["days_since_prev_break"] = float(r["days_since_prev_break"]) if np.isfinite(r["days_since_prev_break"]) else np.nan
            if "consolidation_range_atr" in df.columns:
                meta_full["consolidation_range_atr"] = float(r["consolidation_range_atr"]) if np.isfinite(r["consolidation_range_atr"]) else np.nan

            trader._augment_meta_with_regime_sets(meta_full)

            got_ro = int(float(meta_full.get("risk_on", 0.0) or 0.0))
            got_ro1 = int(float(meta_full.get("risk_on_1", 0.0) or 0.0))
            exp_ro = int(float(r["risk_on"]))
            exp_ro1 = int(float(r["risk_on_1"]))

            if (got_ro != exp_ro) or (got_ro1 != exp_ro1):
                bad.append((str(r["timestamp"]), exp_ro, got_ro, exp_ro1, got_ro1))

        if bad:
            msg = "\n".join([f"{ts} exp risk_on={e} got={g} exp risk_on_1={e1} got={g1}" for ts, e, g, e1, g1 in bad[:20]])
            raise AssertionError(f"risk_on parity mismatch for {len(bad)} rows (showing up to 20):\n{msg}")

    def test_s6_matches_golden_when_present(self):
        from live.live_trader import LiveTrader

        thr = self._load_thresholds()
        df = self._load_golden()

        if "S6_fresh_x_compress" not in df.columns:
            # Not all bundles export this; keep the test deterministic but conditional
            self.skipTest("golden_features.parquet has no S6_fresh_x_compress column")

        needed = ["timestamp", "days_since_prev_break", "consolidation_range_atr", "S6_fresh_x_compress"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise AssertionError(f"golden_features.parquet missing required columns for S6: {missing}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        sample = pd.concat([df.head(20), df.tail(20)], axis=0).drop_duplicates(subset=["timestamp"])

        trader = LiveTrader.__new__(LiveTrader)
        trader.regime_thresholds = thr

        bad = []
        for _, r in sample.iterrows():
            meta_full = {
                "days_since_prev_break": float(r["days_since_prev_break"]) if np.isfinite(r["days_since_prev_break"]) else np.nan,
                "consolidation_range_atr": float(r["consolidation_range_atr"]) if np.isfinite(r["consolidation_range_atr"]) else np.nan,
            }
            trader._augment_meta_with_regime_sets(meta_full)

            got = meta_full.get("S6_fresh_x_compress", np.nan)
            exp = r["S6_fresh_x_compress"]

            # Both may be NaN; treat NaN==NaN as pass here
            if (pd.isna(got) and pd.isna(exp)):
                continue
            if pd.isna(got) != pd.isna(exp):
                bad.append((str(r["timestamp"]), exp, got))
                continue

            if float(got) != float(exp):
                bad.append((str(r["timestamp"]), exp, got))

        if bad:
            msg = "\n".join([f"{ts} exp={e} got={g}" for ts, e, g in bad[:20]])
            raise AssertionError(f"S6 parity mismatch for {len(bad)} rows (showing up to 20):\n{msg}")


if __name__ == "__main__":
    unittest.main()
