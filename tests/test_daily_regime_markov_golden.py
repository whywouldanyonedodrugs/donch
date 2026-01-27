import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

def _pick_col_by_suffix(df, suffixes):
    cols = list(df.columns)
    for suf in suffixes:
        cands = [c for c in cols if c.endswith(suf)]
        if cands:
            return sorted(cands, key=len)[0]
    return None

class TestDailyRegimeAndMarkovGolden(unittest.TestCase):
    def _golden_path(self) -> Path:
        return Path(
            os.environ.get(
                "DONCH_GOLDEN_FEATURES_PARQUET",
                "/root/apps/donch/results/meta_export/golden_features.parquet",
            )
        )

    def _fixtures_dir(self) -> Path:
        return Path(
            os.environ.get(
                "DONCH_REGIME_FIXTURES_DIR",
                "/root/apps/donch/tests/fixtures/regime",
            )
        )

    @staticmethod
    def _to_utc_ts(s: pd.Series) -> pd.Series:
        out = pd.to_datetime(s, utc=True, errors="coerce")
        return out

    @staticmethod
    def _pick_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
        for c in aliases:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _tf_timedelta(tf: str) -> pd.Timedelta:
        t = str(tf).strip().lower()
        if t in ("1d", "d", "1day", "day"):
            return pd.Timedelta(days=1)
        if t in ("4h", "h4", "4hour", "4hours"):
            return pd.Timedelta(hours=4)
        raise ValueError(f"Unsupported tf: {tf}")

    def _load_fixture_ohlcv_as_open_ts(self, symbol: str, tf: str) -> pd.DataFrame:
        """
        Offline fixtures are right/right with timestamp at BAR CLOSE.
        Regime snapshot functions expect CCXT-style BAR OPEN timestamps.
        Convert by shifting index back by one TF.
        """
        d = self._fixtures_dir()
        sym = str(symbol).upper()

        tf_tag = str(tf).strip().upper()
        if tf_tag == "4H":
            tf_tag = "4H"
        if tf_tag == "1D":
            tf_tag = "1D"

        p = d / f"{sym}_{tf_tag}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing fixture parquet: {p}")

        df = pd.read_parquet(p)
        if "timestamp" not in df.columns:
            raise ValueError(f"{p} must have explicit 'timestamp' column")

        ts_close = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.assign(timestamp=ts_close).dropna(subset=["timestamp"]).sort_values("timestamp")

        # Ensure required OHLCV columns exist
        need = {"open", "high", "low", "close"}
        missing = sorted(list(need - set(df.columns)))
        if missing:
            raise ValueError(f"{p} missing required OHLCV columns: {missing}")

        # Set index to CLOSE timestamp then shift back to OPEN timestamp.
        df = df.set_index("timestamp", drop=True)
        df.index = pd.to_datetime(df.index, utc=True)

        td = self._tf_timedelta(tf)
        df.index = df.index - td
        df.index.name = "timestamp"

        # Keep only expected columns
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()

        # Numeric coercion
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    def _load_golden(self) -> pd.DataFrame:
        p = self._golden_path()
        if not p.exists():
            raise FileNotFoundError(f"Missing golden parquet: {p}")
        df = pd.read_parquet(p)
        if "timestamp" not in df.columns:
            raise ValueError("golden_features.parquet missing 'timestamp' column")
        df = df.copy()
        df["timestamp"] = self._to_utc_ts(df["timestamp"])
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        return df

    @staticmethod
    def _select_timestamps(df: pd.DataFrame, n: int = 10) -> list[pd.Timestamp]:
        # Deterministic spread across the range.
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
        if len(ts) == 0:
            return []
        if len(ts) <= n:
            return list(ts)
        idx = np.linspace(0, len(ts) - 1, n).round().astype(int)
        return list(ts.iloc[idx].tolist())

    def test_daily_regime_matches_golden_when_present(self):
        from live.regime_features import compute_daily_regime_snapshot
        import config as cfg

        df = self._load_golden()

        # Golden column aliases (export formats vary)
        col_code = _pick_col_by_suffix(g, ["regime_code_1d"])
        col_volp = _pick_col_by_suffix(g, ["vol_prob_low_1d"])
        col_trend = _pick_col_by_suffix(g, ["trend_regime_1d"])
        col_volreg = _pick_col_by_suffix(g, ["vol_regime_1d"])
        col_regime = _pick_col_by_suffix(g, ["daily_regime_1d", "regime_1d"])  # optional, only if you store string regime


        if all(x is None for x in [col_code, col_volp, col_trend, col_volreg, col_regime]):
            self.skipTest("golden has no daily regime columns (skipping)")


        bench = str(getattr(cfg, "REGIME_BENCHMARK_SYMBOL", "BTCUSDT")).upper()
        df_daily = self._load_fixture_ohlcv_as_open_ts(bench, "1D")

        ma_period = int(getattr(cfg, "REGIME_MA_PERIOD", 100))
        atr_period = int(getattr(cfg, "REGIME_ATR_PERIOD", 20))
        atr_mult = float(getattr(cfg, "REGIME_ATR_MULT", 2.0))

        # Build per-timestamp expected values (use first row per timestamp)
        g = df.groupby("timestamp", as_index=True).first()

        sample_ts = self._select_timestamps(df, n=10)
        if not sample_ts:
            raise AssertionError("No timestamps available for daily regime test")

        bad = []
        tested = 0

        for ts in sample_ts:
            exp = g.loc[ts] if ts in g.index else None
            if exp is None:
                continue

            try:
                snap = compute_daily_regime_snapshot(
                    df_daily,
                    asof_ts=ts,
                    ma_period=ma_period,
                    atr_period=atr_period,
                    atr_mult=atr_mult,
                )
            except Exception as e:
                bad.append((str(ts), f"compute_failed:{type(e).__name__}:{e}", None))
                continue

            tested += 1

            # Compare each golden column that exists
            if col_code is not None and pd.notna(exp.get(col_code, np.nan)):
                if int(snap["regime_code_1d"]) != int(exp[col_code]):
                    bad.append((str(ts), f"regime_code exp={int(exp[col_code])}", f"got={int(snap['regime_code_1d'])}"))

            if col_str is not None and pd.notna(exp.get(col_str, None)):
                if str(snap["daily_regime_str_1d"]) != str(exp[col_str]):
                    bad.append((str(ts), f"regime_str exp={exp[col_str]}", f"got={snap['daily_regime_str_1d']}"))

            if col_tr is not None and pd.notna(exp.get(col_tr, None)):
                if str(snap["trend_regime_1d"]) != str(exp[col_tr]):
                    bad.append((str(ts), f"trend exp={exp[col_tr]}", f"got={snap['trend_regime_1d']}"))

            if col_vr is not None and pd.notna(exp.get(col_vr, None)):
                if str(snap["vol_regime_1d"]) != str(exp[col_vr]):
                    bad.append((str(ts), f"vol_reg exp={exp[col_vr]}", f"got={snap['vol_regime_1d']}"))

            if col_vpl is not None and pd.notna(exp.get(col_vpl, np.nan)):
                # Markov fits can differ slightly across environments; tolerate small numeric drift.
                tol = float(os.environ.get("DONCH_VOL_PROB_TOL", "1e-3"))
                if not np.isfinite(float(exp[col_vpl])):
                    continue
                if abs(float(snap["vol_prob_low_1d"]) - float(exp[col_vpl])) > tol:
                    bad.append(
                        (str(ts), f"vol_prob exp={float(exp[col_vpl]):.6f}", f"got={float(snap['vol_prob_low_1d']):.6f} tol={tol}")
                    )

        if tested == 0:
            if bad:
                raise AssertionError(
                    "Daily regime test could not compute any timestamps. "
                    f"First failures:\n{msg}"
                )
            raise AssertionError("Daily regime test did not evaluate any timestamps (no comparable rows)")

        if bad:
            msg = "\n".join([f"{ts} {a} {b}" for ts, a, b in bad[:25]])
            raise AssertionError(f"Daily regime parity mismatch for {len(bad)} cases (showing up to 25):\n{msg}")

    def test_markov4h_matches_golden_when_present(self):
        from live.regime_features import compute_markov4h_snapshot

        df = self._load_golden()

        col_prob = _pick_col_by_suffix(g, ["markov_prob_up_4h", "markov_prob_4h"])
        col_state = _pick_col_by_suffix(g, ["markov_state_up_4h", "markov_state_4h"])
        if col_prob is None and col_state is None:
            self.skipTest("golden has no Markov(4h) columns (skipping)")


        markov_asset = str(os.environ.get("DONCH_MARKOV4H_ASSET", "ETHUSDT")).upper()
        alpha = float(os.environ.get("DONCH_MARKOV4H_ALPHA", "0.2"))

        df4h = self._load_fixture_ohlcv_as_open_ts(markov_asset, "4H")

        g = df.groupby("timestamp", as_index=True).first()
        sample_ts = self._select_timestamps(df, n=10)
        if not sample_ts:
            raise AssertionError("No timestamps available for Markov test")

        bad = []
        tested = 0

        for ts in sample_ts:
            exp = g.loc[ts] if ts in g.index else None
            if exp is None:
                continue

            try:
                snap = compute_markov4h_snapshot(df4h, asof_ts=ts, alpha=alpha)
            except Exception as e:
                bad.append((str(ts), f"compute_failed:{type(e).__name__}:{e}", None))
                continue

            tested += 1

            if col_prob is not None and pd.notna(exp.get(col_prob, np.nan)):
                tol = float(os.environ.get("DONCH_MARKOV_PROB_TOL", "5e-3"))
                if not np.isfinite(float(exp[col_prob])):
                    continue
                if abs(float(snap["markov_prob_up_4h"]) - float(exp[col_prob])) > tol:
                    bad.append(
                        (str(ts), f"prob exp={float(exp[col_prob]):.6f}", f"got={float(snap['markov_prob_up_4h']):.6f} tol={tol}")
                    )

            if col_state is not None and pd.notna(exp.get(col_state, np.nan)):
                if int(snap["markov_state_4h"]) != int(exp[col_state]):
                    bad.append((str(ts), f"state exp={int(exp[col_state])}", f"got={int(snap['markov_state_4h'])}"))

        if tested == 0:
            raise AssertionError("Markov test did not evaluate any timestamps (no comparable rows)")

        if bad:
            msg = "\n".join([f"{ts} {a} {b}" for ts, a, b in bad[:25]])
            raise AssertionError(f"Markov 4h parity mismatch for {len(bad)} cases (showing up to 25):\n{msg}")


if __name__ == "__main__":
    unittest.main()
