import os
import unittest
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def _pick_col_by_suffix(df: pd.DataFrame, suffixes: Sequence[str]) -> Optional[str]:
    cols = list(df.columns)
    for suf in suffixes:
        cands = [c for c in cols if str(c).endswith(str(suf))]
        if cands:
            # shortest tends to select non-prefixed before S1_* etc, but still matches if only prefixed exists
            return sorted(cands, key=len)[0]
    return None


def _evenly_spaced(ts: Iterable[pd.Timestamp], n: int) -> List[pd.Timestamp]:
    xs = pd.to_datetime(list(ts), utc=True, errors="coerce")
    xs = [x for x in xs if pd.notna(x)]
    xs = sorted(set(xs))
    if not xs:
        return []
    if len(xs) <= n:
        return xs
    idx = np.linspace(0, len(xs) - 1, n).round().astype(int)
    out = []
    seen = set()
    for i in idx:
        t = xs[int(i)]
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


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

    def _fixture_index_style(self) -> str:
        # "close" = fixture timestamps are bar CLOSE timestamps (right/right). (Recommended.)
        # "open"  = shift fixture timestamps backward by one TF to bar OPEN timestamps.
        return str(os.environ.get("DONCH_FIXTURE_INDEX_STYLE", "close")).strip().lower()

    def _load_golden(self) -> pd.DataFrame:
        p = self._golden_path()
        if not p.exists():
            raise FileNotFoundError(f"Missing golden parquet: {p}")
        df = pd.read_parquet(p)
        if "timestamp" not in df.columns:
            raise ValueError("golden_features.parquet missing 'timestamp' column")
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        return df

    def _load_fixture_ohlcv(self, symbol: str, tf_tag: str) -> pd.DataFrame:
        """
        Fixtures are exported right/right with timestamp at BAR CLOSE (UTC) and an explicit 'timestamp' column.

        By default we keep index as bar-close timestamps ("close").
        If DONCH_FIXTURE_INDEX_STYLE=open, we shift index back by 1 TF to get bar-open timestamps.
        """
        d = self._fixtures_dir()
        p = d / f"{symbol.upper()}_{tf_tag.upper()}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing fixture parquet: {p}")

        df = pd.read_parquet(p)
        if "timestamp" not in df.columns:
            raise ValueError(f"{p} must have explicit 'timestamp' column")

        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp", drop=True)

        need = {"open", "high", "low", "close", "volume"}
        missing = sorted(list(need - set(df.columns)))
        if missing:
            raise ValueError(f"{p} missing required OHLCV columns: {missing}")

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"]).sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if self._fixture_index_style() == "open":
            td = pd.Timedelta(tf_tag.lower())
            df = df.copy()
            df.index = df.index - td
            df.index.name = "timestamp"

        return df

    def test_daily_regime_matches_golden_when_present(self):
        from live.regime_features import compute_daily_regime_snapshot
        import config as cfg

        df = self._load_golden()

        # robust column picks
        col_code = _pick_col_by_suffix(df, ["regime_code_1d"])
        if col_code is None:
            self.skipTest("golden has no regime_code_1d-like column (skipping)")

        col_volp = _pick_col_by_suffix(df, ["vol_prob_low_1d", "vol_prob_1d"])
        col_trend = _pick_col_by_suffix(df, ["trend_regime_1d"])
        col_volreg = _pick_col_by_suffix(df, ["vol_regime_1d"])
        col_regime = _pick_col_by_suffix(df, ["daily_regime_str_1d", "daily_regime_1d", "regime_1d"])

        # one row per timestamp (golden has multiple symbols per ts)
        g = df.groupby("timestamp", as_index=False).first().set_index("timestamp").sort_index()

        bench = str(getattr(cfg, "REGIME_BENCHMARK_SYMBOL", "BTCUSDT")).upper()
        daily = self._load_fixture_ohlcv(bench, "1D")

        # choose timestamps within fixture window
        cand = [t for t in g.index if (t >= daily.index.min()) and (t <= daily.index.max() + pd.Timedelta("2D"))]
        ts_list = _evenly_spaced(cand, int(os.environ.get("DONCH_DAILY_MAX_TS", "60")))
        if not ts_list:
            raise AssertionError("No comparable timestamps available for daily regime test")

        tol = float(os.environ.get("DONCH_VOL_PROB_TOL", "1e-3"))

        tested = 0
        bad = []

        for ts in ts_list:
            exp = g.loc[ts]

            if pd.isna(exp.get(col_code, np.nan)):
                continue

            try:
                snap = compute_daily_regime_snapshot(daily, asof_ts=ts)
            except Exception as e:
                bad.append((ts, f"compute_failed:{type(e).__name__}:{e}"))
                continue

            tested += 1

            if int(snap["regime_code_1d"]) != int(exp[col_code]):
                bad.append((ts, f"regime_code exp={int(exp[col_code])} got={int(snap['regime_code_1d'])}"))

            if col_regime is not None and pd.notna(exp.get(col_regime, None)):
                got = snap.get("daily_regime_str_1d", snap.get("daily_regime_1d", ""))
                if str(got) != str(exp[col_regime]):
                    bad.append((ts, f"regime_str exp={exp[col_regime]} got={got}"))

            if col_trend is not None and pd.notna(exp.get(col_trend, None)):
                if str(snap.get("trend_regime_1d", "")) != str(exp[col_trend]):
                    bad.append((ts, f"trend exp={exp[col_trend]} got={snap.get('trend_regime_1d')}"))

            if col_volreg is not None and pd.notna(exp.get(col_volreg, None)):
                if str(snap.get("vol_regime_1d", "")) != str(exp[col_volreg]):
                    bad.append((ts, f"vol_reg exp={exp[col_volreg]} got={snap.get('vol_regime_1d')}"))

            if col_volp is not None and pd.notna(exp.get(col_volp, np.nan)):
                expv = float(exp[col_volp])
                gotv = float(snap.get("vol_prob_low_1d", np.nan))
                if np.isfinite(expv) and abs(gotv - expv) > tol:
                    bad.append((ts, f"vol_prob exp={expv:.6f} got={gotv:.6f} tol={tol}"))

        if tested == 0:
            raise AssertionError(
                "Daily regime test did not evaluate any timestamps. "
                "Either golden has no comparable non-null rows in fixture window, or fixtures lack warmup history."
            )

        if bad:
            msg = "\n".join([f"{ts} {reason}" for ts, reason in bad[:25]])
            raise AssertionError(f"Daily regime parity mismatch for {len(bad)} cases (showing up to 25):\n{msg}")

    def test_markov4h_matches_golden_when_present(self):
        from live.regime_features import compute_markov4h_snapshot

        df = self._load_golden()

        col_prob = _pick_col_by_suffix(df, ["markov_prob_up_4h", "markov_prob_4h"])
        col_state = _pick_col_by_suffix(df, ["markov_state_4h", "markov_state_up_4h"])
        if col_prob is None or col_state is None:
            self.skipTest("golden has no Markov(4h) columns (skipping)")

        g = df.groupby("timestamp", as_index=False).first().set_index("timestamp").sort_index()

        markov_asset = str(os.environ.get("DONCH_MARKOV4H_ASSET", "ETHUSDT")).upper()
        df4h = self._load_fixture_ohlcv(markov_asset, "4H")

        cand = [t for t in g.index if (t >= df4h.index.min()) and (t <= df4h.index.max() + pd.Timedelta("4H"))]
        ts_list = _evenly_spaced(cand, int(os.environ.get("DONCH_MARKOV_MAX_TS", "60")))
        if not ts_list:
            raise AssertionError("No comparable timestamps available for Markov test")

        tol = float(os.environ.get("DONCH_MARKOV_PROB_TOL", "5e-3"))
        alpha = float(os.environ.get("DONCH_MARKOV4H_ALPHA", "0.2"))

        tested = 0
        bad = []

        for ts in ts_list:
            exp = g.loc[ts]
            if pd.isna(exp.get(col_prob, np.nan)) or pd.isna(exp.get(col_state, np.nan)):
                continue

            try:
                snap = compute_markov4h_snapshot(df4h, asof_ts=ts, alpha=alpha)
            except Exception as e:
                bad.append((ts, f"compute_failed:{type(e).__name__}:{e}"))
                continue

            tested += 1

            exp_prob = float(exp[col_prob])
            got_prob = float(snap["markov_prob_up_4h"])
            if np.isfinite(exp_prob) and abs(got_prob - exp_prob) > tol:
                bad.append((ts, f"prob exp={exp_prob:.6f} got={got_prob:.6f} tol={tol}"))

            exp_state = int(exp[col_state])
            got_state = int(snap["markov_state_4h"])
            if got_state != exp_state:
                bad.append((ts, f"state exp={exp_state} got={got_state}"))

        if tested == 0:
            raise AssertionError("Markov test did not evaluate any timestamps (no comparable rows)")

        if bad:
            msg = "\n".join([f"{ts} {reason}" for ts, reason in bad[:25]])
            raise AssertionError(f"Markov 4h parity mismatch for {len(bad)} cases (showing up to 25):\n{msg}")


if __name__ == "__main__":
    unittest.main()
