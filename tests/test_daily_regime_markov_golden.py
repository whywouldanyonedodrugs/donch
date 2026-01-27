import unittest
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from live.regime_features import compute_daily_regime_snapshot, compute_markov4h_snapshot


def _pick_col_by_suffix(df: pd.DataFrame, suffixes: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for suf in suffixes:
        for c in cols:
            if c == suf or c.endswith(suf):
                return c
    return None


def _load_golden(path: Path) -> pd.DataFrame:
    g = pd.read_parquet(path)
    if "timestamp" in g.columns:
        ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
        g = g.assign(timestamp=ts).dropna(subset=["timestamp"]).set_index("timestamp")
    else:
        if not isinstance(g.index, pd.DatetimeIndex):
            raise ValueError("golden parquet must have 'timestamp' column or DatetimeIndex")
        g = g.copy()
        g.index = pd.to_datetime(g.index, utc=True, errors="coerce")
        g = g.dropna().sort_index()
    if "symbol" not in g.columns:
        raise ValueError("golden parquet missing 'symbol' column")
    return g.sort_index()


def _load_fixture(fixtures_dir: Path, symbol: str, tf_tag: str) -> pd.DataFrame:
    p = fixtures_dir / f"{symbol.upper()}_{tf_tag.upper()}.parquet"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_parquet(p)
    if "timestamp" not in df.columns:
        raise ValueError(f"{p.name} missing 'timestamp' column")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp", drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df.sort_index()


def _first_nonnull_at_ts(g: pd.DataFrame, ts: pd.Timestamp, col: str) -> Optional[float]:
    if col not in g.columns:
        return None
    if ts not in g.index:
        return None
    v = g.loc[ts, col]
    if isinstance(v, pd.Series):
        v = v.dropna()
        if v.empty:
            return None
        v = v.iloc[0]
    if pd.isna(v):
        return None
    return float(v)


class TestDailyRegimeAndMarkovGolden(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.golden_path = self.repo_root / "results" / "meta_export" / "golden_features.parquet"
        self.fixtures_dir = self.repo_root / "tests" / "fixtures" / "regime"

        if not self.golden_path.exists():
            raise unittest.SkipTest(f"missing golden parquet at {self.golden_path}")
        if not self.fixtures_dir.exists():
            raise unittest.SkipTest(f"missing fixtures dir at {self.fixtures_dir}")

    def test_daily_regime_matches_golden_when_present(self) -> None:
        g = _load_golden(self.golden_path)

        col_code = _pick_col_by_suffix(g, ["regime_code_1d"])
        if not col_code:
            raise unittest.SkipTest("golden has no regime_code_1d column (or suffix match)")

        # vol_prob column name varies across exports; accept any of these.
        col_volp = _pick_col_by_suffix(g, ["vol_prob_1d", "vol_prob_low_1d", "vol_prob_high_1d"])
        # Some goldens won't have vol prob; regime_code parity is still required.
        tol = 1e-3

        daily = _load_fixture(self.fixtures_dir, "BTCUSDT", "1D")

        # Candidate timestamps: those with a non-null expected regime code.
        cand = []
        for ts in g.index.unique():
            exp_code = _first_nonnull_at_ts(g, ts, col_code)
            if exp_code is None:
                continue
            # must be within fixture window (daily bars are close-labeled)
            if ts < daily.index.min():
                continue
            if ts > daily.index.max() + pd.Timedelta("1D"):
                continue
            cand.append(ts)
        cand = sorted(cand)

        tested = 0
        bad: List[str] = []

        for ts in cand:
            exp_code = _first_nonnull_at_ts(g, ts, col_code)
            if exp_code is None:
                continue

            try:
                snap = compute_daily_regime_snapshot(daily, asof_ts=ts)
            except Exception as e:
                bad.append(f"{ts} compute_failed:{type(e).__name__}:{e}")
                continue

            got_code = snap.get("regime_code_1d", None)
            if got_code is None or (isinstance(got_code, float) and np.isnan(got_code)):
                bad.append(f"{ts} got regime_code_1d missing/NaN")
                continue

            if int(got_code) != int(exp_code):
                bad.append(f"{ts} regime_code exp={int(exp_code)} got={int(got_code)}")

            if col_volp:
                exp_volp = _first_nonnull_at_ts(g, ts, col_volp)
                if exp_volp is not None:
                    got_volp = snap.get("vol_prob_1d", None)
                    # snap always uses canonical key vol_prob_1d
                    if got_volp is None or (isinstance(got_volp, float) and np.isnan(got_volp)):
                        bad.append(f"{ts} vol_prob missing/NaN exp={exp_volp}")
                    else:
                        if abs(float(got_volp) - float(exp_volp)) > tol:
                            bad.append(f"{ts} vol_prob exp={float(exp_volp):.6f} got={float(got_volp):.6f} tol={tol}")

            tested += 1
            if tested >= 200:
                break

        if tested == 0:
            raise AssertionError(
                "Daily regime test did not evaluate any timestamps. "
                "Either golden has no comparable non-null rows in fixture window, or fixtures lack warmup history."
            )

        if bad:
            msg = "\n".join(bad[:25])
            raise AssertionError(f"Daily regime parity mismatch for {len(bad)} cases (showing up to 25):\n{msg}")

    def test_markov4h_matches_golden_when_present(self) -> None:
        g = _load_golden(self.golden_path)

        col_prob = _pick_col_by_suffix(g, ["markov_prob_up_4h", "markov_prob_4h"])
        col_state = _pick_col_by_suffix(g, ["markov_state_4h"])
        if not col_prob or not col_state:
            raise unittest.SkipTest("golden missing markov 4h columns")

        df4h = _load_fixture(self.fixtures_dir, "BTCUSDT", "4H")

        tol = 5e-3
        tested = 0
        bad: List[str] = []

        cand = []
        for ts in g.index.unique():
            exp_prob = _first_nonnull_at_ts(g, ts, col_prob)
            exp_state = _first_nonnull_at_ts(g, ts, col_state)
            if exp_prob is None or exp_state is None:
                continue
            if ts < df4h.index.min():
                continue
            if ts > df4h.index.max() + pd.Timedelta("4h"):
                continue
            cand.append(ts)
        cand = sorted(cand)

        for ts in cand:
            exp_prob = _first_nonnull_at_ts(g, ts, col_prob)
            exp_state = _first_nonnull_at_ts(g, ts, col_state)
            if exp_prob is None or exp_state is None:
                continue

            try:
                snap = compute_markov4h_snapshot(df4h, asof_ts=ts)
            except Exception as e:
                bad.append(f"{ts} compute_failed:{type(e).__name__}:{e}")
                continue

            got_prob = snap.get("markov_prob_up_4h", None)
            got_state = snap.get("markov_state_4h", None)

            if got_prob is None or (isinstance(got_prob, float) and np.isnan(got_prob)):
                bad.append(f"{ts} prob missing/NaN exp={float(exp_prob):.6f}")
            else:
                if abs(float(got_prob) - float(exp_prob)) > tol:
                    bad.append(f"{ts} prob exp={float(exp_prob):.6f} got={float(got_prob):.6f} tol={tol}")

            if got_state is None or (isinstance(got_state, float) and np.isnan(got_state)):
                bad.append(f"{ts} state missing/NaN exp={int(exp_state)}")
            else:
                if int(got_state) != int(exp_state):
                    bad.append(f"{ts} state exp={int(exp_state)} got={int(got_state)}")

            tested += 1
            if tested >= 200:
                break

        if tested == 0:
            raise AssertionError("Markov test did not evaluate any timestamps (no comparable golden rows in fixture window).")

        if bad:
            msg = "\n".join(bad[:25])
            raise AssertionError(f"Markov 4h parity mismatch for {len(bad)} cases (showing up to 25):\n{msg}")
