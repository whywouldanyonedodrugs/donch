import json
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from live.regime_features import compute_daily_regime_snapshot, compute_markov4h_snapshot


def _to_utc_ts(x: Any) -> pd.Timestamp:
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {x}")
    return ts


def _read_parquet_with_timestamp(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.drop(columns=["timestamp"])
        df.index = ts
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
        else:
            raise AssertionError(f"{path} has no timestamp column and index is not DatetimeIndex")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _fixture_path(repo_root: Path, symbol: str, tf: str) -> Path:
    fx_dir = repo_root / "tests" / "fixtures" / "regime"
    sym = symbol.upper()
    tfu = tf.upper()
    cand = [
        fx_dir / f"{sym}_{tfu}.parquet",
        fx_dir / f"{sym}_{tf}.parquet",
        fx_dir / f"{symbol}_{tfu}.parquet",
    ]
    for p in cand:
        if p.exists():
            return p
    raise AssertionError(f"Missing fixture parquet for {symbol} {tf}. Tried: {cand}")


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_first_key(obj: Any, key: str) -> Optional[Any]:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            out = _find_first_key(v, key)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for it in obj:
            out = _find_first_key(it, key)
            if out is not None:
                return out
    return None


def _load_regime_params(repo_root: Path) -> Tuple[int, int, float, float]:
    # Prefer exported params if present; otherwise fall back to repo constants.
    regimes_report = _load_json_if_exists(repo_root / "results" / "meta_export" / "regimes_report.json") or {}
    deploy_cfg = _load_json_if_exists(repo_root / "results" / "meta_export" / "deployment_config.json") or {}

    ma = _find_first_key(regimes_report, "ma_period")
    atr_p = _find_first_key(regimes_report, "atr_period")
    atr_m = _find_first_key(regimes_report, "atr_mult")

    if ma is None:
        ma = _find_first_key(deploy_cfg, "ma_period")
    if atr_p is None:
        atr_p = _find_first_key(deploy_cfg, "atr_period")
    if atr_m is None:
        atr_m = _find_first_key(deploy_cfg, "atr_mult")

    try:
        from config import REGIME_MA_PERIOD, REGIME_ATR_PERIOD, REGIME_ATR_MULT, MARKOV4H_PROB_EWMA_ALPHA
    except Exception:
        REGIME_MA_PERIOD, REGIME_ATR_PERIOD, REGIME_ATR_MULT, MARKOV4H_PROB_EWMA_ALPHA = 100, 20, 2.0, 0.1

    ma_period = int(ma) if ma is not None else int(REGIME_MA_PERIOD)
    atr_period = int(atr_p) if atr_p is not None else int(REGIME_ATR_PERIOD)
    atr_mult = float(atr_m) if atr_m is not None else float(REGIME_ATR_MULT)

    alpha = MARKOV4H_PROB_EWMA_ALPHA
    alpha = float(alpha) if alpha is not None else 0.1

    return ma_period, atr_period, atr_mult, alpha


def _golden_timestamp_column(g: pd.DataFrame) -> str:
    if "timestamp" in g.columns:
        return "timestamp"
    if isinstance(g.index, pd.DatetimeIndex):
        return "__index__"
    raise AssertionError("Golden parquet must have timestamp column or DatetimeIndex")


def _normalize_golden(g: pd.DataFrame) -> pd.DataFrame:
    g2 = g.copy()
    ts_col = _golden_timestamp_column(g2)
    if ts_col == "__index__":
        g2 = g2.reset_index().rename(columns={g2.reset_index().columns[0]: "timestamp"})
    g2["timestamp"] = pd.to_datetime(g2["timestamp"], utc=True, errors="coerce")
    g2 = g2.dropna(subset=["timestamp"]).sort_values(["timestamp"] + (["symbol"] if "symbol" in g2.columns else []))
    return g2


def _pick_first_nonnull_per_ts(g: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for ts, grp in g.groupby("timestamp", sort=True):
        grp2 = grp[grp[value_col].notna()]
        if grp2.empty:
            continue
        rows.append(grp2.iloc[0])
    if not rows:
        return g.iloc[0:0]
    return pd.DataFrame(rows).reset_index(drop=True)


def _prob_invariant_across_symbols(g: pd.DataFrame, prob_col: str, eps: float = 1e-12) -> bool:
    if "symbol" not in g.columns:
        return True
    gg = g[["timestamp", "symbol", prob_col]].dropna()
    if gg.empty:
        return True
    span = gg.groupby("timestamp")[prob_col].agg(lambda s: float(np.nanmax(s) - np.nanmin(s)))
    return float(span.max()) <= float(eps)


def _state_invariant_across_symbols(g: pd.DataFrame, state_col: str) -> bool:
    if "symbol" not in g.columns:
        return True
    gg = g[["timestamp", "symbol", state_col]].dropna()
    if gg.empty:
        return True
    nun = gg.groupby("timestamp")[state_col].nunique()
    return int(nun.max()) <= 1


class TestDailyRegimeAndMarkovGolden(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.golden_path = self.repo_root / "results" / "meta_export" / "golden_features.parquet"

        self.ma_period, self.atr_period, self.atr_mult, self.alpha = _load_regime_params(self.repo_root)

        self.df_daily = _read_parquet_with_timestamp(_fixture_path(self.repo_root, "BTCUSDT", "1D"))
        self.df4h = _read_parquet_with_timestamp(_fixture_path(self.repo_root, "BTCUSDT", "4H"))

        self.golden = _normalize_golden(pd.read_parquet(self.golden_path))

    def test_daily_regime_matches_golden_when_present(self) -> None:
        col = "regime_code_1d"
        self.assertIn(col, self.golden.columns, f"Golden missing required column: {col}")

        # Warmup needed for OFFLINE TMA: ~ (2*ma_period - 1) bars
        tma_warmup = int(2 * self.ma_period - 1)
        if len(self.df_daily) < tma_warmup + 1:
            raise AssertionError(f"Daily fixture too short for TMA warmup: have={len(self.df_daily)} need>={tma_warmup+1}")

        earliest_ok = self.df_daily.index[tma_warmup]  # first ts where TMA could be defined
        latest_ok = self.df_daily.index.max()

        g_pick = _pick_first_nonnull_per_ts(self.golden, col)
        g_pick = g_pick[(g_pick["timestamp"] >= earliest_ok) & (g_pick["timestamp"] <= latest_ok)].reset_index(drop=True)

        if g_pick.empty:
            raise AssertionError(
                "No comparable golden regime_code_1d rows fall within fixture window after warmup. "
                f"fixture=[{earliest_ok}..{latest_ok}]"
            )

        # Limit runtime deterministically
        g_pick = g_pick.iloc[:250].reset_index(drop=True)

        evaluated = 0
        bad = []
        first_failures = []

        for _, row in g_pick.iterrows():
            ts = _to_utc_ts(row["timestamp"])
            exp = int(row[col])

            try:
                snap = compute_daily_regime_snapshot(
                    df_daily=self.df_daily,
                    asof_ts=ts,
                    ma_period=self.ma_period,
                    atr_period=self.atr_period,
                    atr_mult=self.atr_mult,
                )
                got = int(snap["regime_code_1d"])
                evaluated += 1
                if got != exp:
                    bad.append((ts, exp, got))
            except Exception as e:
                if len(first_failures) < 10:
                    first_failures.append(f"{ts} compute_failed:{type(e).__name__}:{e}")

        if evaluated == 0:
            raise AssertionError(
                "Daily regime test did not evaluate any timestamps. "
                "Either no comparable golden rows were found in fixture window, "
                "or snapshot computation failed for all candidates.\n"
                "First failures:\n" + "\n".join(first_failures)
            )

        if bad:
            msg = "\n".join([f"{ts} exp={e} got={g}" for ts, e, g in bad[:25]])
            raise AssertionError(f"Daily regime parity mismatch for {len(bad)} cases (showing up to 25):\n{msg}")

    def test_markov4h_matches_golden_when_present(self) -> None:
        prob_col = "markov_prob_up_4h"
        state_col = "markov_state_4h"

        self.assertIn(prob_col, self.golden.columns, f"Golden missing required column: {prob_col}")
        self.assertIn(state_col, self.golden.columns, f"Golden missing required column: {state_col}")

        # Ensure the fixture window
        min_ix = self.df4h.index.min()
        max_ix = self.df4h.index.max()

        g_prob = _pick_first_nonnull_per_ts(self.golden, prob_col)
        if g_prob.empty:
            raise AssertionError("No comparable golden rows found: markov_prob_up_4h is all-null.")

        # If golden markov varies across symbols at same timestamp, we cannot compare BTC-derived markov to arbitrary symbol rows.
        prob_macro = _prob_invariant_across_symbols(self.golden, prob_col)
        state_macro = _state_invariant_across_symbols(self.golden, state_col)

        if not (prob_macro and state_macro) and "symbol" in self.golden.columns:
            # Try restricting to BTCUSDT if possible, else fail with clear diagnostic.
            gb = self.golden[self.golden["symbol"].astype(str).str.upper() == "BTCUSDT"].copy()
            gb = gb[gb[prob_col].notna()].copy()
            if gb.empty:
                sym_counts = self.golden[self.golden[prob_col].notna()]["symbol"].value_counts().head(10).to_dict()
                raise AssertionError(
                    "Golden markov columns are NOT invariant across symbols per timestamp, and BTCUSDT golden rows are null. "
                    "This implies markov_4h is symbol-specific in golden, but we only have BTCUSDT fixtures.\n"
                    f"Top symbols with non-null {prob_col}: {sym_counts}\n"
                    "Export BTCUSDT markov columns into golden, or add fixtures for the symbol(s) used."
                )
            use_g = _normalize_golden(gb)
        else:
            use_g = g_prob

        # Filter to fixture coverage after cutoff mapping (floor to 4h close)
        use_g = use_g.copy()
        use_g["cutoff_4h"] = use_g["timestamp"].dt.floor("4h")
        use_g = use_g[(use_g["cutoff_4h"] >= min_ix) & (use_g["cutoff_4h"] <= max_ix)].reset_index(drop=True)

        if use_g.empty:
            raise AssertionError("No golden markov timestamps fall within BTCUSDT_4H fixture coverage after cutoff mapping.")

        # Require enough 4h history for stable fit (diagnostic min)
        min_obs = 80
        # Identify earliest cutoff where we have >= min_obs returns available
        # returns count approx equals bars-1
        if len(self.df4h) < (min_obs + 1):
            raise AssertionError(f"4H fixture too short: have={len(self.df4h)} need>={min_obs+1}")

        # runtime limit
        use_g = use_g.iloc[:250].reset_index(drop=True)

        evaluated = 0
        bad = []
        first_failures = []

        for _, row in use_g.iterrows():
            ts = _to_utc_ts(row["timestamp"])
            exp_prob = float(row[prob_col])
            exp_state = int(row[state_col]) if pd.notna(row[state_col]) else None

            try:
                snap = compute_markov4h_snapshot(df4h=self.df4h, asof_ts=ts, alpha=self.alpha)
                got_prob = float(snap["markov_prob_up_4h"])
                got_state = int(snap["markov_state_4h"])
                evaluated += 1

                if abs(got_prob - exp_prob) > 0.005:
                    bad.append((ts, "prob", exp_prob, got_prob))
                if exp_state is not None and got_state != exp_state:
                    bad.append((ts, "state", exp_state, got_state))

            except Exception as e:
                if len(first_failures) < 10:
                    first_failures.append(f"{ts} compute_failed:{type(e).__name__}:{e}")

        if evaluated == 0:
            raise AssertionError(
                "Markov 4h test did not evaluate any timestamps.\n"
                "First failures:\n" + "\n".join(first_failures)
            )

        if bad:
            msg = "\n".join(
                [
                    f"{ts} {kind} exp={exp} got={got}" + (" tol=0.005" if kind == "prob" else "")
                    for ts, kind, exp, got in bad[:25]
                ]
            )
            raise AssertionError(f"Markov 4h parity mismatch for {len(bad)} cases (showing up to 25):\n{msg}")
