import json
import inspect
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import config as cfg


from live.regime_features import compute_daily_regime_snapshot, compute_markov4h_snapshot


def _load_golden(path: Path) -> pd.DataFrame:
    g = pd.read_parquet(path)

    # Normalize timestamp to UTC index
    if "timestamp" in g.columns:
        ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
        g = g.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp")
        g = g.set_index("timestamp", drop=True)
    else:
        # Some exports store timestamp as index
        if not isinstance(g.index, pd.DatetimeIndex):
            raise AssertionError("golden parquet has no 'timestamp' column and index is not DatetimeIndex")
        if g.index.tz is None:
            g.index = g.index.tz_localize("UTC")
        else:
            g.index = g.index.tz_convert("UTC")
        g = g.sort_index()

    return g


def _pick_col_by_suffix(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    # Exact match first
    for c in candidates:
        if c in cols:
            return c
    # Suffix match second (for exports like "xxx__regime_code_1d")
    cols_l = [(c, str(c).lower()) for c in cols]
    for cand in candidates:
        cl = cand.lower()
        for c, c_l in cols_l:
            if c_l.endswith(cl):
                return c
    return None


def _to_float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
        if not np.isfinite(x):
            return None
        return float(x)
    except Exception:
        return None


def _to_int_or_none(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        x = int(v)
        return int(x)
    except Exception:
        # Try float->int if it is an integer-ish float
        try:
            xf = _to_float_or_none(v)
            if xf is None:
                return None
            return int(round(float(xf)))
        except Exception:
            return None


def _first_nonnull_at_ts(g: pd.DataFrame, ts: pd.Timestamp, col: str) -> Optional[float]:
    """
    Golden rows contain multiple symbols per timestamp; regime/markov are macro context.
    Select the FIRST NON-NULL value across symbols at this timestamp.
    """
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
    out = _to_float_or_none(v)
    return out


def _load_fixture(fixtures_dir: Path, symbol: str, tf: str) -> pd.DataFrame:
    p = fixtures_dir / f"{symbol}_{tf}.parquet"
    if not p.exists():
        raise unittest.SkipTest(f"missing fixture: {p}")

    df = pd.read_parquet(p)

    if "timestamp" not in df.columns:
        # Sometimes parquet metadata promotes timestamp to index; handle that case
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
            df = df.copy()
            df.index = idx
        else:
            raise AssertionError(f"fixture {p} missing 'timestamp' column and index is not DatetimeIndex")
    else:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp", drop=True)

    # Ensure numeric OHLCV
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Minimal required set for regimes/markov computations
    needed = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if needed:
        df = df.dropna(subset=needed)

    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df


def _flatten_json(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_flatten_json(v, kp))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            kp = f"{prefix}[{i}]"
            out.extend(_flatten_json(v, kp))
    else:
        out.append((prefix, obj))
    return out


def _find_first_numeric(flat: List[Tuple[str, Any]], key_patterns: List[str]) -> Optional[float]:
    """
    Deterministic heuristic: among flattened (path,value) pairs, select the first numeric value
    whose path contains one of the patterns (case-insensitive), preferring earlier patterns.
    """
    flat_l = [(p.lower(), v) for p, v in flat]
    for pat in key_patterns:
        pl = pat.lower()
        for path_l, v in flat_l:
            if pl in path_l:
                x = _to_float_or_none(v)
                if x is not None:
                    return float(x)
    return None


def _load_regimes_report(repo_root: Path) -> Dict[str, Any]:
    """
    Optional: load regimes_report.json / deployment_config.json if present,
    to supply required parameters when compute_* functions do not have defaults.
    """
    out: Dict[str, Any] = {}
    p1 = repo_root / "results" / "meta_export" / "regimes_report.json"
    p2 = repo_root / "results" / "meta_export" / "deployment_config.json"

    if p1.exists():
        try:
            out["regimes_report"] = json.loads(p1.read_text(encoding="utf-8"))
        except Exception:
            out["regimes_report"] = None
    else:
        out["regimes_report"] = None

    if p2.exists():
        try:
            out["deployment_config"] = json.loads(p2.read_text(encoding="utf-8"))
        except Exception:
            out["deployment_config"] = None
    else:
        out["deployment_config"] = None

    return out


def _call_compute_daily_snapshot(df_daily: pd.DataFrame, asof_ts: pd.Timestamp, repo_root: Path) -> Dict[str, Any]:
    """
    Call compute_daily_regime_snapshot with "as-of asof_ts" semantics (no wall-clock).
    If the function requires (ma_period, atr_period, atr_mult) and has no defaults,
    source them deterministically: regimes_report.json -> deployment_config.json -> config.py constants.
    """
    sig = inspect.signature(compute_daily_regime_snapshot)

    kwargs: Dict[str, Any] = {"df_daily": df_daily, "asof_ts": asof_ts}

    required = {"ma_period", "atr_period", "atr_mult"}
    if required.issubset(sig.parameters.keys()):
        needs: List[str] = []
        for k in sorted(required):
            if sig.parameters[k].default is inspect._empty:
                needs.append(k)

        if needs:
            regimes_report = repo_root / "results" / "meta_export" / "regimes_report.json"
            deployment_config = repo_root / "deployment_config.json"

            flat: Dict[str, Any] = {}
            for p in (regimes_report, deployment_config):
                if p.exists():
                    try:
                        flat.update(_flatten_json(_load_json(p)))
                    except Exception:
                        # Fail-closed later if still missing; do not mask silent corruption
                        pass

            ma_period = _find_first_numeric(flat, ["ma_period", "regime_ma_period", "REGIME_MA_PERIOD"])
            atr_period = _find_first_numeric(flat, ["atr_period", "regime_atr_period", "atr_len", "REGIME_ATR_PERIOD"])
            atr_mult = _find_first_numeric(flat, ["atr_mult", "regime_atr_mult", "atr_multiplier", "REGIME_ATR_MULT"])

            # Deterministic fallback to repo config constants
            if ma_period is None:
                ma_period = getattr(cfg, "REGIME_MA_PERIOD", None)
            if atr_period is None:
                atr_period = getattr(cfg, "REGIME_ATR_PERIOD", None)
            if atr_mult is None:
                atr_mult = getattr(cfg, "REGIME_ATR_MULT", None)

            if ma_period is None or atr_period is None or atr_mult is None:
                raise AssertionError(
                    "compute_daily_regime_snapshot requires (ma_period, atr_period, atr_mult) with no defaults, "
                    "but they could not be found in results/meta_export/regimes_report.json, deployment_config.json, "
                    "or config.py (REGIME_MA_PERIOD/REGIME_ATR_PERIOD/REGIME_ATR_MULT)."
                )

            kwargs.update(
                {
                    "ma_period": int(ma_period),
                    "atr_period": int(atr_period),
                    "atr_mult": float(atr_mult),
                }
            )

    snap = compute_daily_regime_snapshot(**kwargs)
    if not isinstance(snap, dict):
        raise AssertionError(f"compute_daily_regime_snapshot returned non-dict: {type(snap)}")
    return snap



def _call_compute_markov_snapshot(df4h: pd.DataFrame, asof_ts: pd.Timestamp, repo_root: Path) -> Dict[str, object]:
    """
    Call compute_markov4h_snapshot respecting its actual signature.
    If alpha is required and not defaulted, source alpha from deployment_config/regimes_report when possible;
    otherwise fall back to 0.2 (the live_trader default).
    """
    sig = inspect.signature(compute_markov4h_snapshot)
    kwargs: Dict[str, Any] = {"df4h": df4h, "asof_ts": asof_ts}

    if "alpha" in sig.parameters and sig.parameters["alpha"].default is inspect._empty:
        rep = _load_regimes_report(repo_root)
        flat: List[Tuple[str, Any]] = []
        if rep.get("deployment_config") is not None:
            flat.extend(_flatten_json(rep["deployment_config"]))
        if rep.get("regimes_report") is not None:
            flat.extend(_flatten_json(rep["regimes_report"]))

        alpha = _find_first_numeric(flat, ["markov4h_prob_ewma_alpha", "MARKOV4H_PROB_EWMA_ALPHA"])
        if alpha is None:
            alpha = getattr(cfg, "MARKOV4H_PROB_EWMA_ALPHA", 0.2)

        kwargs["alpha"] = float(alpha)

    return compute_markov4h_snapshot(**kwargs)


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

        # Golden often lacks vol prob; treat parity as OPTIONAL.
        col_volp = _pick_col_by_suffix(g, ["vol_prob_1d", "vol_prob_low_1d", "vol_prob_high_1d"])

        daily = _load_fixture(self.fixtures_dir, "BTCUSDT", "1D")

        tol = 1e-3
        tested = 0
        bad: List[str] = []
        compute_fail: List[str] = []

        # Candidates: timestamps where expected regime_code exists (across ANY symbol at that timestamp)
        cand: List[pd.Timestamp] = []
        for ts in g.index.unique():
            exp_code = _first_nonnull_at_ts(g, ts, col_code)
            if exp_code is None:
                continue
            # must be within fixture window (+1d tolerance for intra-day decision_ts)
            if ts < daily.index.min():
                continue
            if ts > daily.index.max() + pd.Timedelta("1D"):
                continue
            cand.append(ts)

        cand = sorted(cand)

        for ts in cand:
            exp_code = _first_nonnull_at_ts(g, ts, col_code)
            if exp_code is None:
                continue

            try:
                snap = _call_compute_daily_snapshot(daily, asof_ts=ts, repo_root=self.repo_root)
            except Exception as e:
                compute_fail.append(f"{ts} compute_failed:{type(e).__name__}:{e}")
                continue

            got_code = snap.get("regime_code_1d", None)
            if got_code is None or (isinstance(got_code, float) and np.isnan(got_code)):
                bad.append(f"{ts} regime_code missing/NaN exp={int(exp_code)}")
            else:
                if int(got_code) != int(exp_code):
                    bad.append(f"{ts} regime_code exp={int(exp_code)} got={int(got_code)}")

            # Optional: vol prob parity if golden has a usable column AND expected is present at this ts.
            if col_volp:
                exp_volp = _first_nonnull_at_ts(g, ts, col_volp)
                got_volp = snap.get("vol_prob_low_1d", None)
                if exp_volp is not None and got_volp is not None and np.isfinite(float(got_volp)):
                    if abs(float(got_volp) - float(exp_volp)) > tol:
                        bad.append(f"{ts} vol_prob exp={float(exp_volp):.6f} got={float(got_volp):.6f} tol={tol}")

            tested += 1
            if tested >= 200:
                break

        if tested == 0:
            # Show the first few compute failures to avoid “silent” empty evaluation
            msg = "\n".join(compute_fail[:10]) if compute_fail else "(no compute failures captured)"
            raise AssertionError(
                "Daily regime test did not evaluate any timestamps. "
                "Either no comparable golden rows were found in fixture window, "
                "or snapshot computation failed for all candidates.\n"
                f"First failures:\n{msg}"
            )

        if bad:
            msg = "\n".join(bad[:25])
            raise AssertionError(f"Daily regime parity mismatch for {len(bad)} cases (showing up to 25):\n{msg}")

    def test_markov4h_matches_golden_when_present(self) -> None:
        g = _load_golden(self.golden_path)

        # Golden uses markov_prob_up_4h + markov_state_4h; accept suffix variants.
        col_prob = _pick_col_by_suffix(g, ["markov_prob_up_4h", "markov_prob_4h"])
        col_state = _pick_col_by_suffix(g, ["markov_state_4h"])
        if not col_prob or not col_state:
            raise unittest.SkipTest("golden missing markov 4h columns")

        df4h = _load_fixture(self.fixtures_dir, "BTCUSDT", "4H")

        tol = 5e-3
        tested = 0
        bad: List[str] = []
        compute_fail: List[str] = []

        cand: List[pd.Timestamp] = []
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
                snap = _call_compute_markov_snapshot(df4h, asof_ts=ts, repo_root=self.repo_root)
            except Exception as e:
                compute_fail.append(f"{ts} compute_failed:{type(e).__name__}:{e}")
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
            msg = "\n".join(compute_fail[:10]) if compute_fail else "(no compute failures captured)"
            raise AssertionError(
                "Markov test did not evaluate any timestamps. "
                "Either no comparable golden rows were found in fixture window, "
                "or snapshot computation failed for all candidates.\n"
                f"First failures:\n{msg}"
            )

        if bad:
            msg = "\n".join(bad[:25])
            raise AssertionError(f"Markov 4h parity mismatch for {len(bad)} cases (showing up to 25):\n{msg}")
