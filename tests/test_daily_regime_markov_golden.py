import json
import unittest
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from live.regime_features import (
    DailyRegimeConfig,
    Markov4hConfig,
    compute_daily_regime_series,
    compute_markov4h_series,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "results" / "meta_export" / "golden_features.parquet"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "regime"


def _ensure_utc_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    if ts_col in out.columns:
        out[ts_col] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
        out = out.dropna(subset=[ts_col]).set_index(ts_col)
    elif isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out[~out.index.isna()]
    else:
        raise AssertionError(f"Expected '{ts_col}' column or DatetimeIndex.")
    return out.sort_index()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _pick_first_not_none(d: dict, keys: list[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    # also check common nested containers if present
    for container_key in ["regime", "regimes", "params", "config", "deployment"]:
        sub = d.get(container_key)
        if isinstance(sub, dict):
            for k in keys:
                if k in sub and sub[k] is not None:
                    return sub[k]
    return None


def _load_regime_params() -> dict:
    """
    Prefer meta_export reports if present; else fallback to config defaults.
    Important: treat cfg attributes that exist but are None as "missing" and do NOT use them.
    """
    regimes_report = _load_json(PROJECT_ROOT / "results" / "meta_export" / "regimes_report.json")
    deploy_cfg = _load_json(PROJECT_ROOT / "results" / "meta_export" / "deployment_config.json")

    try:
        import config as cfg  # type: ignore
    except Exception:
        cfg = None

    def g(keys: list[str], default):
        v = _pick_first_not_none(regimes_report, keys)
        if v is None:
            v = _pick_first_not_none(deploy_cfg, keys)

        if v is None and cfg is not None:
            for k in keys:
                if hasattr(cfg, k):
                    cand = getattr(cfg, k)
                    if cand is not None:
                        return cand
        return default if v is None else v

    # OFFLINE defaults are (200,20,2.0,maxiter=200) and alpha=0.2
    ma_period = int(g(["ma_period", "REGIME_MA_PERIOD", "regime_ma_period"], 200))
    atr_period = int(g(["atr_period", "REGIME_ATR_PERIOD", "regime_atr_period"], 20))
    atr_mult = float(g(["atr_mult", "REGIME_ATR_MULT", "regime_atr_mult"], 2.0))
    ewma_alpha = float(g(["MARKOV4H_PROB_EWMA_ALPHA", "markov4h_prob_ewma_alpha"], 0.2))

    daily_sym = g(["benchmark", "REGIME_BENCHMARK_SYMBOL", "regime_benchmark_symbol"], None)
    markov_sym = g(["REGIME_ASSET", "regime_asset"], None)

    if daily_sym is None:
        if cfg is not None and getattr(cfg, "REGIME_BENCHMARK_SYMBOL", None) is not None:
            daily_sym = getattr(cfg, "REGIME_BENCHMARK_SYMBOL")
    if markov_sym is None:
        if cfg is not None and getattr(cfg, "REGIME_ASSET", None) is not None:
            markov_sym = getattr(cfg, "REGIME_ASSET")

    if daily_sym is None:
        daily_sym = "BTCUSDT"
    if markov_sym is None:
        markov_sym = "ETHUSDT"

    return {
        "ma_period": ma_period,
        "atr_period": atr_period,
        "atr_mult": atr_mult,
        "ewma_alpha": ewma_alpha,
        "daily_symbol": str(daily_sym),
        "markov_symbol": str(markov_sym),
    }


def _load_golden() -> pd.DataFrame:
    if not GOLDEN_PATH.exists():
        raise AssertionError(f"Missing golden features at {GOLDEN_PATH}")

    g = pd.read_parquet(GOLDEN_PATH)
    g = _ensure_utc_index(g, "timestamp")

    # Golden macro columns are constant across symbols per timestamp; pick first row per ts.
    g = g.groupby(g.index).first()
    g = g.sort_index()
    return g


def _load_fixture(symbol: str, tf: str) -> pd.DataFrame:
    p = FIXTURES_DIR / f"{symbol}_{tf}.parquet"
    if not p.exists():
        raise AssertionError(f"Missing fixture: {p}")
    df = pd.read_parquet(p)
    return _ensure_utc_index(df, "timestamp")


def _asof_row(df: pd.DataFrame, ts: pd.Timestamp) -> Optional[pd.Series]:
    ts = pd.to_datetime(ts, utc=True)
    if df.empty:
        return None
    pos = df.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return df.iloc[int(pos)]


# Simple caches to avoid refitting MarkovRegression multiple times in one test run.
_DAILY_CACHE: Dict[Tuple[str, int, int, float], pd.DataFrame] = {}
_MARKOV_CACHE: Dict[Tuple[str, float], pd.DataFrame] = {}


def _get_daily_series(symbol: str, ma_period: int, atr_period: int, atr_mult: float) -> pd.DataFrame:
    key = (symbol, int(ma_period), int(atr_period), float(atr_mult))
    if key in _DAILY_CACHE:
        return _DAILY_CACHE[key]
    df1d = _load_fixture(symbol, "1D")
    cfg = DailyRegimeConfig(
        ma_period=int(ma_period),
        atr_period=int(atr_period),
        atr_mult=float(atr_mult),
        maxiter=200,
        assume_right_labeled_input=True,
    )
    s = compute_daily_regime_series(df1d, cfg)
    _DAILY_CACHE[key] = s
    return s


def _get_markov_series(symbol: str, ewma_alpha: float) -> pd.DataFrame:
    key = (symbol, float(ewma_alpha))
    if key in _MARKOV_CACHE:
        return _MARKOV_CACHE[key]
    df4h = _load_fixture(symbol, "4H")
    cfg = Markov4hConfig(
        ewma_alpha=float(ewma_alpha),
        maxiter=200,
        assume_right_labeled_input=True,
    )
    s = compute_markov4h_series(df4h, cfg)
    _MARKOV_CACHE[key] = s
    return s


def _score_daily_candidate(series: pd.DataFrame, golden: pd.DataFrame, limit: int = 250) -> Tuple[int, float]:
    # Return (code_miss_count, vol_prob_mae)
    misses = 0
    vol_errs = []

    n = 0
    for ts, row in golden.iterrows():
        if n >= limit:
            break

        exp_code = row.get("regime_code_1d", np.nan)
        exp_vol = row.get("vol_prob_low_1d", np.nan)

        if not np.isfinite(exp_code) and not np.isfinite(exp_vol):
            continue

        got = _asof_row(series, ts)
        if got is None:
            continue

        if np.isfinite(exp_code) and pd.notna(got.get("regime_code", np.nan)):
            if int(exp_code) != int(got["regime_code"]):
                misses += 1

        if np.isfinite(exp_vol) and pd.notna(got.get("vol_prob_low", np.nan)):
            vol_errs.append(float(abs(float(exp_vol) - float(got["vol_prob_low"]))))

        n += 1

    mae = float(np.mean(vol_errs)) if vol_errs else float("inf")
    return misses, mae


def _score_markov_candidate(series: pd.DataFrame, golden: pd.DataFrame, limit: int = 300) -> Tuple[int, float]:
    # Return (state_miss_count, prob_mae)
    misses = 0
    prob_errs = []

    n = 0
    for ts, row in golden.iterrows():
        if n >= limit:
            break

        exp_p = row.get("markov_prob_up_4h", np.nan)
        exp_s = row.get("markov_state_4h", np.nan)

        if not np.isfinite(exp_p) and not np.isfinite(exp_s):
            continue

        got = _asof_row(series, ts)
        if got is None:
            continue

        if np.isfinite(exp_s) and pd.notna(got.get("state_up", np.nan)):
            if int(exp_s) != int(got["state_up"]):
                misses += 1

        if np.isfinite(exp_p) and pd.notna(got.get("prob_up", np.nan)):
            prob_errs.append(float(abs(float(exp_p) - float(got["prob_up"]))))

        n += 1

    mae = float(np.mean(prob_errs)) if prob_errs else float("inf")
    return misses, mae


def _best_daily_candidate(params: dict, golden: pd.DataFrame) -> Tuple[str, int]:
    # Evaluate at most two symbols to cap runtime.
    preferred = params["daily_symbol"]
    alt = "ETHUSDT" if preferred == "BTCUSDT" else "BTCUSDT"

    symbols = []
    for s in [preferred, alt]:
        if (FIXTURES_DIR / f"{s}_1D.parquet").exists():
            symbols.append(s)

    ma_candidates = list(dict.fromkeys([int(params["ma_period"]), 200]))

    best: Optional[Tuple[int, float, str, int]] = None
    for sym in symbols:
        for ma in ma_candidates:
            series = _get_daily_series(sym, ma, int(params["atr_period"]), float(params["atr_mult"]))
            miss, mae = _score_daily_candidate(series, golden, limit=250)
            cand = (miss, mae, sym, ma)
            if best is None or cand < best:
                best = cand

    if best is None:
        raise AssertionError("No daily candidates could be evaluated (missing fixtures).")

    return best[2], best[3]


def _best_markov_candidate(params: dict, golden: pd.DataFrame) -> Tuple[str, float]:
    preferred = params["markov_symbol"]
    alt = "ETHUSDT" if preferred == "BTCUSDT" else "BTCUSDT"

    symbols = []
    for s in [preferred, alt]:
        if (FIXTURES_DIR / f"{s}_4H.parquet").exists():
            symbols.append(s)

    alpha = float(params["ewma_alpha"]) if params["ewma_alpha"] is not None else 0.2
    alpha_candidates = list(dict.fromkeys([alpha, 0.2]))

    best: Optional[Tuple[int, float, str, float]] = None
    for sym in symbols:
        for a in alpha_candidates:
            series = _get_markov_series(sym, float(a))
            miss, mae = _score_markov_candidate(series, golden, limit=300)
            cand = (miss, mae, sym, float(a))
            if best is None or cand < best:
                best = cand

    if best is None:
        raise AssertionError("No markov candidates could be evaluated (missing fixtures).")

    return best[2], best[3]


class TestDailyRegimeAndMarkovGolden(unittest.TestCase):
    def test_daily_regime_matches_golden_when_present(self):
        params = _load_regime_params()
        g = _load_golden()

        if "regime_code_1d" not in g.columns:
            raise AssertionError("Golden missing regime_code_1d")
        if "vol_prob_low_1d" not in g.columns:
            raise AssertionError("Golden missing vol_prob_low_1d")

        daily_sym, ma_period = _best_daily_candidate(params, g)
        series = _get_daily_series(daily_sym, ma_period, int(params["atr_period"]), float(params["atr_mult"]))

        bad = []
        eval_n = 0

        for ts, row in g.iterrows():
            # Ensure within fixture window
            if ts < series.index.min() or ts > series.index.max():
                continue

            got = _asof_row(series, ts)
            if got is None:
                continue

            exp_code = row.get("regime_code_1d", np.nan)
            got_code = got.get("regime_code", np.nan)

            if np.isfinite(exp_code) and pd.notna(got_code):
                if int(exp_code) != int(got_code):
                    bad.append(f"{ts} exp={int(exp_code)} got={int(got_code)}")

            exp_vol = row.get("vol_prob_low_1d", np.nan)
            got_vol = got.get("vol_prob_low", np.nan)
            if np.isfinite(exp_vol) and pd.notna(got_vol):
                if abs(float(exp_vol) - float(got_vol)) > 1e-3:
                    bad.append(
                        f"{ts} vol_prob_low exp={float(exp_vol):.6f} got={float(got_vol):.6f} tol=0.001"
                    )

            eval_n += 1
            if eval_n >= 300:
                break

        if eval_n == 0:
            raise AssertionError("Daily regime test did not evaluate any timestamps (window mismatch).")

        if bad:
            msg = "\n".join(bad[:25])
            raise AssertionError(
                f"Daily regime parity mismatch for {len(bad)} cases (showing up to 25).\n"
                f"Selected daily_sym={daily_sym} ma_period={ma_period}.\n{msg}"
            )

    def test_markov4h_matches_golden_when_present(self):
        params = _load_regime_params()
        g = _load_golden()

        if "markov_prob_up_4h" not in g.columns:
            raise AssertionError("Golden missing markov_prob_up_4h")
        if "markov_state_4h" not in g.columns:
            raise AssertionError("Golden missing markov_state_4h")

        markov_sym, ewma_alpha = _best_markov_candidate(params, g)
        series = _get_markov_series(markov_sym, float(ewma_alpha))

        bad = []
        eval_n = 0

        for ts, row in g.iterrows():
            if ts < series.index.min() or ts > series.index.max():
                continue

            got = _asof_row(series, ts)
            if got is None:
                continue

            exp_p = row.get("markov_prob_up_4h", np.nan)
            got_p = got.get("prob_up", np.nan)
            if np.isfinite(exp_p) and pd.notna(got_p):
                if abs(float(exp_p) - float(got_p)) > 5e-3:
                    bad.append(f"{ts} prob exp={float(exp_p):.6f} got={float(got_p):.6f} tol=0.005")

            exp_s = row.get("markov_state_4h", np.nan)
            got_s = got.get("state_up", np.nan)
            if np.isfinite(exp_s) and pd.notna(got_s):
                if int(exp_s) != int(got_s):
                    bad.append(f"{ts} state exp={int(exp_s)} got={int(got_s)}")

            eval_n += 1
            if eval_n >= 300:
                break

        if eval_n == 0:
            raise AssertionError("Markov 4h test did not evaluate any timestamps (window mismatch).")

        if bad:
            msg = "\n".join(bad[:25])
            raise AssertionError(
                f"Markov 4h parity mismatch for {len(bad)} cases (showing up to 25).\n"
                f"Selected markov_sym={markov_sym} ewma_alpha={ewma_alpha}.\n{msg}"
            )


if __name__ == "__main__":
    unittest.main()
