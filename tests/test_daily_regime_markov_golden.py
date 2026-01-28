import json
import unittest
from pathlib import Path

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


def _pick_first(d: dict, keys: list[str], default):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _load_regime_params() -> dict:
    """
    Prefer meta_export reports if present; else fallback to config defaults.
    """
    regimes_report = _load_json(PROJECT_ROOT / "results" / "meta_export" / "regimes_report.json")
    deploy_cfg = _load_json(PROJECT_ROOT / "results" / "meta_export" / "deployment_config.json")

    # config.py fallback (repo root)
    try:
        import config as cfg  # type: ignore
    except Exception:
        cfg = None

    def g(keys, default):
        v = _pick_first(regimes_report, keys, None)
        if v is None:
            v = _pick_first(deploy_cfg, keys, None)
        if v is None and cfg is not None:
            for k in keys:
                if hasattr(cfg, k):
                    return getattr(cfg, k)
        return default if v is None else v

    # OFFLINE defaults are (200,20,2.0,maxiter=200) and alpha=0.2
    ma_period = int(g(["ma_period", "REGIME_MA_PERIOD", "regime_ma_period"], 200))
    atr_period = int(g(["atr_period", "REGIME_ATR_PERIOD", "regime_atr_period"], 20))
    atr_mult = float(g(["atr_mult", "REGIME_ATR_MULT", "regime_atr_mult"], 2.0))
    ewma_alpha = float(g(["MARKOV4H_PROB_EWMA_ALPHA", "markov4h_prob_ewma_alpha"], 0.2))

    # Symbols:
    # Daily uses benchmark if present, else REGIME_ASSET; Markov uses REGIME_ASSET.
    daily_sym = g(["benchmark", "REGIME_BENCHMARK_SYMBOL", "regime_benchmark_symbol"], None)
    markov_sym = g(["REGIME_ASSET", "regime_asset"], None)

    if daily_sym is None and cfg is not None:
        daily_sym = getattr(cfg, "REGIME_BENCHMARK_SYMBOL", getattr(cfg, "REGIME_ASSET", "BTCUSDT"))
    if markov_sym is None and cfg is not None:
        markov_sym = getattr(cfg, "REGIME_ASSET", "ETHUSDT")

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


def _load_fixture(symbol: str, suffix: str) -> pd.DataFrame:
    # suffix examples: "1D", "4H"
    p = FIXTURES_DIR / f"{symbol}_{suffix}.parquet"
    if not p.exists():
        raise AssertionError(f"Missing fixture: {p}")
    df = pd.read_parquet(p)
    df = _ensure_utc_index(df, "timestamp")
    return df


def _load_golden() -> pd.DataFrame:
    if not GOLDEN_PATH.exists():
        raise AssertionError(f"Missing golden file: {GOLDEN_PATH}")
    g = pd.read_parquet(GOLDEN_PATH)
    g = _ensure_utc_index(g, "timestamp") if "timestamp" in g.columns else _ensure_utc_index(g.reset_index(), "timestamp")
    # Golden rows are macro-identical across symbols per timestamp, but keep deterministic choice:
    if "symbol" in g.columns:
        g = g.sort_values(["symbol"])
    g = g.groupby(g.index).first()
    return g.sort_index()


class TestDailyRegimeAndMarkovGolden(unittest.TestCase):
    def test_daily_regime_matches_golden_when_present(self):
        params = _load_regime_params()
        g = _load_golden()

        daily_sym = params["daily_symbol"]
        df1d = _load_fixture(daily_sym, "1D")

        cfg = DailyRegimeConfig(
            ma_period=int(params["ma_period"]),
            atr_period=int(params["atr_period"]),
            atr_mult=float(params["atr_mult"]),
            maxiter=200,
        )
        series = compute_daily_regime_series(df1d, cfg)

        # Columns in golden (observed): regime_code_1d, vol_prob_low_1d
        if "regime_code_1d" not in g.columns:
            raise AssertionError("Golden missing regime_code_1d")

        have_vol = "vol_prob_low_1d" in g.columns

        bad = []
        eval_n = 0

        for ts, row in g.iterrows():
            # Ensure within fixture window (as-of semantics)
            if ts < series.index.min() or ts > pd.Timestamp(series.index.max()).tz_convert("UTC") + pd.Timedelta(days=1):
                continue

            # daily series index is at 00:00, so as-of ts is just last daily row <= ts
            pos = series.index.searchsorted(ts, side="right") - 1
            if pos < 0:
                continue
            got = series.iloc[int(pos)]

            exp_code = int(row["regime_code_1d"])
            got_code = int(got["regime_code"]) if pd.notna(got["regime_code"]) else None

            if got_code != exp_code:
                bad.append(f"{ts} exp={exp_code} got={got_code}")

            if have_vol:
                exp_vol = float(row["vol_prob_low_1d"])
                got_vol = float(got["vol_prob_low"]) if pd.notna(got["vol_prob_low"]) else np.nan
                if not np.isfinite(got_vol) or abs(exp_vol - got_vol) > 1e-3:
                    bad.append(f"{ts} vol_prob_low exp={exp_vol:.6f} got={got_vol:.6f} tol=0.001")

            eval_n += 1
            if eval_n >= 300:
                break

        if eval_n == 0:
            raise AssertionError("Daily regime test did not evaluate any timestamps (window mismatch).")

        if bad:
            msg = "\n".join(bad[:25])
            raise AssertionError(f"Daily regime parity mismatch for {len(bad)} cases (showing up to 25):\n{msg}")

    def test_markov4h_matches_golden_when_present(self):
        params = _load_regime_params()
        g = _load_golden()

        markov_sym = params["markov_symbol"]
        df4h = _load_fixture(markov_sym, "4H")

        cfg = Markov4hConfig(
            ewma_alpha=float(params["ewma_alpha"]),
            maxiter=200,
        )
        series = compute_markov4h_series(df4h, cfg)

        if "markov_prob_up_4h" not in g.columns:
            raise AssertionError("Golden missing markov_prob_up_4h")
        if "markov_state_4h" not in g.columns:
            raise AssertionError("Golden missing markov_state_4h")

        bad = []
        eval_n = 0

        for ts, row in g.iterrows():
            # Ensure within 4h fixture window
            if ts < series.index.min() or ts > series.index.max():
                continue

            pos = series.index.searchsorted(ts, side="right") - 1
            if pos < 0:
                continue
            got = series.iloc[int(pos)]

            exp_p = float(row["markov_prob_up_4h"])
            got_p = float(got["prob_up"])
            if abs(exp_p - got_p) > 5e-3:
                bad.append(f"{ts} prob exp={exp_p:.6f} got={got_p:.6f} tol=0.005")

            exp_s = int(row["markov_state_4h"])
            got_s = int(got["state_up"])
            if exp_s != got_s:
                bad.append(f"{ts} state exp={exp_s} got={got_s}")

            eval_n += 1
            if eval_n >= 300:
                break

        if eval_n == 0:
            raise AssertionError("Markov 4h test did not evaluate any timestamps (window mismatch).")

        if bad:
            msg = "\n".join(bad[:25])
            raise AssertionError(f"Markov 4h parity mismatch for {len(bad)} cases (showing up to 25):\n{msg}")


if __name__ == "__main__":
    unittest.main()
