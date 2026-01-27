#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def _tf_to_timedelta(tf: str) -> pd.Timedelta:
    t = tf.strip()
    t = t.replace("MIN", "min").replace("Min", "min")
    t = t.replace("H", "h").replace("D", "D")
    return pd.Timedelta(t)


def _load_fixture_as_open_ts(path: Path, tf: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing 'timestamp' column")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.drop(columns=["timestamp"]).copy()
    df.index = ts
    df.index.name = "timestamp"

    need = {"open", "high", "low", "close", "volume"}
    miss = sorted(list(need - set(df.columns)))
    if miss:
        raise ValueError(f"{path} missing OHLCV cols: {miss}")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).sort_index()

    # Fixtures are close-timestamped (right/right). Convert to bar-open timestamps.
    td = _tf_to_timedelta(tf)
    df = df.copy()
    df.index = df.index - td
    df.index.name = "timestamp"
    return df


def _drop_incomplete_last_bar_open_ts(df: pd.DataFrame, tf: str, asof_ts: pd.Timestamp) -> pd.DataFrame:
    # df indexed at BAR OPEN times
    floor = asof_ts.floor(tf)
    return df.loc[df.index < floor].copy()


def _fit_markov(ret: pd.Series) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ret = ret.dropna()
    if len(ret) < 200:
        raise ValueError(f"insufficient returns: {len(ret)} (<200)")
    mod = MarkovRegression(ret, k_regimes=2, switching_variance=True, trend="c")
    res = mod.fit(disp=False)
    params = {}
    for i in range(2):
        # statsmodels uses sigma2[i] naming for switching variance
        key = f"sigma2[{i}]"
        if key in res.params:
            params[key] = float(res.params[key])
    return res, params


def _daily_variants(daily: pd.DataFrame, asof_ts: pd.Timestamp) -> Dict[str, float]:
    df = _drop_incomplete_last_bar_open_ts(daily, "1D", asof_ts)
    close = df["close"].astype(float)

    outs: Dict[str, float] = {}

    ret_pct = close.pct_change()
    ret_log = np.log(close).diff()

    for ret_name, ret in [("pct", ret_pct), ("log", ret_log)]:
        res, params = _fit_markov(ret)
        # low-vol regime by fitted sigma2
        sigs = [params.get("sigma2[0]", np.nan), params.get("sigma2[1]", np.nan)]
        low_idx = int(np.nanargmin(np.array(sigs)))

        probs_f = res.filtered_marginal_probabilities.copy()
        probs_s = res.smoothed_marginal_probabilities.copy()

        for prob_name, probs in [("filtered", probs_f), ("smoothed", probs_s)]:
            # Align probs index to ret index (statsmodels already does)
            p_low = float(probs.iloc[-1, low_idx])
            outs[f"{ret_name}_{prob_name}"] = p_low

    return outs


def _markov4h_variants(h4: pd.DataFrame, asof_ts: pd.Timestamp, alpha: float) -> Dict[str, Tuple[float, int]]:
    df = _drop_incomplete_last_bar_open_ts(h4, "4h", asof_ts)
    close = df["close"].astype(float)

    outs: Dict[str, Tuple[float, int]] = {}

    ret_pct = close.pct_change()
    ret_log = np.log(close).diff()

    for ret_name, ret in [("pct", ret_pct), ("log", ret_log)]:
        res, _ = _fit_markov(ret)

        probs_f = res.filtered_marginal_probabilities.copy()
        probs_s = res.smoothed_marginal_probabilities.copy()

        for prob_name, probs in [("filtered", probs_f), ("smoothed", probs_s)]:
            # Determine UP regime by weighted mean return (as-of history only)
            r = ret.dropna()
            p0 = probs.iloc[:, 0].reindex(r.index)
            p1 = probs.iloc[:, 1].reindex(r.index)
            w0 = p0.to_numpy()
            w1 = p1.to_numpy()
            rv = r.to_numpy()

            mu0 = float(np.sum(w0 * rv) / max(float(np.sum(w0)), 1e-12))
            mu1 = float(np.sum(w1 * rv) / max(float(np.sum(w1)), 1e-12))
            state_up = 0 if mu0 > mu1 else 1

            prob_up = probs.iloc[:, state_up].copy()
            # EWMA smoothing
            prob_up = prob_up.ewm(alpha=float(alpha), adjust=False).mean()

            last_prob = float(prob_up.dropna().iloc[-1])
            last_state = int(last_prob > 0.5)
            outs[f"{ret_name}_{prob_name}"] = (last_prob, last_state)

    return outs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="Path to golden_features.parquet")
    ap.add_argument("--fixtures-dir", required=True, help="Dir containing SYMBOL_1D.parquet and SYMBOL_4H.parquet")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--max-ts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    golden = Path(args.golden).resolve()
    fixtures = Path(args.fixtures_dir).resolve()
    sym = args.symbol.upper()

    df = pd.read_parquet(golden)
    if "timestamp" not in df.columns:
        raise ValueError("golden parquet missing 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)

    need_cols = ["vol_prob_low_1d", "regime_code_1d", "markov_prob_up_4h", "markov_state_up_4h"]
    present = [c for c in need_cols if c in df.columns]
    if not present:
        raise ValueError(f"golden parquet missing all target cols: {need_cols}")

    # pick timestamps where at least one of the targets is present (not NaN)
    mask = False
    for c in present:
        mask = mask | df[c].notna()
    df2 = df.loc[mask].copy()
    if df2.empty:
        raise ValueError("no rows with non-null target cols in golden parquet")

    # sample evenly across time
    n = min(int(args.max_ts), len(df2))
    idxs = np.linspace(0, len(df2) - 1, n).astype(int)
    df_s = df2.iloc[idxs].copy()

    # Try to load EWMA alpha from config if available
    alpha = 0.2
    try:
        import config as cfg  # type: ignore
        alpha = float(getattr(cfg, "MARKOV4H_PROB_EWMA_ALPHA", alpha))
    except Exception:
        pass

    p1d = fixtures / f"{sym}_1D.parquet"
    p4h = fixtures / f"{sym}_4H.parquet"
    daily = _load_fixture_as_open_ts(p1d, "1D")
    h4 = _load_fixture_as_open_ts(p4h, "4h")

    # aggregate errors by variant
    daily_err: Dict[str, List[float]] = {}
    m4_err: Dict[str, List[float]] = {}
    m4_state_err: Dict[str, int] = {}

    for _, row in df_s.iterrows():
        ts = pd.Timestamp(row["timestamp"]).tz_convert("UTC")

        if "vol_prob_low_1d" in df.columns and pd.notna(row.get("vol_prob_low_1d", np.nan)):
            exp = float(row["vol_prob_low_1d"])
            outs = _daily_variants(daily, ts)
            for k, got in outs.items():
                daily_err.setdefault(k, []).append(abs(got - exp))

        if "markov_prob_up_4h" in df.columns and pd.notna(row.get("markov_prob_up_4h", np.nan)):
            exp_p = float(row["markov_prob_up_4h"])
            exp_s = int(row["markov_state_up_4h"]) if pd.notna(row.get("markov_state_up_4h", np.nan)) else None
            outs2 = _markov4h_variants(h4, ts, alpha=alpha)
            for k, (got_p, got_s) in outs2.items():
                m4_err.setdefault(k, []).append(abs(got_p - exp_p))
                if exp_s is not None:
                    m4_state_err[k] = m4_state_err.get(k, 0) + int(got_s != exp_s)

    print("=== Daily vol_prob_low_1d variant errors (lower is better) ===")
    for k, v in sorted(daily_err.items(), key=lambda kv: (np.mean(kv[1]), np.max(kv[1]))):
        print(f"{k:14s} n={len(v):3d} mean_abs={np.mean(v):.6f} max_abs={np.max(v):.6f}")

    print("\n=== Markov4h prob/state variant errors (lower is better) ===")
    for k, v in sorted(m4_err.items(), key=lambda kv: (np.mean(kv[1]), np.max(kv[1]))):
        se = m4_state_err.get(k, 0)
        print(f"{k:14s} n={len(v):3d} mean_abs={np.mean(v):.6f} max_abs={np.max(v):.6f} state_mismatch={se}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
