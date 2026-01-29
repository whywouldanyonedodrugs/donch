#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from live.regime_features import (
    DailyRegimeConfig,
    Markov4hConfig,
    compute_daily_regime_series,
    compute_markov4h_series,
)

import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


def _infer_ewma_alpha(prob_raw: pd.Series, prob_ewm: pd.Series) -> float | None:
    """
    Infer alpha from ewm recursion:
      ewm_t = alpha * x_t + (1-alpha) * ewm_{t-1}
      alpha = (ewm_t - ewm_{t-1}) / (x_t - ewm_{t-1})
    """
    x = prob_raw.astype(float)
    y = prob_ewm.astype(float)
    y_prev = y.shift(1)
    denom = (x - y_prev)
    num = (y - y_prev)
    a = num / denom
    a = a.replace([np.inf, -np.inf], np.nan).dropna()
    # keep plausible alphas
    a = a[(a > 0) & (a < 1)]
    if len(a) == 0:
        return None
    return float(a.median())


def _mae(a: pd.Series, b: pd.Series) -> float:
    x = a.astype(float)
    y = b.astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(x[m] - y[m])))


def _acc_int(a: pd.Series, b: pd.Series) -> float:
    x = a.astype("Int64")
    y = b.astype("Int64")
    m = x.notna() & y.notna()
    if int(m.sum()) == 0:
        return float("nan")
    return float((x[m].astype(int).values == y[m].astype(int).values).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intermediate-dir", required=True)
    ap.add_argument("--fixtures-dir", required=True)
    ap.add_argument("--symbol", required=True)

    ap.add_argument("--ma-period", type=int, default=200)
    ap.add_argument("--atr-period", type=int, default=20)
    ap.add_argument("--atr-mult", type=float, default=2.0)
    ap.add_argument("--maxiter", type=int, default=200)

    ap.add_argument("--ewma-alpha", type=float, default=None)

    # SHIFT: OFFLINE intermediates are left-labeled; LIVE fixtures are bar-close labeled.
    # To compare, shift intermediate timestamps forward to bar-close time.
    ap.add_argument("--shift-daily", type=str, default="1D")   # +1D
    ap.add_argument("--shift-4h", type=str, default="4H")      # +4H
    ap.add_argument("--no-shift-intermediate", action="store_true")

    args = ap.parse_args()

    inter_dir = Path(args.intermediate_dir)
    fix_dir = Path(args.fixtures_dir)

    daily_inter_p = inter_dir / f"{args.symbol}_daily_regime_intermediate.parquet"
    markov_inter_p = inter_dir / f"{args.symbol}_markov4h_intermediate.parquet"

    daily_fix_p = fix_dir / f"{args.symbol}_1D.parquet"
    markov_fix_p = fix_dir / f"{args.symbol}_4H.parquet"

    if not daily_inter_p.exists():
        raise SystemExit(f"Missing daily intermediate: {daily_inter_p}")
    if not markov_inter_p.exists():
        raise SystemExit(f"Missing 4h intermediate: {markov_inter_p}")
    if not daily_fix_p.exists():
        raise SystemExit(f"Missing daily fixture: {daily_fix_p}")
    if not markov_fix_p.exists():
        raise SystemExit(f"Missing 4h fixture: {markov_fix_p}")

    daily_inter = _ensure_utc_index(pd.read_parquet(daily_inter_p), "timestamp")
    markov_inter = _ensure_utc_index(pd.read_parquet(markov_inter_p), "timestamp")

    daily_fix = _ensure_utc_index(pd.read_parquet(daily_fix_p), "timestamp")
    markov_fix = _ensure_utc_index(pd.read_parquet(markov_fix_p), "timestamp")

    if not args.no_shift_intermediate:
        daily_inter = daily_inter.copy()
        markov_inter = markov_inter.copy()
        daily_inter.index = daily_inter.index + pd.Timedelta(args.shift_daily)
        markov_inter.index = markov_inter.index + pd.Timedelta(args.shift_4h)

    # EWMA alpha: if not provided, infer from intermediate columns if present.
    ewma_alpha = args.ewma_alpha
    if ewma_alpha is None:
        if "prob_up_raw" in markov_inter.columns and "prob_up_ewm" in markov_inter.columns:
            inf = _infer_ewma_alpha(markov_inter["prob_up_raw"], markov_inter["prob_up_ewm"])
            if inf is not None:
                ewma_alpha = inf
    if ewma_alpha is None:
        ewma_alpha = 0.2

    dcfg = DailyRegimeConfig(
        ma_period=int(args.ma_period),
        atr_period=int(args.atr_period),
        atr_mult=float(args.atr_mult),
        maxiter=int(args.maxiter),
    )
    mcfg = Markov4hConfig(
        ewma_alpha=float(ewma_alpha),
        maxiter=int(args.maxiter),
    )

    daily_live = compute_daily_regime_series(daily_fix, dcfg)
    markov_live = compute_markov4h_series(markov_fix, mcfg)

    # Align on overlapping timestamps
    daily_idx = daily_live.index.intersection(daily_inter.index)
    markov_idx = markov_live.index.intersection(markov_inter.index)

    if len(daily_idx) == 0:
        raise SystemExit("No overlapping timestamps for DAILY between live series and intermediates.")
    if len(markov_idx) == 0:
        raise SystemExit("No overlapping timestamps for MARKOV4H between live series and intermediates.")

    dl = daily_live.loc[daily_idx]
    di = daily_inter.loc[daily_idx]

    ml = markov_live.loc[markov_idx]
    mi = markov_inter.loc[markov_idx]

    # DAILY: compare regime_code + vol_prob_low
    daily_prob_col = "vol_prob_low"
    daily_prob_exp = "vol_prob_low"
    daily_code_col = "regime_code"
    daily_code_exp = "regime_code"

    daily_prob_mae = _mae(di[daily_prob_exp], dl[daily_prob_col])
    daily_prob_max = float(np.nanmax(np.abs(di[daily_prob_exp].astype(float) - dl[daily_prob_col].astype(float))))
    daily_code_acc = _acc_int(di[daily_code_exp], dl[daily_code_col])

    # MARKOV: compare prob_up + state_up
    markov_prob_col = "prob_up"
    markov_prob_exp = "prob_up_ewm" if "prob_up_ewm" in mi.columns else "prob_up"
    markov_state_col = "state_up"
    markov_state_exp = "state_up"

    markov_prob_mae = _mae(mi[markov_prob_exp], ml[markov_prob_col])
    markov_prob_max = float(np.nanmax(np.abs(mi[markov_prob_exp].astype(float) - ml[markov_prob_col].astype(float))))
    markov_state_acc = _acc_int(mi[markov_state_exp], ml[markov_state_col])

    print(f"EWMA alpha used: {ewma_alpha}")
    print(f"DAILY:   n={len(daily_idx)} prob_mae={daily_prob_mae:.6f} prob_max={daily_prob_max:.6f} code_acc={daily_code_acc:.4f}")
    print(f"MARKOV4H:n={len(markov_idx)} prob_mae={markov_prob_mae:.6f} prob_max={markov_prob_max:.6f} state_acc={markov_state_acc:.4f}")

    print("\nSANITY timestamps:")
    print("daily_fixture:", daily_fix.index.min(), "->", daily_fix.index.max())
    print("daily_inter  :", daily_inter.index.min(), "->", daily_inter.index.max())
    print("4h_fixture   :", markov_fix.index.min(), "->", markov_fix.index.max())
    print("4h_inter     :", markov_inter.index.min(), "->", markov_inter.index.max())


if __name__ == "__main__":
    main()
