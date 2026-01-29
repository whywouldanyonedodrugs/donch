#!/usr/bin/env python3
"""
diag_golden_regime_vs_intermediates.py

Diagnose which benchmark symbol (BTCUSDT vs ETHUSDT, etc.) the golden "macro" regime
columns were derived from by comparing golden_features.parquet against OFFLINE
intermediate exports.

Semantics:
- Golden timestamps are decision_ts (5m bar-close timestamps).
- OFFLINE intermediates are left-edge labeled (default pandas resample), so to compare
  as-of decision_ts using bar-close labeling, we shift:
    - daily intermediates by +1D
    - 4H intermediates by +4h
  (unless --no-shift-intermediate is used)
- As-of lookup: for each decision_ts, use the last intermediate row with ts <= decision_ts.
No wall-clock leakage. Deterministic as-of-only.

This tool does not recompute regimes; it only compares golden vs intermediates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
bootstrap_repo_root()

# ---------------------------------------------------------------------
# Bootstrap: allow running as `python tools/diag_...py` from anywhere
# without relying on PYTHONPATH hacks.
# ---------------------------------------------------------------------
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


def _to_num(x) -> float:
    """
    Robust scalar->float coercion.
    Handles pd.NA, strings, objects. Non-numeric -> np.nan.
    """
    try:
        if x is None:
            return float("nan")
        # pd.NA / NaT
        if pd.isna(x):
            return float("nan")
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _load_golden(path: Path) -> pd.DataFrame:
    g = pd.read_parquet(path)
    g = _ensure_utc_index(g, "timestamp")
    # Golden macro columns are constant across symbols per timestamp; pick first row per ts.
    g = g.groupby(g.index).first().sort_index()
    return g


def _asof_row(df: pd.DataFrame, ts: pd.Timestamp) -> Optional[pd.Series]:
    ts = pd.to_datetime(ts, utc=True)
    if df.empty:
        return None
    pos = df.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return df.iloc[int(pos)]


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _eval_daily(golden: pd.DataFrame, daily_inter: pd.DataFrame, limit: int) -> Dict[str, Any]:
    exp_code_c = _pick_col(golden, ["regime_code_1d", "regime_code"])
    exp_prob_c = _pick_col(golden, ["vol_prob_low_1d", "vol_prob_low"])

    got_code_c = _pick_col(daily_inter, ["regime_code", "regime_code_1d"])
    got_prob_c = _pick_col(daily_inter, ["vol_prob_low", "vol_prob_low_1d"])

    if exp_code_c is None:
        raise AssertionError("Golden missing daily regime code column (expected regime_code_1d).")
    if exp_prob_c is None:
        raise AssertionError("Golden missing daily vol prob column (expected vol_prob_low_1d).")
    if got_code_c is None:
        raise AssertionError("Daily intermediate missing regime code column (expected regime_code).")
    if got_prob_c is None:
        raise AssertionError("Daily intermediate missing vol prob column (expected vol_prob_low).")

    n = 0
    code_ok = 0
    prob_errs = []
    prob_max = 0.0
    mism = []

    for ts, row in golden.iterrows():
        if n >= limit:
            break

        exp_code = _to_num(row.get(exp_code_c, np.nan))
        exp_prob = _to_num(row.get(exp_prob_c, np.nan))

        # skip rows where both are missing/non-numeric
        if pd.isna(exp_code) and pd.isna(exp_prob):
            continue

        got = _asof_row(daily_inter, ts)
        if got is None:
            continue

        got_code = _to_num(got.get(got_code_c, np.nan))
        got_prob = _to_num(got.get(got_prob_c, np.nan))

        if not pd.isna(exp_code) and not pd.isna(got_code):
            if int(exp_code) == int(got_code):
                code_ok += 1
            else:
                mism.append(f"{ts} code exp={int(exp_code)} got={int(got_code)}")

        if not pd.isna(exp_prob) and not pd.isna(got_prob):
            e = float(abs(float(exp_prob) - float(got_prob)))
            prob_errs.append(e)
            prob_max = max(prob_max, e)
            if e > 1e-3:
                mism.append(
                    f"{ts} vol_prob_low exp={float(exp_prob):.6f} got={float(got_prob):.6f} err={e:.6f}"
                )

        n += 1

    prob_mae = float(np.mean(prob_errs)) if prob_errs else float("nan")
    code_acc = float(code_ok / max(n, 1)) if n else float("nan")

    return {
        "n": n,
        "code_acc": code_acc,
        "prob_mae": prob_mae,
        "prob_max": prob_max,
        "mismatches_sample": mism[:10],
    }


def _eval_markov(golden: pd.DataFrame, markov_inter: pd.DataFrame, limit: int) -> Dict[str, Any]:
    exp_p_c = _pick_col(golden, ["markov_prob_up_4h", "prob_up"])
    exp_s_c = _pick_col(golden, ["markov_state_4h", "state_up"])

    got_p_c = _pick_col(markov_inter, ["prob_up_ewm", "prob_up"])
    got_s_c = _pick_col(markov_inter, ["state_up", "markov_state_4h"])

    if exp_p_c is None or exp_s_c is None:
        raise AssertionError("Golden missing markov columns (expected markov_prob_up_4h and markov_state_4h).")
    if got_p_c is None or got_s_c is None:
        raise AssertionError("Markov intermediate missing columns (expected prob_up/prob_up_ewm and state_up).")

    n = 0
    state_ok = 0
    prob_errs = []
    prob_max = 0.0
    mism = []

    for ts, row in golden.iterrows():
        if n >= limit:
            break

        exp_p = _to_num(row.get(exp_p_c, np.nan))
        exp_s = _to_num(row.get(exp_s_c, np.nan))

        if pd.isna(exp_p) and pd.isna(exp_s):
            continue

        got = _asof_row(markov_inter, ts)
        if got is None:
            continue

        got_p = _to_num(got.get(got_p_c, np.nan))
        got_s = _to_num(got.get(got_s_c, np.nan))

        if not pd.isna(exp_p) and not pd.isna(got_p):
            e = float(abs(float(exp_p) - float(got_p)))
            prob_errs.append(e)
            prob_max = max(prob_max, e)
            if e > 5e-3:
                mism.append(f"{ts} prob exp={float(exp_p):.6f} got={float(got_p):.6f} err={e:.6f}")

        if not pd.isna(exp_s) and not pd.isna(got_s):
            if int(exp_s) == int(got_s):
                state_ok += 1
            else:
                mism.append(f"{ts} state exp={int(exp_s)} got={int(got_s)}")

        n += 1

    prob_mae = float(np.mean(prob_errs)) if prob_errs else float("nan")
    state_acc = float(state_ok / max(n, 1)) if n else float("nan")

    return {
        "n": n,
        "state_acc": state_acc,
        "prob_mae": prob_mae,
        "prob_max": prob_max,
        "mismatches_sample": mism[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-path", type=str, default="results/meta_export/golden_features.parquet")
    ap.add_argument("--intermediate-dir", type=str, required=True)
    ap.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT", help="Comma-separated symbols to compare.")
    ap.add_argument("--no-shift-intermediate", action="store_true")
    ap.add_argument("--shift-daily", type=str, default="1D")
    ap.add_argument("--shift-4h", type=str, default="4h")  # 'H' deprecated
    ap.add_argument("--limit", type=int, default=1500, help="Max golden decision_ts to evaluate per symbol.")
    args = ap.parse_args()

    golden_path = (_REPO_ROOT / args.golden_path).resolve()
    inter_dir = (_REPO_ROOT / args.intermediate_dir).resolve()

    if not golden_path.exists():
        raise SystemExit(f"Missing golden: {golden_path}")
    if not inter_dir.exists():
        raise SystemExit(f"Missing intermediate dir: {inter_dir}")

    golden = _load_golden(golden_path)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    shift_daily = pd.Timedelta(args.shift_daily)
    shift_4h = pd.Timedelta(args.shift_4h)

    daily_rank: List[Tuple[float, str, Dict[str, Any]]] = []
    markov_rank: List[Tuple[float, str, Dict[str, Any]]] = []

    for sym in symbols:
        daily_p = inter_dir / f"{sym}_daily_regime_intermediate.parquet"
        markov_p = inter_dir / f"{sym}_markov4h_intermediate.parquet"

        if not daily_p.exists():
            print(f"[SKIP] {sym} daily missing: {daily_p}")
            continue
        if not markov_p.exists():
            print(f"[SKIP] {sym} markov missing: {markov_p}")
            continue

        daily_inter = _ensure_utc_index(pd.read_parquet(daily_p), "timestamp")
        markov_inter = _ensure_utc_index(pd.read_parquet(markov_p), "timestamp")

        if not args.no_shift_intermediate:
            daily_inter = daily_inter.copy()
            markov_inter = markov_inter.copy()
            daily_inter.index = daily_inter.index + shift_daily
            markov_inter.index = markov_inter.index + shift_4h

        d = _eval_daily(golden, daily_inter, limit=int(args.limit))
        m = _eval_markov(golden, markov_inter, limit=int(args.limit))

        # Ranking scores: lower is better
        d_score = float(d["prob_mae"]) + 10.0 * (1.0 - float(d["code_acc"]))
        m_score = float(m["prob_mae"]) + 1.0 * (1.0 - float(m["state_acc"]))

        daily_rank.append((d_score, sym, d))
        markov_rank.append((m_score, sym, m))

        print(f"\n=== {sym} ===")
        print(f"DAILY   n={d['n']} code_acc={d['code_acc']:.6f} prob_mae={d['prob_mae']:.6f} prob_max={d['prob_max']:.6f}")
        for s in d["mismatches_sample"]:
            print(f"  {s}")
        print(f"MARKOV  n={m['n']} state_acc={m['state_acc']:.6f} prob_mae={m['prob_mae']:.6f} prob_max={m['prob_max']:.6f}")
        for s in m["mismatches_sample"]:
            print(f"  {s}")
        print(f"Scores: daily={d_score:.6f} markov={m_score:.6f}")

    if daily_rank:
        daily_rank.sort(key=lambda x: x[0])
        best = daily_rank[0]
        print(f"\nBEST DAILY match: {best[1]} (score={best[0]:.6f})")
    else:
        print("\nNo DAILY comparisons were run (missing files).")

    if markov_rank:
        markov_rank.sort(key=lambda x: x[0])
        best = markov_rank[0]
        print(f"BEST MARKOV match: {best[1]} (score={best[0]:.6f})")
    else:
        print("No MARKOV comparisons were run (missing files).")


if __name__ == "__main__":
    main()
