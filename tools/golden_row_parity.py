#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_syspath(repo_root: Path) -> None:
    s = str(repo_root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_fixture(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported fixture format: {path.suffix} (use .parquet or .csv)")


def _pick_ts_col(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f"--ts_col={explicit} not found in fixture columns")
        return explicit
    for c in ("decision_ts", "asof_ts", "ts", "timestamp", "time", "datetime", "entry_ts"):
        if c in df.columns:
            return c
    raise KeyError("Could not infer timestamp column. Provide --ts_col.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Golden-row parity: validate bundle scorer vs golden fixture outputs.")
    ap.add_argument("--bundle_dir", required=True, help="Bundle directory (e.g., results/meta_export)")
    ap.add_argument("--fixture", required=True, help="Golden fixture (.parquet or .csv)")
    ap.add_argument("--max_rows", type=int, default=2000, help="Max rows to check")
    ap.add_argument("--ts_col", default=None, help="Timestamp column name in fixture (optional)")
    ap.add_argument("--symbol_col", default="symbol", help="Symbol column name in fixture")
    ap.add_argument("--p_raw_col", default="p_raw", help="Expected raw prob col (optional)")
    ap.add_argument("--p_cal_col", default="p_cal", help="Expected calibrated prob col (optional)")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    args = ap.parse_args()

    repo_root = _repo_root()
    _ensure_repo_on_syspath(repo_root)

    # Import in proper package context (fixes relative imports inside live/*)
    from live.artifact_bundle import BundleError, SchemaError, load_bundle  # type: ignore
    from live.winprob_loader import WinProbScorer  # type: ignore

    bundle_dir = Path(args.bundle_dir).resolve()
    fixture_path = Path(args.fixture).resolve()

    df = _load_fixture(fixture_path)
    if df.empty:
        raise RuntimeError("Fixture dataframe is empty.")

    if args.symbol_col not in df.columns:
        raise KeyError(f"Fixture missing required column: {args.symbol_col}")

    ts_col = _pick_ts_col(df, args.ts_col)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])

    bundle = load_bundle(str(bundle_dir))
    scorer = WinProbScorer(bundle=bundle, strict_schema=True)

    required = list(getattr(scorer, "raw_features", []) or [])
    if not required:
        raise RuntimeError("scorer.raw_features is empty")

    numeric_cols = set(getattr(scorer, "numeric_cols", []) or [])
    cat_cols = set(getattr(scorer, "cat_cols", []) or [])

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Fixture missing {len(missing_cols)} required feature columns. First 10: {missing_cols[:10]}")

    has_p_raw = args.p_raw_col in df.columns
    has_p_cal = args.p_cal_col in df.columns

    n = min(int(args.max_rows), len(df))
    sub = df.iloc[:n].copy()

    schema_fail = 0
    prob_fail = 0
    max_abs_raw = 0.0
    max_abs_cal = 0.0

    for idx, row in sub.iterrows():
        raw_row: Dict[str, Any] = {}
        for k in required:
            v = row[k]
            if pd.isna(v):
                raw_row[k] = None
                continue
            if k in cat_cols:
                # categorical codes must be ints (or None)
                raw_row[k] = int(v)
            else:
                raw_row[k] = float(v)

        try:
            p_raw, p_cal = scorer.score_with_details(raw_row)  # returns (p_raw, p_cal)
        except (SchemaError, BundleError, Exception) as e:
            schema_fail += 1
            print(f"[SCHEMA_FAIL] idx={idx} ts={row[ts_col]} err={type(e).__name__}:{e}", flush=True)
            continue

        if has_p_raw and (not pd.isna(row[args.p_raw_col])):
            exp = float(row[args.p_raw_col])
            got = float(p_raw)
            diff = abs(got - exp)
            max_abs_raw = max(max_abs_raw, diff)
            if not np.isclose(got, exp, rtol=args.rtol, atol=args.atol):
                prob_fail += 1
                print(f"[P_RAW_MISMATCH] idx={idx} ts={row[ts_col]} got={got:.10f} exp={exp:.10f} diff={diff:.10f}", flush=True)

        if has_p_cal and (not pd.isna(row[args.p_cal_col])):
            exp = float(row[args.p_cal_col])
            got = float(p_cal)
            diff = abs(got - exp)
            max_abs_cal = max(max_abs_cal, diff)
            if not np.isclose(got, exp, rtol=args.rtol, atol=args.atol):
                prob_fail += 1
                print(f"[P_CAL_MISMATCH] idx={idx} ts={row[ts_col]} got={got:.10f} exp={exp:.10f} diff={diff:.10f}", flush=True)

    print(
        f"Checked rows={n} schema_fail={schema_fail} prob_fail={prob_fail} "
        f"max_abs_raw={max_abs_raw:.10f} max_abs_cal={max_abs_cal:.10f}",
        flush=True,
    )
    return 0 if (schema_fail == 0 and prob_fail == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
