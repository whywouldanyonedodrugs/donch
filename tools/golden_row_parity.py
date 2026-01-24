#!/usr/bin/env python3
"""
Golden row parity / bundle scoring sanity check.

- loads the meta bundle (feature_manifest + model + calibration)
- loads a fixture Parquet containing golden raw feature rows
- validates each row against the bundle's strict schema
- scores each row and (optionally) compares p_raw / p_cal to columns in the fixture, if present

Run:
  python tools/golden_row_parity.py --bundle_dir results/meta_export --fixture results/meta_export/golden_features.parquet
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Ensure repo root is importable when running as "python tools/..."
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, bool):
        # caller may pass bool-like values; convert to 0/1 numeric
        return 1.0 if x else 0.0

    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return 1.0
        if s in ("false", "f", "no", "n", "0"):
            return 0.0
        try:
            return float(x)
        except Exception:
            return None

    try:
        return float(x)
    except Exception:
        return None


def _as_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return str(x)


def _fmt(x: Any, nd: int = 6) -> str:
    try:
        if x is None:
            return "None"
        xf = float(x)
        if not math.isfinite(xf):
            if math.isnan(xf):
                return "nan"
            return "inf" if xf > 0 else "-inf"
        return f"{xf:.{nd}f}"
    except Exception:
        return repr(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", required=True, help="Path to exported meta bundle dir.")
    ap.add_argument("--fixture", required=True, help="Path to golden fixture parquet.")
    ap.add_argument("--max_rows", type=int, default=0, help="Limit rows processed (0 = all).")
    ap.add_argument("--symbol", default="", help="Optional symbol filter if fixture has a 'symbol' column.")
    ap.add_argument("--ts_col", default="", help="Optional timestamp column name to show in output.")
    ap.add_argument("--quiet", action="store_true", help="Only print summary.")
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    fixture_path = Path(args.fixture).resolve()
    if not bundle_dir.exists():
        print(f"ERROR: bundle_dir does not exist: {bundle_dir}")
        return 2
    if not fixture_path.exists():
        print(f"ERROR: fixture does not exist: {fixture_path}")
        return 2

    # Import from the same code-path as the service.
    from live.winprob_loader import WinProbScorer, SchemaError  # type: ignore

    try:
        scorer = WinProbScorer(str(bundle_dir))
    except Exception as e:
        print(f"ERROR: failed to initialize WinProbScorer from {bundle_dir}: {e!r}")
        return 2

    df = pd.read_parquet(fixture_path)
    if args.symbol and "symbol" in df.columns:
        df = df[df["symbol"].astype(str) == args.symbol].copy()

    if df.empty:
        print("No rows to process (empty fixture or filter removed everything).")
        return 0

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    raw_features = list(scorer.raw_features)
    spec_by_name = getattr(scorer, "_raw_spec_by_name", {})  # internal mapping (used by the service too)

    missing_cols = [c for c in raw_features if c not in df.columns]
    if missing_cols:
        print("ERROR: fixture missing required raw feature columns:")
        for c in missing_cols[:50]:
            print(f"  - {c}")
        if len(missing_cols) > 50:
            print(f"  ... ({len(missing_cols)-50} more)")
        return 2

    has_p_raw = "p_raw" in df.columns
    has_p_cal = "p_cal" in df.columns

    n = 0
    n_schema_ok = 0
    n_scored = 0
    n_p_raw_match = 0
    n_p_cal_match = 0
    schema_fail_examples = []
    score_fail_examples = []
    mismatch_examples = []

    tol = 1e-10

    for idx, row in df.iterrows():
        n += 1
        ts_val = row.get(args.ts_col) if args.ts_col else None
        sym_val = row.get("symbol") if "symbol" in df.columns else None

        raw_row: Dict[str, Any] = {}
        for k in raw_features:
            v = row[k]
            spec = spec_by_name.get(k)
            kind = getattr(spec, "kind", None) if spec is not None else None

            if kind == "categorical":
                raw_row[k] = _as_str(v)
            else:
                raw_row[k] = _as_float(v)

        try:
            details = scorer.score_with_details(raw_row)
            if details.get("schema_ok"):
                n_schema_ok += 1
                n_scored += 1

            p_raw = details.get("p_raw")
            p_cal = details.get("p_cal")

            if has_p_raw and details.get("schema_ok"):
                exp = _as_float(row["p_raw"])
                if exp is not None and p_raw is not None and abs(float(p_raw) - float(exp)) <= tol:
                    n_p_raw_match += 1
                elif exp is not None:
                    mismatch_examples.append(("p_raw", idx, sym_val, ts_val, exp, p_raw))

            if has_p_cal and details.get("schema_ok"):
                exp = _as_float(row["p_cal"])
                if exp is not None and p_cal is not None and abs(float(p_cal) - float(exp)) <= tol:
                    n_p_cal_match += 1
                elif exp is not None:
                    mismatch_examples.append(("p_cal", idx, sym_val, ts_val, exp, p_cal))

        except SchemaError as e:
            if len(schema_fail_examples) < 10:
                schema_fail_examples.append((idx, sym_val, ts_val, str(e)))
        except Exception as e:
            if len(score_fail_examples) < 10:
                score_fail_examples.append((idx, sym_val, ts_val, repr(e)))

    print(f"Rows processed: {n}")
    print(f"Schema OK:      {n_schema_ok}/{n}")
    print(f"Scored OK:      {n_scored}/{n}")
    if has_p_raw:
        print(f"p_raw matches:  {n_p_raw_match}/{n_schema_ok}")
    if has_p_cal:
        print(f"p_cal matches:  {n_p_cal_match}/{n_schema_ok}")

    if schema_fail_examples and not args.quiet:
        print("\nSchema failures (first 10):")
        for i, sym, ts, err in schema_fail_examples:
            where = f"idx={i}"
            if sym is not None:
                where += f" symbol={sym}"
            if ts is not None:
                where += f" ts={ts}"
            print(f"  {where}: {err}")

    if score_fail_examples and not args.quiet:
        print("\nScore failures (first 10):")
        for i, sym, ts, err in score_fail_examples:
            where = f"idx={i}"
            if sym is not None:
                where += f" symbol={sym}"
            if ts is not None:
                where += f" ts={ts}"
            print(f"  {where}: {err}")

    if mismatch_examples and not args.quiet:
        print("\nProbability mismatches (first 10):")
        for kind, i, sym, ts, exp, got in mismatch_examples[:10]:
            where = f"idx={i}"
            if sym is not None:
                where += f" symbol={sym}"
            if ts is not None:
                where += f" ts={ts}"
            print(f"  {where}: {kind} expected={_fmt(exp)} got={_fmt(got)}")

    return 1 if (schema_fail_examples or score_fail_examples or mismatch_examples) else 0


if __name__ == "__main__":
    raise SystemExit(main())
