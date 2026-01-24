#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Silence noisy sklearn feature-name warnings during batch scoring
warnings.filterwarnings(
    "ignore",
    message=r".*does not have valid feature names.*",
    category=UserWarning,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _is_na(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x)
    except Exception:
        return x is None


def _as_num(x: Any) -> Optional[float]:
    if _is_na(x):
        return None
    if isinstance(x, bool):
        # schema forbids bool for numeric; convert deterministically to float
        return 1.0 if x else 0.0
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        s = str(x).strip().lower()
        if s in ("true", "t", "yes", "y"):
            return 1.0
        if s in ("false", "f", "no", "n"):
            return 0.0
        return None


def _as_cat(x: Any) -> Optional[Any]:
    if _is_na(x):
        return None
    # preserve strings
    if isinstance(x, str):
        return x.strip()
    # if float looks like int, cast to int so str(v) becomes "1" not "1.0"
    if isinstance(x, float) and float(x).is_integer():
        return int(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, bool):
        return int(x)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--symbol", default="")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    args = ap.parse_args()

    from live.winprob_loader import WinProbScorer, SchemaError  # type: ignore

    bundle_dir = Path(args.bundle_dir).resolve()
    fixture = Path(args.fixture).resolve()

    scorer = WinProbScorer(str(bundle_dir))
    raw_feats = list(scorer.raw_features)
    spec_by = getattr(scorer, "_raw_spec_by_name", {})

    df = pd.read_parquet(fixture) if fixture.suffix.lower() in (".parquet", ".pq") else pd.read_csv(fixture)

    if args.symbol and "symbol" in df.columns:
        df = df[df["symbol"].astype(str) == args.symbol].copy()

    if df.empty:
        print("No rows to process.")
        return 0

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    missing = [c for c in raw_feats if c not in df.columns]
    if missing:
        print(f"ERROR: fixture missing {len(missing)} required raw feature columns. First 25:")
        for c in missing[:25]:
            print(f"  - {c}")
        return 2

    has_p_raw = "p_raw" in df.columns
    has_p_cal = "p_cal" in df.columns

    n = 0
    schema_ok = 0
    scored_ok = 0
    p_raw_match = 0
    p_cal_match = 0

    schema_fail = []
    score_fail = []
    mism = []

    for idx, row in df.iterrows():
        n += 1

        raw_row: Dict[str, Any] = {}
        for k in raw_feats:
            spec = spec_by.get(k)
            kind = getattr(spec, "kind", None) if spec is not None else None
            v = row[k]
            if kind == "categorical":
                raw_row[k] = _as_cat(v)
            else:
                raw_row[k] = _as_num(v)

        try:
            p_raw, p_cal = scorer.score_with_details(raw_row)
            schema_ok += 1
            scored_ok += 1

            if has_p_raw and not _is_na(row["p_raw"]):
                exp = float(row["p_raw"])
                if np.isclose(float(p_raw), exp, rtol=args.rtol, atol=args.atol):
                    p_raw_match += 1
                else:
                    mism.append(("p_raw", idx, exp, float(p_raw)))

            if has_p_cal and not _is_na(row["p_cal"]):
                exp = float(row["p_cal"])
                if np.isclose(float(p_cal), exp, rtol=args.rtol, atol=args.atol):
                    p_cal_match += 1
                else:
                    mism.append(("p_cal", idx, exp, float(p_cal)))

        except SchemaError as e:
            if len(schema_fail) < 10:
                schema_fail.append((idx, str(e)))
        except Exception as e:
            if len(score_fail) < 10:
                score_fail.append((idx, repr(e)))

    print(f"Rows processed: {n}")
    print(f"Schema OK:      {schema_ok}/{n}")
    print(f"Scored OK:      {scored_ok}/{n}")
    if has_p_raw:
        print(f"p_raw matches:  {p_raw_match}/{schema_ok}")
    if has_p_cal:
        print(f"p_cal matches:  {p_cal_match}/{schema_ok}")

    if schema_fail:
        print("\nSchema failures (first 10):")
        for i, err in schema_fail:
            print(f"  idx={i}: {err}")

    if score_fail:
        print("\nScore failures (first 10):")
        for i, err in score_fail:
            print(f"  idx={i}: {err}")

    if mism:
        print("\nProbability mismatches (first 10):")
        for kind, i, exp, got in mism[:10]:
            print(f"  idx={i}: {kind} expected={exp:.10f} got={got:.10f} diff={abs(got-exp):.10f}")

    return 0 if (not schema_fail and not score_fail and not mism) else 1


if __name__ == "__main__":
    raise SystemExit(main())
