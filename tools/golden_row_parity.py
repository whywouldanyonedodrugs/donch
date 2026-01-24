#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    # tools/ -> repo root
    return Path(__file__).resolve().parents[1]


def _import_bundle_and_scorer():
    """
    Support both layouts:
      - package layout: donch/ (preferred)
      - flat layout: modules at repo root
    """
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Try package import first
    try:
        from donch.artifact_bundle import load_bundle  # type: ignore
        from donch.winprob_loader import WinProbScorer  # type: ignore
        from donch.artifact_bundle import SchemaError, BundleError  # type: ignore
        return load_bundle, WinProbScorer, SchemaError, BundleError
    except Exception:
        pass

    # Fallback: flat modules
    try:
        from artifact_bundle import load_bundle, SchemaError, BundleError  # type: ignore
        from winprob_loader import WinProbScorer  # type: ignore
        return load_bundle, WinProbScorer, SchemaError, BundleError
    except Exception as e:
        raise RuntimeError(
            "Cannot import bundle/scorer modules. Expected either donch/* package layout "
            "or flat modules at repo root."
        ) from e


def _read_manifest_feature_lists(bundle_dir: Path) -> Tuple[List[str], List[str]]:
    man_path = bundle_dir / "feature_manifest.json"
    obj = json.loads(man_path.read_text(encoding="utf-8"))
    feats = obj.get("features") if isinstance(obj, dict) else None
    if isinstance(feats, dict):
        num_cols = list(feats.get("numeric_cols") or feats.get("num_cols") or [])
        cat_cols = list(feats.get("cat_cols") or [])
        return num_cols, cat_cols

    # Accept schema-container formats too
    if isinstance(obj, dict):
        num_cols = list(obj.get("numeric_cols") or obj.get("num_cols") or [])
        cat_cols = list(obj.get("cat_cols") or [])
        return num_cols, cat_cols

    raise ValueError("Unsupported feature_manifest.json format")


def _auto_find_fixture(repo_root: Path) -> Optional[Path]:
    candidates = [
        repo_root / "results" / "meta_export" / "golden_rows.csv",
        repo_root / "results" / "golden_rows.csv",
        repo_root / "results" / "meta_export" / "golden_features.parquet",
        repo_root / "results" / "golden_features.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_fixture(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported fixture type: {path}")

    # Normalize symbol + timestamp
    if "symbol" not in df.columns:
        raise ValueError("Fixture must have column: symbol")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    ts_col = None
    for c in ("decision_ts", "timestamp", "ts"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("Fixture must have a timestamp column: decision_ts or timestamp or ts")

    df["_ts"] = pd.to_datetime(df[ts_col], utc=True, errors="raise")

    # Keep deterministic order
    df = df.sort_values(["symbol", "_ts"]).reset_index(drop=True)
    return df


def _pick_expected_prob_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Accept a few common column names.
    If not present, we still validate schema + compute probabilities.
    """
    raw_candidates = ["p_raw", "p_raw_offline", "prob_raw", "score_raw"]
    cal_candidates = ["p_cal", "p_cal_offline", "prob_cal", "prob", "wp", "win_probability"]

    p_raw_col = next((c for c in raw_candidates if c in df.columns), None)
    p_cal_col = next((c for c in cal_candidates if c in df.columns), None)
    return p_raw_col, p_cal_col


def _as_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Golden-row parity: manifest schema + scorer parity.")
    ap.add_argument("--bundle_dir", type=str, default="results/meta_export", help="Meta bundle directory")
    ap.add_argument("--fixture", type=str, default="", help="Path to golden_rows.csv or golden_features.parquet")
    ap.add_argument("--max_rows", type=int, default=0, help="Limit number of rows (0 = all)")
    ap.add_argument("--tol_num_abs", type=float, default=1e-10, help="Absolute tol for numeric feature comparisons (if expected/live provided)")
    ap.add_argument("--tol_num_rel", type=float, default=1e-9, help="Relative tol for numeric feature comparisons (if expected/live provided)")
    ap.add_argument("--tol_p", type=float, default=1e-10, help="Absolute tol for probability comparisons vs fixture (if present)")
    args = ap.parse_args()

    repo_root = _repo_root()
    bundle_dir = (repo_root / args.bundle_dir).resolve()
    if not bundle_dir.exists():
        print(f"ERROR: bundle_dir not found: {bundle_dir}", file=sys.stderr)
        return 2

    fixture_path = Path(args.fixture) if args.fixture else (_auto_find_fixture(repo_root) or Path(""))
    if not fixture_path or not fixture_path.exists():
        print("ERROR: fixture not provided and no default golden fixture found.", file=sys.stderr)
        print("Looked for:", file=sys.stderr)
        for p in [
            repo_root / "results" / "meta_export" / "golden_rows.csv",
            repo_root / "results" / "golden_rows.csv",
            repo_root / "results" / "meta_export" / "golden_features.parquet",
            repo_root / "results" / "golden_features.parquet",
        ]:
            print(f"  - {p}", file=sys.stderr)
        return 2
    fixture_path = fixture_path.resolve()

    load_bundle, WinProbScorer, SchemaError, BundleError = _import_bundle_and_scorer()

    # Load bundle + scorer (live scorer is authoritative here)
    bundle = load_bundle(bundle_dir, strict=True)
    scorer = WinProbScorer(bundle=bundle, strict_schema=True)

    # Manifest features (authoritative key set)
    num_cols, cat_cols = _read_manifest_feature_lists(bundle_dir)
    feat_cols = list(num_cols) + list(cat_cols)

    df = _load_fixture(fixture_path)
    if args.max_rows and args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()

    # Fixture must contain all manifest features (for golden-row parity gating)
    missing_feat_cols = [c for c in feat_cols if c not in df.columns]
    if missing_feat_cols:
        print(f"ERROR: fixture missing manifest feature columns (first 50): {missing_feat_cols[:50]}", file=sys.stderr)
        return 2

    p_raw_col, p_cal_col = _pick_expected_prob_cols(df)

    n = len(df)
    schema_fail = 0
    prob_fail = 0
    max_prob_diff = 0.0
    worst_prob_row = None

    for i, row in df.iterrows():
        raw_row: Dict[str, Any] = {}
        for c in feat_cols:
            raw_row[c] = row[c]

        # Validate + score
        try:
            p_raw, p_cal = scorer.score_with_details(raw_row)
        except SchemaError as e:
            schema_fail += 1
            print(f"[SCHEMA_FAIL] i={i} symbol={row['symbol']} ts={row['_ts'].isoformat()} err={e}")
            continue
        except BundleError as e:
            schema_fail += 1
            print(f"[SCORE_FAIL] i={i} symbol={row['symbol']} ts={row['_ts'].isoformat()} err={e}")
            continue
        except Exception as e:
            schema_fail += 1
            print(f"[ERROR] i={i} symbol={row['symbol']} ts={row['_ts'].isoformat()} err={e!r}")
            continue

        # Compare probabilities if fixture has them
        if p_raw_col:
            exp = _as_float(row[p_raw_col])
            if exp is not None:
                d = abs(float(p_raw) - exp)
                if d > args.tol_p:
                    prob_fail += 1
                    if d > max_prob_diff:
                        max_prob_diff = d
                        worst_prob_row = (i, row["symbol"], row["_ts"], "p_raw", exp, float(p_raw))

        if p_cal_col:
            exp = _as_float(row[p_cal_col])
            if exp is not None:
                d = abs(float(p_cal) - exp)
                if d > args.tol_p:
                    prob_fail += 1
                    if d > max_prob_diff:
                        max_prob_diff = d
                        worst_prob_row = (i, row["symbol"], row["_ts"], "p_cal", exp, float(p_cal))

    ok = (schema_fail == 0) and (prob_fail == 0)

    print("----- Golden parity summary -----")
    print(f"bundle_dir: {bundle_dir}")
    print(f"fixture:    {fixture_path}")
    print(f"rows:       {n}")
    print(f"schema_fail:{schema_fail}")
    print(f"prob_fail:  {prob_fail}")
    if worst_prob_row is not None:
        i, sym, ts, which, exp, got = worst_prob_row
        print(f"worst_prob: i={i} {sym} {ts.isoformat()} {which} exp={exp:.12f} got={got:.12f} |diff|={max_prob_diff:.12g}")
    print("--------------------------------")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
