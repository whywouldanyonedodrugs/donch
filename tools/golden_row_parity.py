#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    # tools/golden_row_parity.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_syspath(repo_root: Path) -> None:
    root_s = str(repo_root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def _load_fixture(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported fixture format: {path.suffix} (use .parquet or .csv)")


def _pick_ts_col(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f"--ts_col={explicit} not found in fixture columns")
        return explicit
    for c in ["decision_ts", "asof_ts", "ts", "timestamp", "time", "datetime"]:
        if c in df.columns:
            return c
    raise KeyError("Could not infer timestamp column. Provide --ts_col.")


def _read_manifest_schema(bundle_dir: Path) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns (feature_names, dtype_map).
    Robust to multiple manifest layouts and dtype representations (string/list/dict).
    """
    mpath = bundle_dir / "feature_manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"Missing feature_manifest.json in bundle_dir: {mpath}")

    obj = json.loads(mpath.read_text())

    # Case A: schema-like mapping name -> dtype/kind
    if isinstance(obj, dict):
        # Common pattern: {"schema": {"feat": "numeric", ...}} or similar
        for k in ("schema", "columns"):
            if k in obj and isinstance(obj[k], dict):
                names = list(obj[k].keys())
                dtype_map = {str(n): str(obj[k][n]) for n in names}
                return names, dtype_map

        # Common pattern: {"numeric_cols":[...], "cat_cols":[...]}
        if "numeric_cols" in obj or "cat_cols" in obj:
            num = list(obj.get("numeric_cols", []) or [])
            cat = list(obj.get("cat_cols", []) or [])
            names = [str(x) for x in (num + cat)]
            dtype_map = {str(x): "numeric" for x in num}
            dtype_map.update({str(x): "categorical" for x in cat})
            return names, dtype_map

    # Case B: list of feature dicts, or dict containing such a list
    feats: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        feats = [x for x in obj if isinstance(x, dict)]
    elif isinstance(obj, dict):
        for k in ("raw_features", "features", "manifest"):
            if k in obj and isinstance(obj[k], list):
                feats = [x for x in obj[k] if isinstance(x, dict)]
                break

    if not feats:
        raise RuntimeError("Could not parse feature_manifest.json into a feature list.")

    names: List[str] = []
    dtype_map: Dict[str, str] = {}

    def _dtype_to_str(v: Any) -> str:
        # Prefer string kinds; tolerate list/dict
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, list):
            # If list of one string, use it; otherwise just label as list
            if len(v) == 1 and isinstance(v[0], str):
                return v[0].strip()
            return "list"
        if isinstance(v, dict):
            # Try common keys
            for kk in ("kind", "type", "dtype"):
                if kk in v:
                    return _dtype_to_str(v[kk])
            return "dict"
        return str(v).strip()

    for f in feats:
        name = f.get("name") or f.get("key")
        if not name:
            continue

        # IMPORTANT: prefer 'kind' first so we don't accidentally take a list-valued 'dtype'
        dt_val = f.get("kind") or f.get("type") or f.get("dtype")
        dt = _dtype_to_str(dt_val)

        n = str(name)
        names.append(n)
        dtype_map[n] = dt

    # Preserve order, drop duplicates
    seen = set()
    ordered: List[str] = []
    for n in names:
        if n not in seen:
            ordered.append(n)
            seen.add(n)

    return ordered, dtype_map


def _is_cat(dtype_str: str) -> bool:
    s = (dtype_str or "").lower()
    return ("cat" in s) or (s in {"categorical", "category"})


def _format_float(x: Optional[float]) -> str:
    if x is None:
        return "None"
    try:
        return f"{float(x):.10f}"
    except Exception:
        return str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description="Golden-row parity: validate bundle scorer vs golden fixture outputs.")
    ap.add_argument("--bundle_dir", required=True, help="Bundle directory (e.g., results/meta_export)")
    ap.add_argument("--fixture", required=True, help="Golden fixture (.parquet or .csv)")
    ap.add_argument("--max_rows", type=int, default=2000, help="Max rows to check")
    ap.add_argument("--ts_col", default=None, help="Timestamp column name in fixture (optional)")
    ap.add_argument("--p_raw_col", default="p_raw", help="Column name for expected raw prob (if present)")
    ap.add_argument("--p_cal_col", default="p_cal", help="Column name for expected calibrated prob (if present)")
    ap.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for prob comparison")
    ap.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for prob comparison")
    ap.add_argument("--require_prob_cols", action="store_true", help="Fail if expected prob columns are missing")
    args = ap.parse_args()

    repo_root = _repo_root()
    _ensure_repo_on_syspath(repo_root)

    bundle_dir = Path(args.bundle_dir).resolve()
    fixture_path = Path(args.fixture).resolve()

    # Import in proper package context (fixes relative imports inside live/*)
    from live.artifact_bundle import BundleError, SchemaError, load_bundle  # type: ignore
    from live.winprob_loader import WinProbScorer  # type: ignore

    df = _load_fixture(fixture_path)
    if df.empty:
        raise RuntimeError("Fixture dataframe is empty.")

    ts_col = _pick_ts_col(df, args.ts_col)

    # Load schema from manifest for column selection
    feat_names, dtype_map = _read_manifest_schema(bundle_dir)

    missing_cols = [c for c in feat_names if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Fixture missing {len(missing_cols)} required feature columns. First 10: {missing_cols[:10]}")

    has_p_raw = args.p_raw_col in df.columns
    has_p_cal = args.p_cal_col in df.columns
    if args.require_prob_cols and (not has_p_raw or not has_p_cal):
        raise RuntimeError(
            f"Expected prob columns missing: have_p_raw={has_p_raw} have_p_cal={has_p_cal} "
            f"(expected {args.p_raw_col}, {args.p_cal_col})"
        )

    bundle = load_bundle(str(bundle_dir))
    scorer = WinProbScorer(bundle=bundle, strict_schema=True)

    n = min(int(args.max_rows), len(df))
    sub = df.iloc[:n].copy()

    # Track failures
    prob_fail = 0
    schema_fail = 0
    max_abs_raw = 0.0
    max_abs_cal = 0.0

    for i, row in sub.iterrows():
        # Build raw row dict using manifest order
        raw_row: Dict[str, Any] = {}
        for k in feat_names:
            v = row[k]
            # Preserve categorical codes as-is; numerics as python floats when possible
            if _is_cat(dtype_map.get(k, "")):
                raw_row[k] = (None if pd.isna(v) else int(v))
            else:
                raw_row[k] = (None if pd.isna(v) else float(v))

        # Score
        try:
            p_raw, p_cal = scorer.score_with_details(raw_row)
        except (SchemaError, BundleError, Exception) as e:
            schema_fail += 1
            tsv = row[ts_col]
            print(f"[SCHEMA_FAIL] idx={i} ts={tsv} err={e}")
            continue

        # Compare to expected if present
        if has_p_raw:
            exp_raw = None if pd.isna(row[args.p_raw_col]) else float(row[args.p_raw_col])
            if exp_raw is not None:
                diff = abs(float(p_raw) - exp_raw)
                max_abs_raw = max(max_abs_raw, diff)
                if not np.isclose(float(p_raw), exp_raw, rtol=args.rtol, atol=args.atol):
                    prob_fail += 1
                    print(
                        f"[P_RAW_MISMATCH] idx={i} ts={row[ts_col]} got={_format_float(p_raw)} exp={_format_float(exp_raw)} diff={diff:.10f}"
                    )

        if has_p_cal:
            exp_cal = None if pd.isna(row[args.p_cal_col]) else float(row[args.p_cal_col])
            if exp_cal is not None:
                diff = abs(float(p_cal) - exp_cal)
                max_abs_cal = max(max_abs_cal, diff)
                if not np.isclose(float(p_cal), exp_cal, rtol=args.rtol, atol=args.atol):
                    prob_fail += 1
                    print(
                        f"[P_CAL_MISMATCH] idx={i} ts={row[ts_col]} got={_format_float(p_cal)} exp={_format_float(exp_cal)} diff={diff:.10f}"
                    )

    print(
        f"Checked rows={n} schema_fail={schema_fail} prob_fail={prob_fail} "
        f"max_abs_raw={max_abs_raw:.10f} max_abs_cal={max_abs_cal:.10f}"
    )

    # Exit code policy
    if schema_fail > 0 or prob_fail > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
