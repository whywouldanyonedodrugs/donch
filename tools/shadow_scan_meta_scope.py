#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd


def _load_manifest_feature_list(manifest_path: Path) -> List[str]:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Unwrap {"schema": {...}}
    if isinstance(obj, dict) and isinstance(obj.get("schema"), dict):
        inner = obj["schema"]
        if any(k in inner for k in ("cat_cols", "num_cols", "numeric_cols")):
            obj = inner

    # Unwrap canonical {"features": {...}}
    if isinstance(obj, dict) and isinstance(obj.get("features"), dict):
        inner = obj["features"]
        if any(k in inner for k in ("cat_cols", "num_cols", "numeric_cols")):
            obj = inner

    if not (isinstance(obj, dict) and any(k in obj for k in ("cat_cols", "num_cols", "numeric_cols"))):
        raise RuntimeError(f"Unsupported feature_manifest.json format: top-level keys={list(obj)[:20]}")

    cat_cols = obj.get("cat_cols") or []
    num_cols = obj.get("numeric_cols")
    if num_cols is None:
        num_cols = obj.get("num_cols")
    num_cols = num_cols or []

    if not isinstance(num_cols, list) or not all(isinstance(x, str) for x in num_cols):
        raise RuntimeError("feature_manifest: numeric_cols/num_cols must be list[str]")
    if not isinstance(cat_cols, list) or not all(isinstance(x, str) for x in cat_cols):
        raise RuntimeError("feature_manifest: cat_cols must be list[str]")

    # Offline authoritative order is numeric_cols + cat_cols
    return [*num_cols, *cat_cols]


def _read_deployment_decision(bundle_dir: Path) -> Tuple[float, str]:
    dep = json.loads((bundle_dir / "deployment_config.json").read_text(encoding="utf-8"))
    decision = dep.get("decision") or {}
    pstar = float(decision.get("threshold"))
    scope = str(decision.get("scope") or "")
    return pstar, scope


def _eval_scope_mask(df: pd.DataFrame, scope: str) -> Tuple[pd.Series, pd.Series, str]:
    """
    Vectorized offline semantics:

    - If scope empty/None: all True
    - If scope == "RISK_ON_1":
        * use risk_on_1 if COLUMN EXISTS
        * else fallback to risk_on (only if risk_on_1 column absent)
        * coerce: pd.to_numeric(errors="coerce").fillna(0)
        * scope_ok = (val == 1)
        * if neither column exists: raise (offline errors)
    - Unknown scope: fail-closed (all False)
    """
    scope = (scope or "").strip()
    if scope == "":
        ok = pd.Series(True, index=df.index)
        val = pd.Series(np.nan, index=df.index)
        return ok, val, "none"

    if scope == "RISK_ON_1":
        if "risk_on_1" in df.columns:
            src = "risk_on_1"
            col = df["risk_on_1"]
        elif "risk_on" in df.columns:
            src = "risk_on"
            col = df["risk_on"]
        else:
            raise RuntimeError("Scope=RISK_ON_1 but neither risk_on_1 nor risk_on column exists (offline raises).")

        val = pd.to_numeric(col, errors="coerce").fillna(0)
        ok = (val == 1)
        return ok, val, src

    # Unknown scope string -> fail closed
    ok = pd.Series(False, index=df.index)
    val = pd.Series(np.nan, index=df.index)
    return ok, val, "unknown"


def _score_bundle(bundle_dir: Path, rows: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    model_path = bundle_dir / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Missing {model_path}")

    model = joblib.load(model_path)

    missing = [c for c in feature_cols if c not in rows.columns]
    if missing:
        raise RuntimeError(f"Rows are missing required feature columns (first 30): {missing[:30]}")

    X = rows[feature_cols]

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(f"Model does not support predict_proba(): type={type(model)}")

    proba = model.predict_proba(X)
    if not (isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2):
        raise RuntimeError(f"Unexpected predict_proba output shape: {getattr(proba, 'shape', None)}")

    p_raw = pd.Series(proba[:, 1], index=rows.index)

    cal_path = bundle_dir / "isotonic.joblib"
    if not cal_path.exists():
        # Allow uncalibrated if calibrator absent
        return p_raw

    calibrator = joblib.load(cal_path)

    arr = p_raw.to_numpy(dtype=float)
    if hasattr(calibrator, "transform"):
        p_cal = calibrator.transform(arr)
    elif hasattr(calibrator, "predict"):
        p_cal = calibrator.predict(arr)
    else:
        raise RuntimeError(f"Unsupported calibrator type={type(calibrator)} (no transform/predict)")

    return pd.Series(np.asarray(p_cal, dtype=float), index=rows.index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True, type=str, help="Meta bundle directory (contains model.joblib, isotonic.joblib, feature_manifest.json, deployment_config.json)")
    ap.add_argument("--rows-parquet", required=True, type=str, help="Parquet file containing feature rows at decision timestamps")
    ap.add_argument("--ts-col", default="decision_ts", type=str, help="Timestamp column name in rows parquet")
    ap.add_argument("--top", default=25, type=int, help="How many examples to print")
    ap.add_argument("--scope", default=None, type=str, help="Override scope (default: from deployment_config.json)")
    ap.add_argument("--pstar", default=None, type=float, help="Override p* threshold (default: from deployment_config.json)")
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir)
    rows_path = Path(args.rows_parquet)

    if not bundle_dir.exists():
        raise SystemExit(f"--bundle-dir not found: {bundle_dir}")
    if not rows_path.exists():
        raise SystemExit(f"--rows-parquet not found: {rows_path}")

    manifest_cols = _load_manifest_feature_list(bundle_dir / "feature_manifest.json")
    pstar0, scope0 = _read_deployment_decision(bundle_dir)

    pstar = float(args.pstar) if args.pstar is not None else float(pstar0)
    scope = str(args.scope) if args.scope is not None else str(scope0)

    df = pd.read_parquet(rows_path)

    if args.ts_col in df.columns:
        # keep original but also parse for sorting/printing
        df["_ts"] = pd.to_datetime(df[args.ts_col], utc=True, errors="coerce")
    else:
        df["_ts"] = pd.NaT

    scope_ok, scope_val, scope_src = _eval_scope_mask(df, scope)
    df["_scope_ok"] = scope_ok
    df["_scope_val"] = scope_val

    in_scope = df[df["_scope_ok"]].copy()
    if in_scope.empty:
        print(f"No in-scope rows found for scope={scope} (source={scope_src}).")
        print("If this is unexpected, confirm the rows parquet actually contains risk_on_1/risk_on values.")
        return 0

    p_cal = _score_bundle(bundle_dir, in_scope, manifest_cols)
    in_scope["_p_cal"] = p_cal
    in_scope["_above"] = (in_scope["_p_cal"] >= pstar)
    in_scope["_dist"] = (in_scope["_p_cal"] - pstar).abs()

    print(f"Bundle={bundle_dir} | scope={scope} (src={scope_src}) | p*={pstar:.6f}")
    print(f"In-scope rows: {len(in_scope)} / {len(df)}")

    # Print closest-to-threshold examples so you see both branches quickly
    closest = in_scope.sort_values(["_dist", "_ts"], ascending=[True, True]).head(int(args.top))

    cols_to_show = [c for c in [args.ts_col, "_ts", "symbol", "_scope_val", "_p_cal", "_above"] if c in closest.columns]
    if not cols_to_show:
        cols_to_show = ["_ts", "_scope_val", "_p_cal", "_above"]

    print("\nExamples (closest to threshold):")
    for _, r in closest.iterrows():
        parts = []
        if "_ts" in cols_to_show:
            parts.append(f"ts={r.get('_ts')}")
        if "symbol" in cols_to_show:
            parts.append(f"symbol={r.get('symbol')}")
        parts.append(f"scope_val={float(r.get('_scope_val')):.3f}")
        parts.append(f"p_cal={float(r.get('_p_cal')):.4f}")
        parts.append(f"p*={pstar:.4f}")
        parts.append("ABOVE" if bool(r.get("_above")) else "BELOW")
        print("  " + " ".join(parts))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
