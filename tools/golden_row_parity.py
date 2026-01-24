#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_module_from_path(module_name, path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec for {module_name} from {path}")

    mod = importlib.util.module_from_spec(spec)

    # CRITICAL: dataclasses (and other machinery) expects the module to exist in sys.modules
    # before class decorators run.
    sys.modules[module_name] = mod

    spec.loader.exec_module(mod)
    return mod


def find_required_files(root: Path) -> Tuple[Path, Path]:
    """
    Find artifact_bundle.py and winprob_loader.py anywhere under repo.
    Deterministic: first match in sorted path order.
    """
    ab = sorted(root.rglob("artifact_bundle.py"))
    wp = sorted(root.rglob("winprob_loader.py"))
    if not ab:
        raise RuntimeError("Could not find artifact_bundle.py under repo")
    if not wp:
        raise RuntimeError("Could not find winprob_loader.py under repo")
    return ab[0], wp[0]


def pick_ts_col(df: pd.DataFrame) -> str:
    for c in ("timestamp", "decision_ts", "entry_ts", "ts"):
        if c in df.columns:
            return c
    raise ValueError("Fixture must contain a timestamp column: timestamp|decision_ts|entry_ts|ts")


def pick_expected_prob_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    raw_cands = ("p_raw", "wp_raw", "prob_raw", "score_raw")
    cal_cands = ("p_cal", "wp_cal", "prob_cal", "score_cal", "p", "wp", "win_probability")
    raw = next((c for c in raw_cands if c in df.columns), None)
    cal = next((c for c in cal_cands if c in df.columns), None)
    return raw, cal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_dir", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--tol_p", type=float, default=1e-6)
    args = ap.parse_args()

    root = repo_root()
    ab_path, wp_path = find_required_files(root)

    ab = load_module_from_path("_artifact_bundle_dyn", ab_path)
    wp = load_module_from_path("_winprob_loader_dyn", wp_path)

    load_bundle = getattr(ab, "load_bundle")
    SchemaError = getattr(ab, "SchemaError")
    BundleError = getattr(ab, "BundleError")
    WinProbScorer = getattr(wp, "WinProbScorer")

    bundle_dir = (root / args.bundle_dir).resolve()
    fixture_path = (root / args.fixture).resolve()

    bundle = load_bundle(bundle_dir, strict=True)
    scorer = WinProbScorer(bundle=bundle, strict_schema=True)

    raw_feats = list(getattr(scorer, "raw_features", []) or [])
    if not raw_feats:
        print("[FAIL] scorer.raw_features empty", flush=True)
        return 2

    if fixture_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(fixture_path)
    else:
        df = pd.read_csv(fixture_path, low_memory=False)

    if "symbol" not in df.columns:
        raise ValueError("Fixture must contain column: symbol")

    ts_col = pick_ts_col(df)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["symbol", ts_col])

    allow = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if allow:
        df = df[df["symbol"].isin(allow)]

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)

    missing = sorted(set(raw_feats) - set(df.columns))
    if missing:
        print(f"[FAIL] fixture missing required feature columns: {missing[:50]}", flush=True)
        return 2

    raw_exp_col, cal_exp_col = pick_expected_prob_cols(df)

    n = len(df)
    bad = 0
    max_abs_err = 0.0

    for r in df.itertuples(index=False):
        row = r._asdict()
        sym = row["symbol"]
        ts = row[ts_col]
        raw_row: Dict[str, Any] = {k: row[k] for k in raw_feats}

        try:
            p_raw, p_cal = scorer.score_with_details(raw_row)  # returns (p_raw, p_cal)
            p_cal = float(p_cal)
            if not np.isfinite(p_cal):
                raise RuntimeError("p_cal_nonfinite")
        except (SchemaError, BundleError, Exception) as e:
            bad += 1
            print(f"[FAIL] {sym} ts={pd.to_datetime(ts, utc=True).isoformat()} err={type(e).__name__}:{e}", flush=True)
            continue

        if cal_exp_col is not None:
            try:
                exp = float(row[cal_exp_col])
                err = abs(p_cal - exp)
                max_abs_err = max(max_abs_err, err)
                if err > float(args.tol_p):
                    bad += 1
                    print(
                        f"[FAIL] {sym} ts={pd.to_datetime(ts, utc=True).isoformat()} "
                        f"p_cal_mismatch exp={exp:.8f} got={p_cal:.8f} abs_err={err:.6g}",
                        flush=True,
                    )
            except Exception as e:
                bad += 1
                print(f"[FAIL] {sym} ts={pd.to_datetime(ts, utc=True).isoformat()} bad_expected:{e}", flush=True)

    print(f"[DONE] rows={n} bad={bad} max_abs_cal_err={max_abs_err:.6g}", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
