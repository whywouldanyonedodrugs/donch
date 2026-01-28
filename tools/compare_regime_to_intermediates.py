#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure project root is on sys.path when running as a script from tools/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live.regime_features import (  # noqa: E402
    compute_daily_regime_series,
    compute_markov4h_series,
    DailyRegimeConfig,
    Markov4hConfig,
)


def _load_one(glob_pat: str, base: Path) -> Path:
    hits = sorted(base.glob(glob_pat))
    if not hits:
        raise FileNotFoundError(f"No files matching {glob_pat} under {base}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intermediate-dir", required=True)
    ap.add_argument("--fixtures-dir", required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--min-obs", type=int, default=80)
    ap.add_argument("--ewma-alpha", type=float, default=0.2)
    args = ap.parse_args()

    inter_dir = Path(args.intermediate_dir).resolve()
    fix_dir = Path(args.fixtures_dir).resolve()
    sym = args.symbol

    daily_int_path = _load_one(f"{sym}_daily_regime_intermediate.parquet", inter_dir)
    markov_int_path = _load_one(f"{sym}_markov4h_intermediate.parquet", inter_dir)

    daily_fix = fix_dir / f"{sym}_1D.parquet"
    markov_fix = fix_dir / f"{sym}_4H.parquet"

    df_daily = pd.read_parquet(daily_fix)
    df_4h = pd.read_parquet(markov_fix)

    # Normalize fixtures to timestamp-indexed UTC
    if "timestamp" in df_daily.columns:
        df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"], utc=True)
        df_daily = df_daily.set_index("timestamp")
    if "timestamp" in df_4h.columns:
        df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"], utc=True)
        df_4h = df_4h.set_index("timestamp")

    df_daily = df_daily.sort_index()
    df_4h = df_4h.sort_index()

    daily_int = pd.read_parquet(daily_int_path)
    daily_int["timestamp"] = pd.to_datetime(daily_int["timestamp"], utc=True)
    daily_int = daily_int.set_index("timestamp").sort_index()

    markov_int = pd.read_parquet(markov_int_path)
    markov_int["timestamp"] = pd.to_datetime(markov_int["timestamp"], utc=True)
    markov_int = markov_int.set_index("timestamp").sort_index()

    # Compute series using live implementation
    daily_series = compute_daily_regime_series(
        df_daily,
        DailyRegimeConfig(ma_period=200, atr_period=20, atr_mult=2.0, maxiter=200, min_obs=args.min_obs),
    ).sort_index()

    markov_series = compute_markov4h_series(
        df_4h,
        Markov4hConfig(maxiter=200, ewma_alpha=args.ewma_alpha, min_obs=args.min_obs),
    ).sort_index()

    # Compare daily (offline export uses vol_prob_low + regime_code)
    d = daily_series.join(daily_int[["vol_prob_low", "regime_code"]], how="inner")
    d = d.rename(columns={"vol_prob_low": "vol_prob_low_off", "regime_code": "regime_code_off"})
    d = d.dropna(subset=["vol_prob_low_1d", "vol_prob_low_off", "regime_code_1d", "regime_code_off"])

    if d.empty:
        print("DAILY: no overlapping comparable rows after dropna.")
    else:
        prob_err = np.abs(d["vol_prob_low_1d"].astype(float) - d["vol_prob_low_off"].astype(float))
        code_acc = (d["regime_code_1d"].astype(int) == d["regime_code_off"].astype(int)).mean()
        print(f"DAILY: n={len(d)} prob_mae={prob_err.mean():.6f} prob_max={prob_err.max():.6f} code_acc={code_acc:.4f}")

    # Compare markov (offline export uses prob_up_ewm + state_up)
    m = markov_series.join(markov_int[["prob_up_ewm", "state_up"]], how="inner")
    m = m.rename(columns={"prob_up_ewm": "prob_off", "state_up": "state_off"})
    m = m.dropna(subset=["markov_prob_up_4h", "prob_off", "markov_state_4h", "state_off"])

    if m.empty:
        print("MARKOV4H: no overlapping comparable rows after dropna.")
    else:
        prob_err = np.abs(m["markov_prob_up_4h"].astype(float) - m["prob_off"].astype(float))
        state_acc = (m["markov_state_4h"].astype(int) == m["state_off"].astype(int)).mean()
        print(f"MARKOV4H: n={len(m)} prob_mae={prob_err.mean():.6f} prob_max={prob_err.max():.6f} state_acc={state_acc:.4f}")

    print("\nSANITY timestamps:")
    print("daily_fixture:", df_daily.index.min(), "->", df_daily.index.max())
    print("daily_inter :", daily_int.index.min(), "->", daily_int.index.max())
    print("4h_fixture  :", df_4h.index.min(), "->", df_4h.index.max())
    print("4h_inter    :", markov_int.index.min(), "->", markov_int.index.max())


if __name__ == "__main__":
    main()
