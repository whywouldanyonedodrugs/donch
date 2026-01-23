# tools/inspect_golden_semantics.py
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "meta_export" / "feature_manifest.json"
GOLDEN = ROOT / "golden_features.parquet"

def load_manifest_cols():
    m = json.loads(MANIFEST.read_text())
    num = m["features"]["numeric_cols"]
    cat = m["features"]["cat_cols"]
    return num, cat, num + cat

def load_golden():
    if not GOLDEN.exists():
        raise FileNotFoundError(f"Missing {GOLDEN}")
    df = pd.read_parquet(GOLDEN)
    # Expected columns: timestamp, symbol, + 73 features
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

def summarize_cat(df: pd.DataFrame, cat_cols: list[str]):
    print("\n=== CATEGORICAL COLS: dtype + uniques + top values ===")
    for c in cat_cols:
        if c not in df.columns:
            print(f"  {c}: MISSING")
            continue
        s = df[c]
        nunq = s.nunique(dropna=False)
        top = s.value_counts(dropna=False).head(8)
        print(f"\n{c}: dtype={s.dtype}, nunique={nunq}")
        print(top.to_string())

def slope_inference(df: pd.DataFrame):
    """
    Infer likely slope definition for *_slope_* columns using only golden columns.
    We test:
      - diff(1)
      - diff(1)/hours
    where the base column is the same name without '_slope_' (common pattern),
    OR for eth_* specifically, base is eth_macd_hist_1h/4h, etc.
    """
    print("\n=== SLOPE INFERENCE (golden-only) ===")
    if "timestamp" not in df.columns:
        print("No timestamp column; cannot infer slopes.")
        return

    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    slope_cols = [c for c in df.columns if "slope" in c]
    if not slope_cols:
        print("No slope columns found.")
        return

    # heuristic: extract timeframe hours from suffix like _1h, _4h
    def tf_hours(col: str) -> float | None:
        for suf, h in [("_15m", 0.25), ("_1h", 1.0), ("_4h", 4.0), ("_1d", 24.0)]:
            if col.endswith(suf):
                return h
        return None

    for sc in slope_cols:
        h = tf_hours(sc)
        # candidate base column names
        candidates = []
        if "_slope_" in sc:
            candidates.append(sc.replace("_slope_", "_hist_"))  # common MACD slope on hist
            candidates.append(sc.replace("_slope_", "_line_"))
            candidates.append(sc.replace("_slope_", "_signal_"))
            candidates.append(sc.replace("_slope_", "_"))       # drop slope token
        candidates.append(sc.replace("_slope", ""))             # fallback

        candidates = [c for c in candidates if c in df.columns]
        if not candidates:
            print(f"\n{sc}: no base candidates present in golden")
            continue

        y = df[sc].astype(float)

        best = None
        for bc in candidates:
            x = df[bc].astype(float)
            d = x.groupby(df["symbol"]).diff(1)
            # compare using MAE on finite rows
            mask = np.isfinite(y) & np.isfinite(d)
            if mask.any():
                mae = np.mean(np.abs(y[mask] - d[mask]))
                best = min(best, (mae, f"{bc}.diff(1)"), key=lambda t: t[0]) if best else (mae, f"{bc}.diff(1)")
            if h is not None:
                dp = d / h
                mask2 = np.isfinite(y) & np.isfinite(dp)
                if mask2.any():
                    mae2 = np.mean(np.abs(y[mask2] - dp[mask2]))
                    best = min(best, (mae2, f"{bc}.diff(1)/{h}h"), key=lambda t: t[0]) if best else (mae2, f"{bc}.diff(1)/{h}h")

        print(f"\n{sc}: best_match={best[1]}  MAE={best[0]:.6g}")

def regime_threshold_hints(df: pd.DataFrame):
    """
    For regime-code-like categorical cols, print numeric-feature separation hints:
    for each code column, summarize candidate driver numeric columns by quantiles per code.
    This does not 'guess' a final rule; it shows whether clean thresholds exist.
    """
    print("\n=== REGIME THRESHOLD HINTS (by code -> driver quantiles) ===")

    # candidate mappings by name pattern (kept lean and auditable)
    code_to_driver_patterns = {
        "funding_regime_code": ["funding_rate", "funding_z_7d", "funding_rollsum_3d", "funding_abs"],
        "oi_regime_code": ["oi_z_7d", "oi_pct_1d", "oi_pct_4h", "oi_pct_1h", "oi_level"],
        "btc_risk_regime_code": ["btc_trend_slope", "btc_vol_regime_level", "btc_funding_rate", "btc_oi_z_7d"],
        "regime_code_1d": ["gap_from_1d_ma", "vol_prob_low_1d", "rv_3d", "markov_prob_up_4h"],
    }

    for code_col, drivers in code_to_driver_patterns.items():
        if code_col not in df.columns:
            continue
        codes = df[code_col]
        if codes.isna().all():
            continue
        print(f"\n[{code_col}] dtype={codes.dtype}, unique={codes.nunique(dropna=False)}")

        for drv in drivers:
            if drv not in df.columns:
                continue
            x = df[drv].astype(float)
            tmp = pd.DataFrame({"code": codes, "x": x}).dropna()
            if tmp.empty:
                continue
            # quantiles per code
            q = tmp.groupby("code")["x"].quantile([0.01, 0.1, 0.5, 0.9, 0.99]).unstack()
            print(f"\n  driver={drv}")
            print(q.to_string())

def s_feature_relationships(df: pd.DataFrame):
    """
    Check whether S-features are trivially derived from existing categorical columns
    (e.g., S1 identical to regime_code_1d, or a simple pair encoding).
    """
    print("\n=== S-FEATURE RELATIONSHIPS ===")
    s_cols = [c for c in df.columns if c.startswith("S") and "_x_" in c or c.startswith("S1_") or c.startswith("S2_") or c.startswith("S3_") or c.startswith("S4_") or c.startswith("S5_") or c.startswith("S6_")]
    # Use manifest-like known names if present
    expected = ["S1_regime_code_1d", "S2_markov_x_vol1d", "S3_funding_x_oi", "S4_crowd_x_trend1d", "S5_btcRisk_x_regimeUp", "S6_fresh_x_compress"]
    s_cols = [c for c in expected if c in df.columns] or s_cols
    base_pairs = {
        "S1_regime_code_1d": ["regime_code_1d"],
        "S2_markov_x_vol1d": ["markov_state_4h", "vol_prob_low_1d"],
        "S3_funding_x_oi": ["funding_regime_code", "oi_regime_code"],
        "S5_btcRisk_x_regimeUp": ["btc_risk_regime_code", "regime_up"],
    }

    for sc in s_cols:
        print(f"\n{sc}: dtype={df[sc].dtype}, nunique={df[sc].nunique(dropna=False)}")
        if sc in base_pairs and all(c in df.columns for c in base_pairs[sc]):
            cols = base_pairs[sc]
            # crosstab to see if sc is a deterministic encoding of the tuple
            tmp = df[cols + [sc]].dropna()
            if not tmp.empty:
                # number of distinct sc values per tuple
                g = tmp.groupby(cols)[sc].nunique()
                max_n = int(g.max())
                print(f"  tuple_cols={cols} -> max distinct {sc} per tuple: {max_n}")
                # show a few example mappings
                ex = tmp.drop_duplicates(cols + [sc]).head(12)
                print("  examples:")
                for _, r in ex.iterrows():
                    tup = ",".join(f"{c}={r[c]}" for c in cols)
                    print(f"    {tup} -> {sc}={r[sc]}")

def main():
    num, cat, cols = load_manifest_cols()
    df = load_golden()

    # contract sanity
    feat_cols = [c for c in cols if c in df.columns]
    print(f"manifest_features={len(cols)} golden_rows={len(df)}")
    print(f"ts_range_utc=[{df['timestamp'].min() if 'timestamp' in df.columns else 'NA'} .. {df['timestamp'].max() if 'timestamp' in df.columns else 'NA'}]")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("missing_in_golden:", missing)
    else:
        print("missing_in_golden=0")

    summarize_cat(df, cat)
    slope_inference(df)
    regime_threshold_hints(df)
    s_feature_relationships(df)

if __name__ == "__main__":
    main()
