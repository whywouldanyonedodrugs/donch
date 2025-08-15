# research/cli/audit_dataset.py
import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/data/dataset.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    print(f"rows={len(df)}  cols={len(df.columns)}")
    if "y" in df.columns:
        pos = float(df["y"].mean())
        print(f"class balance: win-rate={pos:.2%}")
    # % missing top features
    cols = [c for c in df.columns if c.endswith("_at_entry") or c.startswith("vwap_") or c in ("y","pnl","pnl_pct")]
    miss = df[cols].isna().mean().sort_values(ascending=False).head(20)
    print("\nTop missing (first 20):")
    print(miss)
    # basic ranges
    for c in ["rsi_at_entry","adx_at_entry","price_boom_pct_at_entry","price_slowdown_pct_at_entry",
              "vwap_z_at_entry","ema_spread_pct_at_entry","eth_macdhist_at_entry",
              "vwap_stack_frac_at_entry","vwap_stack_expansion_pct_at_entry","vwap_stack_slope_pph_at_entry",
              "funding_last_at_entry","oi_last_at_entry","oi_delta_pct_win"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            print(f"{c}: count={s.notna().sum()}  mean={s.mean():.4g}  p10={s.quantile(0.1):.4g}  p90={s.quantile(0.9):.4g}")
if __name__ == "__main__":
    main()
