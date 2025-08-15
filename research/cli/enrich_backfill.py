# research/cli/enrich_backfill.py
import os
import argparse
import asyncio
import pandas as pd
from ..data.loaders import fetch_positions_df
from ..data.features import enrich_funding_oi_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.getenv("PG_DSN"))
    ap.add_argument("--out_csv", default="research/data/funding_oi_backfill.csv")
    ap.add_argument("--oi_interval", default="1h")
    ap.add_argument("--funding_window_h", type=float, default=8.0)
    args = ap.parse_args()

    df = asyncio.get_event_loop().run_until_complete(fetch_positions_df(args.dsn))
    if df.empty:
        print("No trades.")
        return
    df2 = asyncio.get_event_loop().run_until_complete(
        enrich_funding_oi_features(df, window_hours=args.funding_window_h, oi_interval=args.oi_interval)
    )
    out = df2[["id","symbol","opened_at","funding_last_at_entry","oi_last_at_entry","oi_delta_pct_win"]]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out)} enrich rows to {args.out_csv}")

if __name__ == "__main__":
    main()
