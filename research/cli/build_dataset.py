# research/cli/build_dataset.py
import os
import argparse
import asyncio
import pandas as pd

from research.data.loaders import fetch_positions_df
from research.data.features import derive_static_entry_features, enrich_funding_oi_features
from research.data.labels import compute_binary_label_from_realized, add_time_order_index
from research.data.tca import add_cost_audit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.getenv("PG_DSN"), help="Postgres DSN")
    ap.add_argument("--out", default="research/data/dataset.parquet")
    ap.add_argument("--oi_interval", default="1h", choices=["5min","15min","30min","1h","4h","1d"])
    ap.add_argument("--funding_window_h", type=float, default=8.0)
    ap.add_argument("--taker_bps", type=float, default=5.0)
    args = ap.parse_args()

    async def _run():
        df = await fetch_positions_df(args.dsn)
        if df.empty:
            print("No trades found.")
            return None

        df = derive_static_entry_features(df)
        df = await enrich_funding_oi_features(df, window_hours=args.funding_window_h, oi_interval=args.oi_interval)
        df = add_cost_audit(df, taker_bps=args.taker_bps)
        df["y"] = compute_binary_label_from_realized(df)
        df = add_time_order_index(df)
        return df

    df = asyncio.run(_run())
    if df is None:
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
