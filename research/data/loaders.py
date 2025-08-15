# research/data/loaders.py
import os
import asyncio
import asyncpg
import logging
from typing import Optional
import pandas as pd

LOG = logging.getLogger(__name__)

DEFAULT_QUERY = """
SELECT
  id, symbol, side, size, entry_price, stop_price, atr, status,
  opened_at, exit_deadline, closed_at, pnl, pnl_pct, exit_reason,
  -- features captured at entry
  rsi_at_entry, adx_at_entry, atr_pct_at_entry,
  price_boom_pct_at_entry, price_slowdown_pct_at_entry,
  vwap_z_at_entry, ema_fast_at_entry, ema_slow_at_entry,
  listing_age_days_at_entry, session_tag_at_entry,
  day_of_week_at_entry, hour_of_day_at_entry,
  eth_macd_at_entry, eth_macdsignal_at_entry, eth_macdhist_at_entry,
  eth_macd_1h_at_entry, eth_macdsignal_1h_at_entry, eth_macdhist_1h_at_entry,
  vwap_consolidated_at_entry,
  -- new VWAP stack diagnostics at entry (nullable if older trades)
  vwap_stack_frac_at_entry,
  vwap_stack_expansion_pct_at_entry,
  vwap_stack_slope_pph_at_entry,
  vwap_stack_multiplier_at_entry
FROM positions
WHERE status = 'CLOSED'
ORDER BY opened_at ASC;
"""

async def fetch_positions_df(pg_dsn: Optional[str] = None, query: str = DEFAULT_QUERY) -> pd.DataFrame:
    """Fetch CLOSED positions as a time-ordered DataFrame."""
    dsn = pg_dsn or os.getenv("PG_DSN")
    if not dsn:
        raise RuntimeError("PG_DSN is not set. Export PG_DSN or pass pg_dsn.")
    conn: asyncpg.Connection
    LOG.info("Querying positions from Postgres...")
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(query)
    finally:
        await conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    # Ensure times are timezone-aware
    for col in ("opened_at", "closed_at", "exit_deadline"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df

def fetch_positions_df_sync(pg_dsn: Optional[str] = None, query: str = DEFAULT_QUERY) -> pd.DataFrame:
    return asyncio.get_event_loop().run_until_complete(fetch_positions_df(pg_dsn, query))
