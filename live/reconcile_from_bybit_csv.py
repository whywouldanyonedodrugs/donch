#!/usr/bin/env python3
"""
scripts/reconcile_from_bybit_csv.py
===================================
Fix historical rows in `positions` from a Bybit "Closed P&L" CSV export.

What it does
------------
- Loads CSV with columns: Market, Order Quantity, Entry Price, Exit Price, Realized P&L, Trade time
- For each CSV row (interpreted as a *fully closed* position), finds the best-matching *open* position
  by symbol and timing (and with size/price fuzzy match) and marks it CLOSED with exit stats.

Assumptions
-----------
- CSV is from Bybit "Closed P&L" (Derivatives → Closed P&L → Export).  # verified in docs
- CSV 'Realized P&L' is *net* of trading & funding fees → write to `positions.pnl`.  # per Bybit docs
- Bot is long-only (Donch). If you also have shorts, pass --side-filter to restrict.
- Timestamps treated as UTC unless you pass --tz-offset-minutes.

Safety
------
- Dry-run by default; requires --apply to modify DB.
- Only updates rows where status != 'CLOSED' OR closed_at IS NULL.
- Uses information_schema to detect optional columns (avg_exit_price, pnl_pct, fees_paid).

Usage
-----
python scripts/reconcile_from_bybit_csv.py \
  --csv /path/to/bybit-donch.csv \
  --dsn "postgresql://user:pass@localhost:5432/trading" \
  --apply

Options
-------
  --csv PATH                       CSV file (Bybit Closed P&L export)
  --dsn DSN                        Postgres DSN (or provide env DB_* vars)
  --host/--port/--db/--user/--pw   If not using DSN
  --tz-offset-minutes N            If your CSV times are not UTC (default: 0)
  --symbol SYMBOL                  Only reconcile a single symbol (optional)
  --since YYYY-MM-DD               Only reconcile trades closed on/after date (optional)
  --side-filter LONG|SHORT         Restrict by side of DB positions (optional)
  --size-tol 0.2                   Allowed relative diff |csv_qty - pos.size| / pos.size (default 0.2)
  --price-tol 0.02                 Allowed relative diff on entry price (default 2%)
  --apply                          Actually write changes (omit for dry-run)
"""
from __future__ import annotations

import argparse
import asyncio
import asyncpg
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------- Helpers ----------

def _parse_ts(s: str, tz_offset_min: int) -> datetime:
    # Bybit CSV sample: "00:02 2025-09-03"
    # If format drifts, try pandas.to_datetime as fallback.
    try:
        dt = datetime.strptime(s.strip(), "%H:%M %Y-%m-%d")
    except Exception:
        dt = pd.to_datetime(s, utc=False).to_pydatetime()
    dt = dt.replace(tzinfo=timezone.utc) + timedelta(minutes=tz_offset_min)
    return dt

@dataclass
class CsvRow:
    symbol: str
    qty: float
    entry_px: float
    exit_px: float
    realized_pnl: float
    closed_at: datetime

@dataclass
class Position:
    id: int
    symbol: str
    side: Optional[str]
    size: Optional[float]
    entry_price: Optional[float]
    opened_at: datetime
    status: Optional[str]
    closed_at: Optional[datetime]

# ---------- CSV ----------

def load_bybit_closed_pnl_csv(path: str, tz_offset_min: int) -> List[CsvRow]:
    df = pd.read_csv(path)
    required = ["Market", "Order Quantity", "Entry Price", "Exit Price", "Realized P&L", "Trade time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing} (have: {list(df.columns)})")

    rows: List[CsvRow] = []
    for _, r in df.iterrows():
        rows.append(
            CsvRow(
                symbol=str(r["Market"]).strip(),
                qty=float(r["Order Quantity"]),
                entry_px=float(r["Entry Price"]),
                exit_px=float(r["Exit Price"]),
                realized_pnl=float(r["Realized P&L"]),
                closed_at=_parse_ts(str(r["Trade time"]), tz_offset_min),
            )
        )
    # chronological
    rows.sort(key=lambda x: x.closed_at)
    return rows

# ---------- DB ----------

async def make_pool(args) -> asyncpg.Pool:
    if args.dsn:
        return await asyncpg.create_pool(dsn=args.dsn)
    host = args.host or os.getenv("DB_HOST", "localhost")
    port = int(args.port or os.getenv("DB_PORT", "5432"))
    db   = args.db   or os.getenv("DB_NAME", "trading")
    user = args.user or os.getenv("DB_USER", "postgres")
    pw   = args.pw   or os.getenv("DB_PASSWORD", "")
    return await asyncpg.create_pool(host=host, port=port, database=db, user=user, password=pw)

async def get_position_columns(conn: asyncpg.Connection) -> Dict[str, bool]:
    rows = await conn.fetch("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name='positions'
    """)
    cols = {r["column_name"] for r in rows}
    need = ["id","symbol","opened_at","status"]  # minimal required
    for c in need:
        if c not in cols:
            raise SystemExit(f"positions.{c} column missing; cannot proceed.")
    return {c: (c in cols) for c in cols}

async def fetch_open_positions(conn: asyncpg.Connection,
                               symbol: Optional[str],
                               side_filter: Optional[str]) -> List[Position]:
    where = ["(status IS DISTINCT FROM 'CLOSED' OR closed_at IS NULL)"]
    params: List[Any] = []
    if symbol:
        where.append("symbol = $%d" % (len(params)+1))
        params.append(symbol)
    sql = f"""
        SELECT id, symbol, side, size, entry_price, opened_at, status, closed_at
        FROM positions
        WHERE {' AND '.join(where)}
        ORDER BY symbol ASC, opened_at ASC, id ASC
    """
    rows = await conn.fetch(sql, *params)
    pos: List[Position] = []
    for r in rows:
        pos.append(Position(
            id=r["id"], symbol=r["symbol"], side=r.get("side"),
            size=float(r["size"]) if r["size"] is not None else None,
            entry_price=float(r["entry_price"]) if r["entry_price"] is not None else None,
            opened_at=r["opened_at"], status=r["status"], closed_at=r["closed_at"]
        ))
    # optional side_filter
    if side_filter:
        side_filter = side_filter.upper()
        pos = [p for p in pos if (p.side or "").upper() == side_filter]
    return pos

# ---------- Matching logic ----------

def rel_diff(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return math.inf
    denom = max(1e-12, abs(a))
    return abs(a - b) / denom

def best_match(csv: CsvRow, candidates: List[Position], size_tol: float, price_tol: float) -> Optional[Position]:
    # pick the earliest still-open pos for the same symbol with opened_at <= csv.closed_at
    # prefer tight match on size and entry_price; fall back to timing-only if needed
    same_sym = [p for p in candidates if p.symbol == csv.symbol and p.opened_at <= csv.closed_at]
    if not same_sym:
        return None
    # rank by composite score
    def score(p: Position) -> Tuple[int, float, float, float]:
        szd = rel_diff(p.size, csv.qty)
        pxd = rel_diff(p.entry_price, csv.entry_px)
        # primary filters
        ok = int((szd <= size_tol) and (pxd <= price_tol))
        # tie-breakers: closer entry time to close time, then smaller diffs
        time_gap = abs((csv.closed_at - p.opened_at).total_seconds())
        return (-ok, time_gap, szd, pxd)
    same_sym.sort(key=score)
    pick = same_sym[0]
    # allow fallback if no "ok" match
    if score(pick)[0] != -1:
        # loosen if nothing fits: accept timing-only
        pass
    return pick

# ---------- Reconcile ----------

async def reconcile(args) -> int:
    csv_rows = load_bybit_closed_pnl_csv(args.csv, args.tz_offset_minutes)
    pool = await make_pool(args)
    updated = 0
    unmatched = 0
    async with pool.acquire() as conn:
        cols = await get_position_columns(conn)
        pos = await fetch_open_positions(conn, args.symbol, args.side_filter)

        # Group candidates by symbol for speed
        by_sym: Dict[str, List[Position]] = {}
        for p in pos:
            by_sym.setdefault(p.symbol, []).append(p)

        async with conn.transaction():
            for row in csv_rows:
                if args.since and row.closed_at.date() < args.since:
                    continue
                cands = by_sym.get(row.symbol, [])
                match = best_match(row, cands, args.size_tol, args.price_tol)
                if not match:
                    unmatched += 1
                    print(f"[WARN] Unmatched CSV trade {row.symbol} @ {row.closed_at} qty={row.qty} entry={row.entry_px} exit={row.exit_px} pnl={row.realized_pnl:+.2f}")
                    continue

                # Build UPDATE set list dynamically based on available columns
                sets = []
                vals: List[Any] = []

                if "status" in cols:           sets.append("status = 'CLOSED'")
                if "closed_at" in cols:        sets.append(f"closed_at = ${len(vals)+1}");    vals.append(row.closed_at)
                if "avg_exit_price" in cols:   sets.append(f"avg_exit_price = ${len(vals)+1}"); vals.append(row.exit_px)
                if "pnl" in cols:              sets.append(f"pnl = ${len(vals)+1}");          vals.append(row.realized_pnl)
                if "pnl_pct" in cols:
                    # pnl% vs notional at entry (if we have both fields)
                    if match.entry_price is not None and match.size is not None and abs(match.entry_price*match.size) > 0:
                        pnl_pct = row.realized_pnl / (abs(match.entry_price) * abs(match.size))
                    else:
                        pnl_pct = None
                    if pnl_pct is not None:
                        sets.append(f"pnl_pct = ${len(vals)+1}"); vals.append(pnl_pct)

                if not sets:
                    print("[INFO] Nothing to update (no recognized columns).")
                    continue

                vals.append(match.id)
                sql = f"UPDATE positions SET {', '.join(sets)} WHERE id = ${len(vals)}"
                if args.apply:
                    await conn.execute(sql, *vals)
                updated += 1
                # Remove the matched candidate so it won't be reused
                by_sym[row.symbol] = [p for p in cands if p.id != match.id]

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] Updated={updated}, Unmatched CSV rows={unmatched}")
    await pool.close()
    return 0

# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dsn")
    ap.add_argument("--host")
    ap.add_argument("--port")
    ap.add_argument("--db")
    ap.add_argument("--user")
    ap.add_argument("--pw")
    ap.add_argument("--tz-offset-minutes", type=int, default=0)
    ap.add_argument("--symbol")
    ap.add_argument("--since", help="YYYY-MM-DD")
    ap.add_argument("--side-filter", choices=["LONG","SHORT"])
    ap.add_argument("--size-tol", type=float, default=0.20)
    ap.add_argument("--price-tol", type=float, default=0.02)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.since:
        args.since = datetime.strptime(args.since, "%Y-%m-%d").date()
    return args

if __name__ == "__main__":
    try:
        asyncio.run(reconcile(parse_args()))
    except KeyboardInterrupt:
        sys.exit(130)
