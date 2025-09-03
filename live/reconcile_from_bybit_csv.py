#!/usr/bin/env python3
"""
scripts/reconcile_from_bybit_csv.py  (v3 – deterministic match)
===============================================================

Purpose
-------
Repair historical positions using a Bybit "Closed P&L" CSV by matching on:
  (symbol, order quantity, entry price)  ← robust even when DB closed_at is wrong.

What it updates
---------------
- status='CLOSED'
- closed_at  ← CSV trade time (UTC, with optional tz offset)
- avg_exit_price
- pnl
- pnl_pct (if entry_price * size is known)

Usage
-----
python scripts/reconcile_from_bybit_csv.py \
  --csv /path/to/bybit-donch.csv \
  --dsn "postgresql://USER:PASS@HOST:5432/DB" \
  --apply

Options
-------
--tz-offset-minutes INT   # if CSV times are local; UTC+3 → 180
--size-tol 0.001          # relative tolerance for quantity (default 0.1%)
--price-tol 0.001         # relative tolerance for entry price (default 0.1%)
--since YYYY-MM-DD        # optional lower bound by CSV trade time (after tz shift)
--symbol SYMBOL           # restrict to one symbol (optional)
--apply                   # actually write changes (omit = dry-run)
--verbose                 # print per-row diagnostics
"""

from __future__ import annotations
import argparse, asyncio, asyncpg, os, sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------- CSV parsing ----------

def _parse_trade_time(s: str, tz_offset_min: int) -> datetime:
    s = str(s).strip()
    # CSV samples seen: "00:02 2025-09-03" and "03/09/2025 00:02"
    for fmt in ("%H:%M %Y-%m-%d", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc) + timedelta(minutes=tz_offset_min)
        except Exception:
            pass
    # Fallback to pandas
    dt = pd.to_datetime(s, utc=False).to_pydatetime().replace(tzinfo=timezone.utc)
    return dt + timedelta(minutes=tz_offset_min)

@dataclass
class CsvRow:
    symbol: str
    qty: float
    entry_px: float
    exit_px: float
    pnl: float
    closed_at: datetime

def load_csv(path: str, tz_offset_min: int, symbol: Optional[str], since_date: Optional[datetime.date]) -> List[CsvRow]:
    df = pd.read_csv(path)
    required = ["Market","Order Quantity","Entry Price","Exit Price","Realized P&L","Trade time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing} (have: {list(df.columns)})")

    df["__ts"] = df["Trade time"].apply(lambda x: _parse_trade_time(x, tz_offset_min))
    if symbol:
        df = df[df["Market"].astype(str).str.upper() == symbol.upper()]
    if since_date:
        df = df[df["__ts"].dt.date >= since_date]

    rows: List[CsvRow] = []
    for _, r in df.iterrows():
        rows.append(CsvRow(
            symbol=str(r["Market"]).strip(),
            qty=float(r["Order Quantity"]),
            entry_px=float(r["Entry Price"]),
            exit_px=float(r["Exit Price"]),
            pnl=float(r["Realized P&L"]),
            closed_at=r["__ts"],
        ))
    rows.sort(key=lambda x: x.closed_at)
    return rows

# ---------- DB ----------

async def make_pool(args) -> asyncpg.Pool:
    if args.dsn:
        return await asyncpg.create_pool(dsn=args.dsn)
    return await asyncpg.create_pool(
        host=args.host or os.getenv("DB_HOST","localhost"),
        port=int(args.port or os.getenv("DB_PORT","5432")),
        database=args.db or os.getenv("DB_NAME","trading"),
        user=args.user or os.getenv("DB_USER","postgres"),
        password=args.pw or os.getenv("DB_PASSWORD",""),
    )

async def detect_columns(conn: asyncpg.Connection) -> Dict[str,bool]:
    cols = {r["column_name"] for r in await conn.fetch(
        "SELECT column_name FROM information_schema.columns WHERE table_name='positions'"
    )}
    for c in ("id","symbol","size","entry_price","opened_at"):
        if c not in cols:
            raise SystemExit(f"positions.{c} missing; cannot reconcile.")
    return {c: (c in cols) for c in cols}

@dataclass
class Position:
    id: int
    symbol: str
    size: float
    entry_price: float
    opened_at: datetime
    status: Optional[str]
    closed_at: Optional[datetime]
    pnl: Optional[float]

async def load_positions(conn: asyncpg.Connection, symbol: Optional[str]) -> List[Position]:
    where = []
    args: List[Any] = []
    if symbol:
        where.append("symbol = $1"); args.append(symbol)
    sql = f"""
        SELECT id, symbol, size, entry_price, opened_at, status, closed_at, pnl
        FROM positions
        {"WHERE " + " AND ".join(where) if where else ""}
        ORDER BY symbol, opened_at, id
    """
    rows = await conn.fetch(sql, *args)
    out: List[Position] = []
    for r in rows:
        out.append(Position(
            id=r["id"], symbol=r["symbol"], size=float(r["size"] or 0.0),
            entry_price=float(r["entry_price"] or 0.0),
            opened_at=r["opened_at"], status=r.get("status"),
            closed_at=r.get("closed_at"), pnl=float(r["pnl"]) if r["pnl"] is not None else None
        ))
    return out

# ---------- Matching (symbol + size + entry price) ----------

def rel_diff(a: float, b: float) -> float:
    denom = max(1e-12, abs(a))
    return abs(a - b) / denom

def build_index(positions: List[Position]):
    by_sym: Dict[str, List[Position]] = {}
    for p in positions:
        by_sym.setdefault(p.symbol, []).append(p)
    return by_sym

def match_row(row: CsvRow, cands: List[Position], size_tol: float, price_tol: float, used_ids: set[int]) -> Optional[Position]:
    if not cands:
        return None
    # filter by tolerances
    filt = []
    for p in cands:
        if p.id in used_ids:
            continue
        if rel_diff(p.size, row.qty) <= size_tol and rel_diff(p.entry_price, row.entry_px) <= price_tol:
            filt.append(p)
    if not filt:
        return None
    # choose nearest opened_at to the CSV closed time (stable tie-breaker)
    filt.sort(key=lambda p: abs((p.opened_at - row.closed_at).total_seconds()))
    return filt[0]

# ---------- Reconcile ----------

async def reconcile(args) -> int:
    csv_rows = load_csv(args.csv, args.tz_offset_minutes, args.symbol, args.since)
    pool = await make_pool(args)
    updated = 0
    unmatched: List[CsvRow] = []
    async with pool.acquire() as conn:
        cols = await detect_columns(conn)
        pos = await load_positions(conn, args.symbol)
        idx = build_index(pos)
        used_ids: set[int] = set()

        async with conn.transaction():
            for row in csv_rows:
                cands = idx.get(row.symbol, [])
                p = match_row(row, cands, args.size_tol, args.price_tol, used_ids)
                if not p:
                    unmatched.append(row); continue

                sets, vals = [], []
                if "status" in cols:             sets.append("status='CLOSED'")
                if "closed_at" in cols:          sets.append(f"closed_at = ${len(vals)+1}");     vals.append(row.closed_at)
                if "avg_exit_price" in cols:     sets.append(f"avg_exit_price = ${len(vals)+1}"); vals.append(row.exit_px)
                if "pnl" in cols:                sets.append(f"pnl = ${len(vals)+1}");           vals.append(row.pnl)
                if "pnl_pct" in cols:
                    denom = abs(p.entry_price) * abs(p.size)
                    if denom > 0:
                        sets.append(f"pnl_pct = ${len(vals)+1}"); vals.append(row.pnl / denom)

                vals.append(p.id)
                sql = f"UPDATE positions SET {', '.join(sets)} WHERE id = ${len(vals)}"
                if args.apply:
                    await conn.execute(sql, *vals)
                used_ids.add(p.id); updated += 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] Updated={updated}, Unmatched CSV rows={len(unmatched)}")
    if unmatched or args.verbose:
        for u in unmatched:
            print(f"[MISS] {u.symbol} qty={u.qty} entry={u.entry_px} exit={u.exit_px} @ {u.closed_at.isoformat()}")
    await pool.close()
    return 0

# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dsn")
    ap.add_argument("--host"); ap.add_argument("--port"); ap.add_argument("--db"); ap.add_argument("--user"); ap.add_argument("--pw")
    ap.add_argument("--tz-offset-minutes", type=int, default=0)
    ap.add_argument("--size-tol", type=float, default=0.001)
    ap.add_argument("--price-tol", type=float, default=0.001)
    ap.add_argument("--symbol")
    ap.add_argument("--since", help="YYYY-MM-DD")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.since:
        args.since = datetime.strptime(args.since, "%Y-%m-%d").date()
    return args

if __name__ == "__main__":
    try:
        asyncio.run(reconcile(parse_args()))
    except KeyboardInterrupt:
        sys.exit(130)
