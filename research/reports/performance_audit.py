# research/reports/performance_audit.py
"""Comprehensive live-trade performance audit.

This script pulls the *real* trading history from Postgres together with
exchange-level trade data and produces a deep-dive diagnostic.  Metrics include
MAE/MFE distributions, Sharpe ratio, drawdowns, and expectancy analytics.  The
tool also surfaces configuration suggestions based on observed behaviour (e.g.
session bias, stop placement efficiency, exit-reason mix).

Usage
-----

```
python -m research.reports.performance_audit --dsn postgresql://... \
    --bybit-key XXX --bybit-secret YYY --since 2024-01-01
```

Environment variables ``PG_DSN`` or ``DATABASE_URL`` may also be used.  Bybit
credentials default to ``BYBIT_API_KEY`` / ``BYBIT_API_SECRET``.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import math
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional, Sequence

import asyncpg
import ccxt.async_support as ccxt
import pandas as pd

LOG = logging.getLogger("performance_audit")


# ──────────────────────────────── Data classes ────────────────────────────────


@dataclasses.dataclass
class TradeMismatch:
    position_id: int
    symbol: str
    reason: str
    db_qty: float
    exchange_qty: float
    db_avg_price: Optional[float] = None
    exchange_avg_price: Optional[float] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None


@dataclasses.dataclass
class PerformanceSummary:
    total_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_mae: Optional[float]
    avg_mfe: Optional[float]
    sharpe_daily: Optional[float]
    max_drawdown_pct: Optional[float]
    cagr: Optional[float]
    avg_holding_minutes: Optional[float]
    fees_paid: float
    net_pnl: float


# ──────────────────────────────── DB helpers ─────────────────────────────────


async def _fetch_table(
    conn: asyncpg.Connection,
    table: str,
    *,
    where: str | None = None,
    params: Sequence[Any] = (),
) -> list[asyncpg.Record]:
    query = f"SELECT * FROM {table}"
    if where:
        query += f" WHERE {where}"
    if table == "positions":
        query += " ORDER BY opened_at ASC"
    elif table == "fills":
        query += " ORDER BY ts ASC"
    elif table == "equity_snapshots":
        query += " ORDER BY ts ASC"
    return await conn.fetch(query, *params)


async def load_datasets(
    dsn: str,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load positions, fills and equity snapshots from Postgres."""

    conn = await asyncpg.connect(dsn)
    try:
        params: list[Any] = []
        clauses: list[str] = ["status = 'CLOSED'"]
        if start:
            clauses.append("opened_at >= $%d" % (len(params) + 1))
            params.append(start)
        if end:
            clauses.append("opened_at <= $%d" % (len(params) + 1))
            params.append(end)
        where = " AND ".join(clauses)
        positions = await _fetch_table(conn, "positions", where=where, params=params)

        fills_where = None
        fill_params: list[Any] = []
        if positions:
            pos_ids = [r["id"] for r in positions]
            fills_where = "position_id = ANY($1::int[])"
            fill_params = [pos_ids]
        fills = await _fetch_table(conn, "fills", where=fills_where, params=fill_params)

        eq_params: list[Any] = []
        eq_where = None
        if start and end:
            eq_where = "ts BETWEEN $1 AND $2"
            eq_params = [start - timedelta(days=1), end + timedelta(days=1)]
        equity = await _fetch_table(
            conn,
            "equity_snapshots",
            where=eq_where,
            params=eq_params,
        )
    finally:
        await conn.close()

    def _to_df(rows: Iterable[asyncpg.Record]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        for col in ("opened_at", "closed_at", "exit_deadline", "ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        return df

    return _to_df(positions), _to_df(fills), _to_df(equity)


# ──────────────────────────────── Metrics ────────────────────────────────────


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1.0
    return dd.min()


def _cagr(equity_curve: pd.Series, index: pd.DatetimeIndex) -> float:
    if equity_curve.empty:
        return float("nan")
    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    if start <= 0 or end <= 0:
        return float("nan")
    days = max(1.0, (index[-1] - index[0]).total_seconds() / 86400.0)
    years = days / 365.0
    return (end / start) ** (1.0 / years) - 1.0


def compute_performance_summary(
    positions: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    starting_equity: float = 1000.0,
) -> PerformanceSummary:
    if positions.empty:
        raise ValueError("No closed positions found – nothing to analyse.")

    df = positions.copy()
    for col in ("pnl", "pnl_pct", "mae_usd", "mfe_usd", "fees_paid", "holding_minutes", "risk_usd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["fees_paid"].fillna(0.0, inplace=True)
    df["net_pnl"] = df.get("pnl", 0.0) - df["fees_paid"]

    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] < 0]

    gross_profit = wins["net_pnl"].sum()
    gross_loss = -losses["net_pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf
    win_rate = (len(wins) / len(df)) * 100 if len(df) else float("nan")
    expectancy = df["net_pnl"].mean()

    avg_mae = df["mae_usd"].dropna().mean() if "mae_usd" in df else None
    avg_mfe = df["mfe_usd"].dropna().mean() if "mfe_usd" in df else None
    avg_hold = df["holding_minutes"].dropna().mean() if "holding_minutes" in df else None

    if not equity.empty:
        eq = equity.copy()
        eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
        eq.dropna(subset=["equity"], inplace=True)
        eq.sort_values("ts", inplace=True)
        eq.set_index("ts", inplace=True)
        eq_curve = eq["equity"]
    else:
        # Synthetic curve from trades
        df.sort_values("closed_at", inplace=True)
        eq_curve = starting_equity + df["net_pnl"].cumsum()
        eq_curve.index = df["closed_at"].fillna(pd.Timestamp.utcnow())

    sharpe = None
    if len(eq_curve) >= 3:
        daily = eq_curve.resample("1D").last().dropna()
        if len(daily) >= 3:
            rets = daily.pct_change().dropna()
            if not rets.empty:
                sharpe = rets.mean() / rets.std(ddof=1) * math.sqrt(365) if rets.std(ddof=1) > 0 else 0.0

    max_dd = _max_drawdown(eq_curve)
    cagr = _cagr(eq_curve, eq_curve.index)

    return PerformanceSummary(
        total_trades=len(df),
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_mae=avg_mae,
        avg_mfe=avg_mfe,
        sharpe_daily=sharpe,
        max_drawdown_pct=max_dd * 100 if max_dd == max_dd else None,
        cagr=cagr * 100 if cagr == cagr else None,
        avg_holding_minutes=avg_hold,
        fees_paid=df["fees_paid"].sum(),
        net_pnl=df["net_pnl"].sum(),
    )


def mae_mfe_distribution(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "mae_usd" in df:
        mae = pd.to_numeric(df["mae_usd"], errors="coerce").dropna()
        if not mae.empty:
            out["mae"] = {
                "mean": mae.mean(),
                "median": mae.median(),
                "p95": mae.quantile(0.95),
            }
    if "mfe_usd" in df:
        mfe = pd.to_numeric(df["mfe_usd"], errors="coerce").dropna()
        if not mfe.empty:
            out["mfe"] = {
                "mean": mfe.mean(),
                "median": mfe.median(),
                "p95": mfe.quantile(0.95),
            }
    return out


def segment_performance(df: pd.DataFrame, column: str, min_trades: int = 10) -> list[tuple[str, float, float, int]]:
    if column not in df.columns:
        return []
    grouped = df.dropna(subset=[column]).groupby(column)
    stats: list[tuple[str, float, float, int]] = []
    for key, sub in grouped:
        if len(sub) < min_trades:
            continue
        if "net_pnl" in sub:
            pnl_series = sub["net_pnl"]
        elif "pnl" in sub:
            pnl_series = sub["pnl"]
        else:
            continue
        pnl = pd.to_numeric(pnl_series, errors="coerce").dropna()
        if pnl.empty:
            continue
        stats.append((str(key), pnl.mean(), pnl.sum(), len(sub)))
    stats.sort(key=lambda x: x[1])  # ascending by expectancy
    return stats


# ──────────────────────────────── Suggestions ────────────────────────────────


def build_suggestions(df: pd.DataFrame, summary: PerformanceSummary) -> list[str]:
    suggestions: list[str] = []

    if summary.avg_mae is not None and "risk_usd" in df.columns:
        risk = pd.to_numeric(df["risk_usd"], errors="coerce").dropna()
        if not risk.empty:
            mae_to_risk = summary.avg_mae / (risk.mean() or 1.0)
            if mae_to_risk > 1.2:
                suggestions.append(
                    "Average MAE exceeds risk budget; consider tightening ATR-based stops "
                    "or reducing SL_ATR_MULT."
                )

    if summary.max_drawdown_pct is not None and summary.max_drawdown_pct < -15:
        suggestions.append(
            "Max drawdown beyond 15%; review DD_PAUSE settings or reduce per-trade risk."
        )

    if summary.sharpe_daily is not None and summary.sharpe_daily < 0.5:
        suggestions.append(
            "Sharpe ratio is weak (<0.5); investigate volatility filters or regime gating."
        )

    if "exit_reason" in df.columns:
        exit_counts = Counter(str(x).upper() for x in df["exit_reason"].fillna("UNKNOWN"))
        sl_ratio = exit_counts.get("SL", 0) / max(1, len(df))
        if sl_ratio > 0.4:
            suggestions.append(
                "High proportion of stop-loss exits (>40%); revisit trail logic or TP targets."
            )

    session_stats = segment_performance(df, "session_tag_at_entry", min_trades=8)
    if session_stats:
        worst = session_stats[0]
        if worst[1] < 0:
            suggestions.append(
                f"Session '{worst[0]}' shows negative expectancy; consider disabling it via SESSION filter."
            )

    if not suggestions:
        suggestions.append("No immediate configuration warnings detected – maintain current settings.")

    return suggestions


# ──────────────────────── Exchange reconciliation ────────────────────────────


async def _init_exchange(args) -> ccxt.Exchange:
    ex_id = args.exchange.lower()
    if ex_id != "bybit":
        raise RuntimeError("Only Bybit is currently supported for reconciliation.")

    exchange = ccxt.bybit(
        {
            "apiKey": args.bybit_key,
            "secret": args.bybit_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
    )
    if args.bybit_testnet:
        exchange.set_sandbox_mode(True)
    return exchange


async def fetch_trades_for_symbols(
    exchange: ccxt.Exchange,
    symbols: Sequence[str],
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    since_ms = int(since.timestamp() * 1000) if since else None
    until_ms = int(until.timestamp() * 1000) if until else None

    for symbol in symbols:
        ms = since_ms
        trades: list[dict[str, Any]] = []
        while True:
            try:
                batch = await exchange.fetch_my_trades(symbol, since=ms, limit=200)
            except Exception as exc:  # pragma: no cover - network/ccxt errors
                LOG.warning("fetch_my_trades failed for %s: %s", symbol, exc)
                break
            if not batch:
                break
            batch = [t for t in batch if t.get("timestamp")]
            batch.sort(key=lambda t: int(t["timestamp"]))
            trades.extend(batch)
            last_ts = int(batch[-1]["timestamp"])
            if until_ms and last_ts > until_ms:
                trades = [t for t in trades if t["timestamp"] <= until_ms]
                break
            if len(batch) < 200:
                break
            ms = last_ts + 1
            await asyncio.sleep(max(1.0, exchange.rateLimit / 1000.0))
        results[symbol] = trades
    return results


def _aggregate_trades(trades: list[dict[str, Any]], side: str, start: datetime, end: datetime) -> tuple[float, Optional[float]]:
    if not trades:
        return 0.0, None
    total_qty = 0.0
    total_notional = 0.0
    side = side.lower()
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    for t in trades:
        ts = int(t.get("timestamp") or 0)
        if ts < start_ms or ts > end_ms:
            continue
        if str(t.get("side", "")).lower() != side:
            continue
        qty = float(t.get("amount") or 0.0)
        price = float(t.get("price") or 0.0)
        if qty <= 0 or price <= 0:
            continue
        total_qty += qty
        total_notional += qty * price
    if total_qty <= 0:
        return 0.0, None
    return total_qty, total_notional / total_qty


def reconcile_positions(
    positions: pd.DataFrame,
    fills: pd.DataFrame,
    trades_by_symbol: dict[str, list[dict[str, Any]]],
    *,
    tolerance_qty: float = 1e-6,
) -> list[TradeMismatch]:
    mismatches: list[TradeMismatch] = []
    if positions.empty:
        return mismatches

    fills = fills.copy()
    if not fills.empty:
        fills["qty"] = pd.to_numeric(fills["qty"], errors="coerce")
        fills["price"] = pd.to_numeric(fills["price"], errors="coerce")

    for _, row in positions.iterrows():
        pid = int(row["id"])
        symbol = str(row["symbol"])
        opened_at: datetime = row.get("opened_at") or row.get("created_at")
        closed_at: datetime = row.get("closed_at") or opened_at
        if not isinstance(opened_at, pd.Timestamp):
            continue
        if not isinstance(closed_at, pd.Timestamp):
            closed_at = opened_at

        entry_side = "buy" if str(row.get("side", "short")).lower() == "long" else "sell"
        exit_side = "sell" if entry_side == "buy" else "buy"

        symbol_trades = trades_by_symbol.get(symbol, [])
        entry_qty, entry_avg = _aggregate_trades(
            symbol_trades,
            entry_side,
            opened_at - timedelta(minutes=5),
            opened_at + timedelta(minutes=5),
        )
        exit_qty, exit_avg = _aggregate_trades(
            symbol_trades,
            exit_side,
            opened_at,
            closed_at + timedelta(minutes=10),
        )

        db_size = float(row.get("size") or 0.0)
        if abs(abs(entry_qty) - abs(db_size)) > tolerance_qty:
            mismatches.append(
                TradeMismatch(
                    position_id=pid,
                    symbol=symbol,
                    reason="ENTRY_SIZE_MISMATCH",
                    db_qty=db_size,
                    exchange_qty=entry_qty,
                    db_avg_price=float(row.get("entry_price") or 0.0),
                    exchange_avg_price=entry_avg,
                    opened_at=opened_at.to_pydatetime(),
                    closed_at=closed_at.to_pydatetime() if isinstance(closed_at, pd.Timestamp) else None,
                )
            )

        exit_fills = fills[fills["position_id"] == pid]
        db_exit_qty = exit_fills["qty"].abs().sum() if not exit_fills.empty else abs(db_size)
        db_exit_avg = (
            (exit_fills["qty"].abs() * exit_fills["price"]).sum() / db_exit_qty
            if (not exit_fills.empty and db_exit_qty > 0)
            else None
        )

        if abs(exit_qty - db_exit_qty) > tolerance_qty:
            mismatches.append(
                TradeMismatch(
                    position_id=pid,
                    symbol=symbol,
                    reason="EXIT_SIZE_MISMATCH",
                    db_qty=db_exit_qty,
                    exchange_qty=exit_qty,
                    db_avg_price=db_exit_avg,
                    exchange_avg_price=exit_avg,
                    opened_at=opened_at.to_pydatetime(),
                    closed_at=closed_at.to_pydatetime() if isinstance(closed_at, pd.Timestamp) else None,
                )
            )

    return mismatches


# ──────────────────────────────── CLI helpers ────────────────────────────────


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Comprehensive live-trade performance audit")
    ap.add_argument("--dsn", default=os.getenv("PG_DSN") or os.getenv("DATABASE_URL"))
    ap.add_argument("--start", help="ISO start timestamp filter (opened_at)")
    ap.add_argument("--end", help="ISO end timestamp filter (opened_at)")
    ap.add_argument("--since", help="ISO timestamp for exchange trade pull")
    ap.add_argument("--until", help="ISO timestamp for exchange trade pull upper bound")
    ap.add_argument("--exchange", default="bybit")
    ap.add_argument("--bybit-key", default=os.getenv("BYBIT_API_KEY"))
    ap.add_argument("--bybit-secret", default=os.getenv("BYBIT_API_SECRET"))
    ap.add_argument("--bybit-testnet", action="store_true", default=os.getenv("BYBIT_TESTNET", "false").lower() == "true")
    ap.add_argument("--starting-equity", type=float, default=float(os.getenv("STARTING_EQUITY", "1000")))
    ap.add_argument("--skip-exchange", action="store_true", help="Skip exchange reconciliation")
    ap.add_argument("--tolerance", type=float, default=1e-6, help="Quantity tolerance for reconciliation")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def _print_summary(summary: PerformanceSummary):
    lines = [
        "──── Performance Summary ────",
        f"Trades: {summary.total_trades} | Win%: {summary.win_rate:.2f}% | PF: {summary.profit_factor:.3f}",
        f"Expectancy: {summary.expectancy:+.2f} USDT | Net PnL: {summary.net_pnl:+.2f} USDT | Fees: {summary.fees_paid:.2f} USDT",
    ]
    if summary.avg_mae is not None or summary.avg_mfe is not None:
        lines.append(
            f"Avg MAE: {summary.avg_mae:+.2f} USDT | Avg MFE: {summary.avg_mfe:+.2f} USDT"
        )
    if summary.avg_holding_minutes is not None:
        lines.append(f"Avg holding time: {summary.avg_holding_minutes:.1f} minutes")
    if summary.sharpe_daily is not None:
        lines.append(f"Sharpe (daily): {summary.sharpe_daily:.3f}")
    if summary.max_drawdown_pct is not None:
        lines.append(f"Max DD: {summary.max_drawdown_pct:.2f}%")
    if summary.cagr is not None:
        lines.append(f"CAGR: {summary.cagr:.2f}%")
    print("\n".join(lines))


def _print_distribution(dist: dict[str, Any]):
    if not dist:
        print("MAE/MFE distribution unavailable (columns missing).")
        return
    print("──── MAE / MFE Distribution ────")
    for key, stats in dist.items():
        print(f"{key.upper()} → mean: {stats['mean']:+.2f}, median: {stats['median']:+.2f}, p95: {stats['p95']:+.2f}")


def _print_segments(title: str, segs: list[tuple[str, float, float, int]]):
    if not segs:
        return
    print(f"──── {title} ────")
    for name, expectancy, total, n in segs:
        print(f"{name:<15} | n={n:>3} | exp={expectancy:+.2f} | total={total:+.2f}")


def _print_suggestions(suggestions: Sequence[str]):
    print("──── Suggested Actions ────")
    for idx, s in enumerate(suggestions, 1):
        print(f"{idx}. {s}")


async def main_async():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.dsn:
        raise RuntimeError("Postgres DSN missing – set PG_DSN, DATABASE_URL or pass --dsn")

    start = _parse_dt(args.start)
    end = _parse_dt(args.end)
    since = _parse_dt(args.since)
    until = _parse_dt(args.until)

    LOG.info("Loading datasets from Postgres…")
    positions, fills, equity = await load_datasets(args.dsn, start=start, end=end)
    if positions.empty:
        print("No closed trades found for the specified window.")
        return

    LOG.info("Computing performance analytics…")
    positions_aug = positions.copy()
    if "fees_paid" in positions_aug.columns:
        positions_aug["fees_paid"] = pd.to_numeric(positions_aug["fees_paid"], errors="coerce").fillna(0.0)
    else:
        positions_aug["fees_paid"] = 0.0
    if "pnl" in positions_aug.columns:
        positions_aug["pnl"] = pd.to_numeric(positions_aug["pnl"], errors="coerce")
    positions_aug["net_pnl"] = positions_aug.get("pnl", 0.0) - positions_aug["fees_paid"]

    summary = compute_performance_summary(positions_aug, equity, starting_equity=args.starting_equity)
    dist = mae_mfe_distribution(positions_aug)

    _print_summary(summary)
    print()
    _print_distribution(dist)
    print()
    _print_segments("Session Expectancy", segment_performance(positions_aug, "session_tag_at_entry"))
    _print_segments("Day-of-Week Expectancy", segment_performance(positions_aug, "day_of_week_at_entry"))

    suggestions = build_suggestions(positions_aug, summary)
    print()
    _print_suggestions(suggestions)

    mismatches: list[TradeMismatch] = []
    if not args.skip_exchange:
        if not args.bybit_key or not args.bybit_secret:
            raise RuntimeError("Exchange reconciliation requested but API credentials missing.")
        LOG.info("Fetching exchange trades for reconciliation…")
        exchange = await _init_exchange(args)
        try:
            symbols = sorted(set(positions["symbol"].dropna().unique()))
            trade_map = await fetch_trades_for_symbols(exchange, symbols, since=since, until=until)
        finally:
            await exchange.close()
        LOG.info("Reconciling %d positions against %d symbols…", len(positions), len(symbols))
        mismatches = reconcile_positions(positions, fills, trade_map, tolerance_qty=args.tolerance)

    if mismatches:
        print()
        print("──── Reconciliation Warnings ────")
        for mm in mismatches:
            print(
                f"PID {mm.position_id} {mm.symbol}: {mm.reason} | "
                f"db_qty={mm.db_qty:.6f} vs exch_qty={mm.exchange_qty:.6f}"
            )
    else:
        if not args.skip_exchange:
            print()
            print("Exchange reconciliation passed – no quantity mismatches detected.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

