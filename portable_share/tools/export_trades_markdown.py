from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
except Exception:
    psycopg2 = None


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.split("#", 1)[0].strip()
    return out


def as_utc(dt: datetime | None) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso_utc(raw: str | None) -> datetime | None:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fnum(v: Any, digits: int = 4) -> str:
    if v is None:
        return ""
    try:
        n = float(v)
    except Exception:
        return ""
    s = f"{n:.{digits}f}".rstrip("0").rstrip(".")
    if s in {"-0", "-0.0", ""}:
        return "0"
    return s


def ffloat(v: Any) -> float:
    try:
        if v is None:
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def md_escape(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    s = s.replace("\n", " ").replace("\r", " ")
    return s.replace("|", "\\|").strip()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export trades from database to a Markdown table.")
    ap.add_argument("--dsn", default="", help="Postgres DSN. If empty, uses DATABASE_URL from env/.env")
    ap.add_argument("--env-path", default=".env", help="env file path fallback")
    ap.add_argument("--output", default="results/reports/trades_report.md", help="output markdown file path")
    ap.add_argument("--status", choices=["closed", "all"], default="closed", help="trades filter")
    ap.add_argument("--from-utc", default="", help="optional ISO UTC lower bound, e.g. 2026-02-01T00:00:00Z")
    ap.add_argument("--to-utc", default="", help="optional ISO UTC upper bound, e.g. 2026-02-12T23:59:59Z")
    ap.add_argument("--order", choices=["asc", "desc"], default="desc", help="sort by trade close/open time")
    ap.add_argument("--limit", type=int, default=0, help="max number of rows (0 = all)")
    ap.add_argument("--stdout", action="store_true", help="also print markdown to stdout")
    return ap.parse_args()


def resolve_dsn(args: argparse.Namespace) -> str:
    if args.dsn.strip():
        return args.dsn.strip()
    env_dsn = os.getenv("DATABASE_URL", "").strip()
    if env_dsn:
        return env_dsn
    env_map = load_env_file(Path(args.env_path))
    return str(env_map.get("DATABASE_URL", "")).strip()


def fetch_rows(conn, status: str, from_utc: datetime | None, to_utc: datetime | None, order: str, limit: int) -> list[dict[str, Any]]:
    where: list[str] = []
    params: list[Any] = []

    if status == "closed":
        where.append("status = 'CLOSED'")

    if from_utc is not None:
        where.append("COALESCE(closed_at, opened_at) >= %s")
        params.append(from_utc)

    if to_utc is not None:
        where.append("COALESCE(closed_at, opened_at) <= %s")
        params.append(to_utc)

    sql = (
        "SELECT id, symbol, side, status, opened_at, closed_at, size, entry_price, "
        "pnl, pnl_pct, fees_paid, risk_usd, win_probability_at_entry, exit_reason "
        "FROM positions"
    )

    if where:
        sql += " WHERE " + " AND ".join(where)

    ord_sql = "ASC" if order.lower() == "asc" else "DESC"
    sql += f" ORDER BY COALESCE(closed_at, opened_at) {ord_sql}, id {ord_sql}"

    if limit > 0:
        sql += " LIMIT %s"
        params.append(limit)

    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    return rows


def build_markdown(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    pnl_sum = sum(ffloat(r.get("pnl")) for r in rows)
    fees_sum = sum(ffloat(r.get("fees_paid")) for r in rows)
    net_sum = pnl_sum - fees_sum
    wins = sum(1 for r in rows if ffloat(r.get("pnl")) > 0)
    losses = sum(1 for r in rows if ffloat(r.get("pnl")) < 0)
    flat = sum(1 for r in rows if ffloat(r.get("pnl")) == 0)
    n = len(rows)
    decisive = wins + losses
    win_rate = (wins / decisive * 100.0) if decisive > 0 else 0.0
    avg_pnl = (pnl_sum / n) if n > 0 else 0.0

    lines: list[str] = []
    lines.append("# Trades Report")
    lines.append("")
    lines.append(f"Generated: `{now_utc()}`")
    lines.append(f"Status filter: `{args.status}`")
    if args.from_utc:
        lines.append(f"From UTC: `{args.from_utc}`")
    if args.to_utc:
        lines.append(f"To UTC: `{args.to_utc}`")
    lines.append("")
    lines.append(f"Total trades: **{n}**")
    lines.append(f"Wins: **{wins}** | Losses: **{losses}** | Flat: **{flat}** | Win rate: **{win_rate:.2f}%**")
    lines.append(f"Gross PnL: **{fnum(pnl_sum, 2)}** | Fees: **{fnum(fees_sum, 2)}** | Net PnL: **{fnum(net_sum, 2)}** | Avg PnL/trade: **{fnum(avg_pnl, 2)}**")
    lines.append("")

    if not rows:
        lines.append("No trades found for the selected filters.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| ID | Symbol | Side | Status | Opened (UTC) | Closed (UTC) | Size | Entry | PnL | PnL % | Fees | Net | Risk USD | Win Prob | Exit Reason |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")

    for r in rows:
        pnl = ffloat(r.get("pnl"))
        fees = ffloat(r.get("fees_paid"))
        net = pnl - fees
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(r.get("id")),
                    md_escape(r.get("symbol") or ""),
                    md_escape(r.get("side") or ""),
                    md_escape(r.get("status") or ""),
                    md_escape(as_utc(r.get("opened_at"))),
                    md_escape(as_utc(r.get("closed_at"))),
                    fnum(r.get("size"), 6),
                    fnum(r.get("entry_price"), 6),
                    fnum(pnl, 2),
                    fnum(r.get("pnl_pct"), 2),
                    fnum(fees, 2),
                    fnum(net, 2),
                    fnum(r.get("risk_usd"), 2),
                    fnum(r.get("win_probability_at_entry"), 4),
                    md_escape(r.get("exit_reason") or ""),
                ]
            )
            + " |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if psycopg2 is None:
        print("[ERROR] psycopg2 is not installed. Install dependencies from requirements.txt", file=sys.stderr)
        return 1

    try:
        from_utc = parse_iso_utc(args.from_utc)
        to_utc = parse_iso_utc(args.to_utc)
    except Exception as e:
        print(f"[ERROR] Invalid --from-utc/--to-utc format: {e}", file=sys.stderr)
        return 2

    dsn = resolve_dsn(args)
    if not dsn:
        print("[ERROR] DATABASE_URL is missing (set env, .env, or pass --dsn)", file=sys.stderr)
        return 2

    try:
        conn = psycopg2.connect(dsn)
    except Exception as e:
        print(f"[ERROR] Failed to connect to database: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    try:
        rows = fetch_rows(conn, args.status, from_utc, to_utc, args.order, int(args.limit))
        md = build_markdown(rows, args)
    except Exception as e:
        print(f"[ERROR] Failed to build report: {type(e).__name__}: {e}", file=sys.stderr)
        conn.close()
        return 1

    conn.close()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    if args.stdout:
        print(md)

    print(f"[OK] Wrote Markdown report: {out_path}")
    print(f"[OK] Rows exported: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
