#!/usr/bin/env python3
"""
Sync symbols.txt from the active Bybit linear USDT perpetual universe.

This is intended for daily refresh so newly listed perps are automatically included.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt


SYMBOL_RE = re.compile(r"^[A-Z0-9]+USDT$")


def as_utc_iso(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


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


def parse_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    x = v.strip().lower()
    if x in {"1", "true", "yes", "on"}:
        return True
    if x in {"0", "false", "no", "off"}:
        return False
    return default


def _canonical_symbol_id(m: dict[str, Any]) -> str:
    sid = str(m.get("id") or "").upper().strip()
    if sid:
        sid = sid.replace("/", "").split(":")[0]
    if not sid:
        base = str(m.get("base") or "").upper().strip()
        quote = str(m.get("quote") or "").upper().strip()
        if base and quote:
            sid = f"{base}{quote}"
    return sid


def extract_linear_usdt_perp_ids(markets: dict[str, dict[str, Any]]) -> list[str]:
    out: set[str] = set()
    for m in markets.values():
        try:
            if not bool(m.get("swap", False)):
                continue
            if not bool(m.get("linear", False)):
                continue
            if m.get("contract", True) is False:
                continue
            if m.get("active", True) is False:
                continue
            quote = str(m.get("quote") or "").upper().strip()
            settle = str(m.get("settle") or "").upper().strip()
            if quote != "USDT":
                continue
            if settle and settle != "USDT":
                continue

            sid = _canonical_symbol_id(m)
            if not sid.endswith("USDT"):
                continue
            if not SYMBOL_RE.match(sid):
                continue
            out.add(sid)
        except Exception:
            continue
    return sorted(out)


def fetch_bybit_linear_usdt_perps(testnet: bool, timeout_ms: int = 15000) -> list[str]:
    ex = ccxt.bybit(
        {
            "enableRateLimit": True,
            "timeout": timeout_ms,
            "options": {"defaultType": "swap"},
            "testnet": testnet,
        }
    )
    markets = ex.load_markets()
    return extract_linear_usdt_perp_ids(markets)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sync symbols.txt from Bybit linear USDT perps universe.")
    ap.add_argument("--symbols-path", default="symbols.txt", help="target symbols file")
    ap.add_argument("--env-path", default=".env", help="env file to read BYBIT_TESTNET")
    ap.add_argument("--testnet", action="store_true", help="force testnet")
    ap.add_argument("--mainnet", action="store_true", help="force mainnet")
    ap.add_argument("--min-count", type=int, default=100, help="fail if fetched symbol count is below this")
    ap.add_argument("--merge-existing", action="store_true", help="union fetched symbols with current symbols.txt")
    ap.add_argument(
        "--report-path",
        default="results/runtime/symbols_sync_report.json",
        help="where to write sync report json",
    )
    ap.add_argument("--dry-run", action="store_true", help="compute + report only, do not write symbols.txt")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    env_map = load_env_file(Path(args.env_path))

    if args.testnet and args.mainnet:
        print("[ERROR] Use only one of --testnet or --mainnet", file=sys.stderr)
        return 2
    if args.testnet:
        testnet = True
    elif args.mainnet:
        testnet = False
    else:
        testnet = parse_bool(env_map.get("BYBIT_TESTNET"), default=False)

    try:
        fetched = fetch_bybit_linear_usdt_perps(testnet=testnet)
    except Exception as e:
        print(f"[ERROR] Failed to fetch Bybit markets: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if len(fetched) < int(args.min_count):
        print(
            f"[ERROR] Refusing to update symbols: fetched={len(fetched)} < min_count={args.min_count}",
            file=sys.stderr,
        )
        return 1

    symbols_path = Path(args.symbols_path)
    existing: list[str] = []
    if symbols_path.exists():
        existing = [ln.strip().upper() for ln in symbols_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        existing = [s for s in existing if SYMBOL_RE.match(s)]

    target = sorted(set(fetched) | set(existing)) if args.merge_existing else fetched
    added = sorted(set(target) - set(existing))
    removed = sorted(set(existing) - set(target))

    if not args.dry_run:
        symbols_path.write_text("\n".join(target) + "\n", encoding="utf-8")

    report = {
        "synced_at_utc": as_utc_iso(),
        "testnet": bool(testnet),
        "symbols_path": str(symbols_path),
        "count_before": len(existing),
        "count_after": len(target),
        "count_added": len(added),
        "count_removed": len(removed),
        "added_sample": added[:50],
        "removed_sample": removed[:50],
        "dry_run": bool(args.dry_run),
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    mode = "DRY_RUN" if args.dry_run else "UPDATED"
    print(
        f"[OK] {mode} symbols_path={symbols_path} before={len(existing)} after={len(target)} "
        f"added={len(added)} removed={len(removed)} testnet={testnet}"
    )
    print(f"[OK] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
