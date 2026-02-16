#!/usr/bin/env python3
"""
Run daily live ops for autopar:
  1) sync Bybit perps universe -> symbols.txt
  2) export daily autopar package

Default export date is yesterday UTC (complete day window).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def yesterday_utc() -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()


def run_cmd(cmd: list[str]) -> None:
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed rc={p.returncode}: {' '.join(cmd)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run daily universe sync + autopar export.")
    ap.add_argument("--date", default=None, help="UTC date YYYY-MM-DD (default: yesterday UTC)")
    ap.add_argument("--skip-sync", action="store_true", help="skip symbol universe sync step")
    ap.add_argument("--sync-min-count", type=int, default=100, help="minimum fetched symbol count safety check")
    ap.add_argument("--symbols-path", default="symbols.txt", help="symbols file path")
    ap.add_argument("--env-path", default=".env", help="env file path")
    ap.add_argument("--output-root", default="results/autopar_exports", help="autopar output root")
    ap.add_argument("--publish-dir", default=None, help="optional publish destination directory")
    ap.add_argument("--zip", action="store_true", help="zip package")
    ap.add_argument("--include-live-log", action="store_true", help="include live.log (META_DECISION lines)")
    ap.add_argument("--include-settings-snapshot", action="store_true", help="include settings snapshot")
    ap.add_argument("--overwrite", action="store_true", help="overwrite output package folder")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    date_s = args.date or yesterday_utc()
    publish_dir = args.publish_dir or os.getenv("AUTOPAR_PUBLISH_DIR", "")

    py = str(repo_root / ".venv" / "bin" / "python")
    sync_py = str(repo_root / "tools" / "sync_bybit_perps_universe.py")
    export_py = str(repo_root / "tools" / "export_autopar_daily.py")

    try:
        if not args.skip_sync:
            sync_cmd = [
                py,
                sync_py,
                "--symbols-path",
                args.symbols_path,
                "--env-path",
                args.env_path,
                "--min-count",
                str(args.sync_min_count),
            ]
            run_cmd(sync_cmd)

        export_cmd = [
            py,
            export_py,
            "--date",
            date_s,
            "--output-root",
            args.output_root,
            "--config-path",
            "config.yaml",
            "--symbols-path",
            args.symbols_path,
            "--invalid-symbols-path",
            "results/runtime/invalid_symbols.txt",
        ]
        if args.zip:
            export_cmd.append("--zip")
        if args.include_live_log:
            export_cmd.append("--include-live-log")
        if args.include_settings_snapshot:
            export_cmd.append("--include-settings-snapshot")
        if args.overwrite:
            export_cmd.append("--overwrite")
        if publish_dir:
            export_cmd.extend(["--publish-dir", publish_dir])

        run_cmd(export_cmd)
    except Exception as e:
        print(f"[ERROR] autopar_daily_task failed: {e}", file=sys.stderr)
        return 1

    print(f"[OK] autopar_daily_task completed for date={date_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
