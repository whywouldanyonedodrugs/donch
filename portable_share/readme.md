# Donch Portable Edition

This is the portable, sanitized, low-bloat distribution of the Donch bot.

It is prepared for sharing/public Git hosting and excludes local identity data, runtime artifacts, and secrets.

## Goals of this edition

- Keep only what is needed to run and operate the bot
- Remove legacy and non-essential extras
- Ship with safe templates and portable paths
- Keep the original project untouched (this folder is a separate package)

## What is included

- Core runtime code: `live/`
- Main strategy spec: `strategies/donch_pullback_long.yaml`
- Runtime config files: `config.py`, `config.yaml`, `symbols.txt`, `listing_dates.json`
- Minimal utility tools:
  - `tools/sync_bybit_perps_universe.py`
  - `tools/export_trades_markdown.py`
- Environment template: `.env.example`

## What is excluded on purpose

- Real credentials/tokens
- `.env` and machine-local overrides
- `.venv`, `__pycache__`, compiled files
- Backups and temporary files (`*.bak`, snapshots)
- Research/diagnostic scripts not needed for runtime operation
- Autopar export/timer workflow
- Legacy strategy specs

## Directory layout

```text
.
├── .env.example
├── .gitignore
├── config.py
├── config.yaml
├── listing_dates.json
├── requirements.txt
├── symbols.txt
├── live/
├── strategies/
│   └── donch_pullback_long.yaml
├── tools/
│   ├── export_trades_markdown.py
│   └── sync_bybit_perps_universe.py
└── results/
    ├── .gitkeep
    ├── runtime/.gitkeep
    └── meta_export/.gitkeep
```

## System requirements

- Linux/macOS (recommended)
- Python 3.10+ (3.11/3.12 recommended)
- PostgreSQL (for trade/equity persistence)
- Bybit API credentials
- Telegram bot credentials

## Installation

1. Create environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Create your env file:

```bash
cp .env.example .env
```

3. Edit `.env` and set your own values.

4. Prepare runtime folders (if missing):

```bash
mkdir -p results/runtime results/meta_export results/reports
```

5. Put your exported meta-model bundle into:

```text
results/meta_export/
```

## Required environment variables

Set these in `.env`:

- `BYBIT_API_KEY`
- `BYBIT_API_SECRET`
- `BYBIT_TESTNET` (`true` or `false`)
- `TG_BOT_TOKEN`
- `TG_CHAT_ID`
- `DATABASE_URL` (Postgres DSN)

Optional:

- `LOG_LEVEL` (default: `INFO`)
- `DONCH_REGIMES_REPORT`
- `DONCH_CROWD_THRESHOLDS`

## Configuration files

- `config.yaml`: primary runtime parameters (risk, gates, bundle paths, caching)
- `config.py`: baseline defaults and constants
- `symbols.txt`: active trading symbols list
- `strategies/donch_pullback_long.yaml`: active strategy spec

Important defaults already made portable:

- Bundle paths point to `results/meta_export`
- No hardcoded host-specific absolute paths
- No debug-sensitive defaults forced on

## Running the bot

Run from the project root:

```bash
python3 -m live.live_trader
```

## Trade report tool (Markdown)

The portable edition includes a dedicated trade reporting tool that queries the `positions` table and writes a clean Markdown report.

Script:

```text
tools/export_trades_markdown.py
```

Default output:

```text
results/reports/trades_report.md
```

Basic usage:

```bash
python3 tools/export_trades_markdown.py
```

Useful options:

```bash
python3 tools/export_trades_markdown.py --status closed --order desc --limit 200
python3 tools/export_trades_markdown.py --from-utc 2026-02-01T00:00:00Z --to-utc 2026-02-12T23:59:59Z
python3 tools/export_trades_markdown.py --output results/reports/feb_report.md --stdout
```

Arguments:

- `--dsn`: explicit Postgres DSN (overrides env)
- `--env-path`: fallback env file path (default `.env`)
- `--output`: markdown output path
- `--status`: `closed` (default) or `all`
- `--from-utc`: ISO lower bound (UTC)
- `--to-utc`: ISO upper bound (UTC)
- `--order`: `asc` or `desc`
- `--limit`: max rows (`0` = all)
- `--stdout`: also print markdown to terminal

Report contents:

- Summary block (count, win/loss, win rate, gross pnl, fees, net pnl)
- Markdown table with one row per trade
- Key columns: symbol, side, open/close time, size, entry, pnl, fees, net, risk, win-prob, exit reason

## Security and publishing checklist

Before pushing to GitHub:

1. Confirm no `.env` file exists in repo.
2. Confirm `.env.example` contains placeholders only.
3. Confirm no logs/results/artifacts are present.
4. Confirm no absolute host paths are present in config/scripts.
5. Confirm API keys use least privilege and are not reused from personal infra.

Quick checks:

```bash
rg -n "/[A-Za-z0-9._-]+/|API_KEY=|API_SECRET=|TG_BOT_TOKEN=|DATABASE_URL=" .
find . -type f \( -name '*.csv' -o -name '*.parquet' -o -name '*.zip' -o -name '.env' \)
```

## Troubleshooting

### 1) Bot fails at startup due to missing env fields

Cause: required secrets not set.
Fix: populate `.env` using `.env.example`.

### 2) Meta bundle/schema errors

Cause: missing/incomplete `results/meta_export` artifacts.
Fix: place the full compatible bundle under `results/meta_export` and verify manifest/threshold files exist.

### 3) DB connection error

Cause: invalid `DATABASE_URL` or DB unavailable.
Fix: verify DSN, host reachability, credentials, and DB permissions.

### 4) No symbols traded

Cause: restrictive filters/gates or stale/empty `symbols.txt`.
Fix: run universe sync tool and verify strategy/gating config in `config.yaml`.

### 5) Trade report is empty

Cause: no rows match current filters or DB has no closed trades yet.
Fix: run without date filters first, then narrow with `--from-utc` and `--to-utc`.

## Updating from this portable baseline

When you customize:

- Keep secrets only in `.env`
- Keep generated outputs under `results/`
- Do not add personal paths into committed config
- If adding new tools, keep them under `tools/` and document their purpose

## License and responsibility

Use at your own risk. Live trading involves financial risk.
