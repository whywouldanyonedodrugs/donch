# Autopar Live Daily Export

This repository includes:

- `tools/export_autopar_daily.py` to generate parity package files.
- `tools/sync_bybit_perps_universe.py` to refresh `symbols.txt` from active Bybit linear USDT perps.
- `tools/autopar_daily_task.py` to run both steps as one daily job.

## Output package

Default output root: `results/autopar_exports/`

For date `YYYY-MM-DD`, the tool writes:

- `results/autopar_exports/autopar_YYYY-MM-DD/live_decisions.csv`
- `results/autopar_exports/autopar_YYYY-MM-DD/live_trades.csv`
- `results/autopar_exports/autopar_YYYY-MM-DD/symbols_active.txt`
- `results/autopar_exports/autopar_YYYY-MM-DD/run_context.json`
- `results/autopar_exports/autopar_YYYY-MM-DD/live.log` (raw daily service log window)
- `results/autopar_exports/autopar_YYYY-MM-DD/settings_snapshot.json`
- `results/autopar_exports/autopar_YYYY-MM-DD/schema_diagnostics.json`
- `results/autopar_exports/autopar_YYYY-MM-DD/symbol_metadata.csv`
- `results/autopar_exports/autopar_YYYY-MM-DD/reason_mapping.csv`
- `results/autopar_exports/autopar_YYYY-MM-DD/reason_agreement.json`

Optional:

- `autopar_YYYY-MM-DD.zip`

## Run examples

Today UTC window:

```bash
/root/apps/donch/.venv/bin/python tools/export_autopar_daily.py \
  --date 2026-02-10 \
  --zip \
  --overwrite
```

Refresh symbols universe from Bybit (includes newly listed perps):

```bash
/root/apps/donch/.venv/bin/python tools/sync_bybit_perps_universe.py \
  --symbols-path symbols.txt
```

Run full daily task (sync + export) for yesterday UTC:

```bash
/root/apps/donch/.venv/bin/python tools/autopar_daily_task.py \
  --zip \
  --overwrite
```

Publish to a mounted/shared directory (for backtest machine pickup):

```bash
/root/apps/donch/.venv/bin/python tools/export_autopar_daily.py \
  --date 2026-02-10 \
  --zip \
  --publish-dir /mnt/shared/autopar
```

## Automation (systemd timer)

Unit files are provided:

- `deploy/systemd/donch-autopar-daily.service`
- `deploy/systemd/donch-autopar-daily.timer`

Install + enable:

```bash
sudo /root/apps/donch/tools/install_autopar_timer.sh
```

Timer schedule:

- `OnCalendar=*-*-* 00:20:00 UTC`
- Exports previous UTC day and persists missed runs after downtime.

Optional publish destination:

1. Copy example env:

```bash
cp /root/apps/donch/.env.autopar.example /root/apps/donch/.env.autopar
```

2. Edit `.env.autopar` and set:

- `AUTOPAR_PUBLISH_DIR=/path/to/shared/location`

## Notes

- Decisions are parsed from `journalctl -u donch.service` `META_DECISION` lines.
- Dedup rule: for duplicate `(symbol, decision_ts)`, latest row is exported.
- Trades are pulled from PostgreSQL `positions` with `status='CLOSED'` and `closed_at` inside the export window.
- `symbols_active.txt` is generated from `symbols.txt` minus `results/runtime/invalid_symbols.txt`.
- `live_decisions.csv` includes full `err` payload from `META_DECISION` lines.
- `live_decisions.csv` includes `reason_raw`, `reason_canonical`, `schema_fail_class`, `symbol_listed_at_utc`, `symbol_age_days`.
- Listing-age fields are derived from exchange instrument metadata (`load_markets`) rather than OHLCV inference.
- `schema_diagnostics.json` includes daily `schema_fail_rate`, top missing fields/symbols, and class breakdown.
- `reason_mapping.csv` contains daily counts for `reason_raw -> reason_canonical`.
- `reason_agreement.json` reports canonical reason agreement metrics for the run.
