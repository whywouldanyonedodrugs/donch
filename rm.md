# DONCH Live Trading Bot — Technical Manual

> A deep-dive, ops-ready manual for running, maintaining, extending, and upgrading your DONCH crypto bot.
> Target audience: you (operator), future contributors, and SREs.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Secrets & Configuration](#secrets--configuration)
5. [Installation & First Run](#installation--first-run)
6. [How the Bot Works (Runtime Flow)](#how-the-bot-works-runtime-flow)
7. [Operating the Bot](#operating-the-bot)
8. [Running under systemd](#running-under-systemd)
9. [Live Diagnostics & Debugging](#live-diagnostics--debugging)
10. [Strategy Development](#strategy-development)
11. [Meta-Model (WinProb) Integration](#meta-model-winprob-integration)
12. [Risk, Sizing & Protective Orders](#risk-sizing--protective-orders)
13. [Database & Data Retention](#database--data-retention)
14. [Maintenance Playbook](#maintenance-playbook)
15. [Upgrading the Bot](#upgrading-the-bot)
16. [Troubleshooting Cookbook](#troubleshooting-cookbook)
17. [Security Notes](#security-notes)
18. [Appendix: Useful Commands](#appendix-useful-commands)

---

## Architecture Overview

**Key components:**

* **`live/live_trader.py`** — Orchestrator; runs loops for scanning, reporting, Telegram, equity tracking, and position management.
* **Strategy Engine (`live/strategy_engine.py`)** — Loads a **YAML strategy spec** and evaluates rule-based signals against live market data (OHLCV/indicators).
* **Indicators (`live/indicators.py`)** — Technical indicators (EMA, ATR, RSI, ADX, MACD, VWAP-stack, etc.).
* **Meta-Model (`live/winprob_loader.py`)** — Loads **LightGBM** classifier + **OneHotEncoder** + (optional) calibrator from exported artifacts; returns win probability per signal.
* **Exchange Proxy (`live/exchange_proxy.py`)** — Thin wrapper around **ccxt** (Bybit V5 linear “category=linear”, clientOrderId support).
* **DB (`live/database.py`)** — Async Postgres client, schema migrations, position journal, fills, equity snapshots.
* **Risk Manager** — Simple in-process guardrails (loss-streak kill switch, DD pause).
* **Telegram (`live/telegram.py`)** — Bot UI & commands.

**High level:** Rule-engine → builds a **Signal** → optional **meta-model gate** → **risk sizing** → **entry** → **OCO protection** → **journal** → **manage / trail / exit**.

---

## Repository Layout

```
apps/donch/
├─ live/
│  ├─ live_trader.py          # main runner (module: python -m live.live_trader)
│  ├─ strategy_engine.py      # YAML-driven rule evaluation
│  ├─ indicators.py           # TA utilities (EMA, ATR, MACD, VWAP-stack, ...)
│  ├─ winprob_loader.py       # LightGBM + OHE + calibrator loader/scorer
│  ├─ exchange_proxy.py       # ccxt adapter
│  ├─ database.py             # asyncpg schema & queries
│  ├─ filters.py              # legacy entry veto filters
│  ├─ telegram.py             # Telegram bot client
│  └─ shared_utils.py         # helpers (blacklist, etc.)
├─ strategies/
│  └─ donch_pullback_long.yaml   # default strategy spec (YAML)
├─ config.py                  # defaults (Python constants)
├─ config.yaml                # live overrides (runtime-reloadable)
├─ symbols.txt                # universe list (runtime-reloadable)
├─ listing_dates.json         # cache of listing dates (auto-generated)
├─ results/
│  └─ meta_export/            # meta-model artifacts (see below)
└─ .env                       # secrets (Bybit, DB, Telegram)
```

---

## Prerequisites

* **OS**: Ubuntu/Debian recommended (systemd present).
* **Python**: 3.12.x (virtualenv).
* **Packages**: gcc runtime for LightGBM (`libgomp1`).
* **PostgreSQL**: reachable via `DATABASE_URL` (e.g., `postgres://user:pass@host/db`).
* **Bybit**: API key/secret (Standard or Unified; linear perps).
* **Telegram**: bot token, chat id.

### System packages

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential libgomp1
sudo apt-get install -y postgresql-client
```

---

## Secrets & Configuration

### `.env` (required)

```
BYBIT_API_KEY=...
BYBIT_API_SECRET=...
BYBIT_TESTNET=false          # true for testnet
TG_BOT_TOKEN=...
TG_CHAT_ID=...               # numeric id (or @channel if supported by your bot)
DATABASE_URL=postgres://user:pass@host:5432/donch
```

> Permissions: `chmod 600 .env`

### `config.yaml` (runtime reloadable via watchdog)

Override `config.py` defaults here. Common keys:

```yaml
# Strategy
STRATEGY_SPEC_PATH: "strategies/donch_pullback_long.yaml"
TIMEFRAME: "5m"
EMA_TIMEFRAME: "4h"
RSI_TIMEFRAME: "1h"
ADX_TIMEFRAME: "1h"

# Regime detector
REGIME_BENCHMARK_SYMBOL: "BTCUSDT"
REGIME_CACHE_MINUTES: 60

# Universe filtering
RS_MIN_PERCENTILE: 70
VOL_MULTIPLE: 2.0
REGIME_BLOCK_WHEN_DOWN: true

# Risk
RISK_MODE: "fixed"          # "fixed" | "percent"
RISK_USD: 10                # per-trade base risk if fixed
RISK_EQUITY_PCT: 0.01       # if percent
MAX_LOSS_STREAK: 3
DD_PAUSE_ENABLED: true
DD_MAX_PCT: 10.0

# Meta-model
WINPROB_ARTIFACT_DIR: "results/meta_export"
META_PROB_THRESHOLD: 0.60   # gate trades below this probability
DEBUG_SIGNAL_DIAG: true     # rich per-symbol diagnostics in logs

# Sizing multipliers
WINPROB_SIZING_ENABLED: true
WINPROB_PROB_FLOOR: 0.55
WINPROB_PROB_CAP: 0.90
WINPROB_MIN_MULTIPLIER: 0.7
WINPROB_MAX_MULTIPLIER: 1.3

VWAP_STACK_SIZING_ENABLED: true
VWAP_STACK_MIN_MULTIPLIER: 1.0
VWAP_STACK_MAX_MULTIPLIER: 1.4

# Protection & exits
SL_ATR_MULT: 1.8
FINAL_TP_ENABLED: true
FINAL_TP_ATR_MULT: 8.0
PARTIAL_TP_ENABLED: false

# Loops
SCAN_INTERVAL_SEC: 60
SYMBOL_COOLDOWN_HOURS: 2

# Misc
TIME_EXIT_ENABLED: false
```

### `symbols.txt` (runtime reloadable)

One symbol per line, **Bybit linear** (e.g., `ETHUSDT`, `BTCUSDT`, `1000BONKUSDT`, etc.).

---

## Installation & First Run

```bash
cd ~/apps/donch
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# install your project deps (requirements.txt/pyproject.toml assumed)
pip install -r requirements.txt

# verify LightGBM runtime (libgomp1) is present; we installed it above.

# create DB schema on first run (migrations execute automatically)
python -m live.live_trader
```

You should see logs like:

* “StrategyEngine loaded …”
* “WinProb ready (kind=lgbm+ohe, features=31, p\*=0.60)”
* “Starting main signal scan loop.”
* “New market regime: …”
* “New scan cycle for N symbols …”

Stop with `Ctrl+C`.

---

## How the Bot Works (Runtime Flow)

1. **Startup**

   * Loads markets.
   * Loads listing dates (from cache or exchange).
   * Reconcilies DB vs Exchange positions; cancels orphan orders.
   * Kicks off concurrent loops (scan, manage positions, Telegram, equity, reporting).

2. **Main Scan Loop**

   * Gets **market regime** (cached for N minutes).
   * **Builds universe context** (RS percentile + median turnover) for all symbols.
   * Fetches ETH 4h MACD (barometer).
   * For each symbol:

     * Pulls OHLCV (5m base, 1h/1d plus required TFs).
     * Computes indicators.
     * Feeds data to **StrategyEngine** (YAML rules).
     * If `should_enter=True`:

       * Builds **meta-model** feature row; scores win probability.
       * Applies **filters** (belt & suspenders) and **meta gate**.
       * If accepted → proceeds to **\_open\_position**.

3. **Position Lifecycle**

   * Entry = market order with clientOrderId (CID).
   * Attaches protective orders (SL, TP1/TP2 if configured).
   * Journals to DB (position + fills).
   * Trailing logic if partial TP is hit.
   * Time exit / finalization flows update DB & notify Telegram.

---

## Operating the Bot

### Telegram Commands

* `/pause` — pause entries (management continues)
* `/resume` — resume if no kill switch
* `/status` — returns JSON of current status
* `/set KEY VALUE` — live-tune config (aliases mapped; e.g., `/set RISK_USD 15`)
* `/open SYMBOL` — force open (still respects most vetoes unless `FORCE_BYPASS_META=true`)
* `/report 6h|daily|weekly|monthly` — on-demand performance summary
* `/analyze` — triggers external weekly report script (if present)

### Hot Reloads

* Edits to **`config.yaml`** and **`symbols.txt`** auto-apply.
* Strategy spec file (YAML) hot-reloads; a Telegram notice is sent.

---

## Running under systemd

Create `/etc/systemd/system/donch.service`:

```ini
[Unit]
Description=DONCH Live Trader
After=network-online.target
Wants=network-online.target

[Service]
User=root
WorkingDirectory=/root/apps/donch
Environment="PYTHONUNBUFFERED=1"
ExecStart=/root/apps/donch/.venv/bin/python -m live.live_trader
Restart=on-failure
RestartSec=5

# Make sure the venv can import native libs
Environment="LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu"

# Ensure logs go to journald, not a TTY
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable donch
sudo systemctl start donch
sudo systemctl status donch -n 100
journalctl -u donch -f
```

> Stop/restart: `sudo systemctl stop donch` / `sudo systemctl restart donch`

---

## Live Diagnostics & Debugging

* **Per-symbol diagnostics:** set `DEBUG_SIGNAL_DIAG: true`.
  When a symbol passes the StrategyEngine gate, you’ll see a multi-line block with computed features and **GATES** (RS, liquidity, regime, volume spike, micro-ATR, META p vs p\*).
* **Why is nothing trading?**

  * Check **ETH MACD hist** + `REGIME_BLOCK_WHEN_DOWN`.
  * Confirm **RS percentile** and **VOL\_MULTIPLE** thresholds.
  * Ensure **META\_PROB\_THRESHOLD** isn’t too high for current market conditions.
* **Double logs?** Remove duplicate “Building universe context…” messages if you logged it both outside & inside `_build_universe_context()`.

---

## Strategy Development

Strategies live in `strategies/*.yaml` and are loaded by `StrategyEngine`. The engine reads your declared **side**, **required timeframes**, and rule blocks.

### Minimal Strategy Skeleton

```yaml
strategy:
  name: "Donchian Pullback"
  side: "long"                 # or "short"
  required_timeframes: ["5m", "1h", "1d"]

entry:
  donchian:
    length_days: 20            # uses 1d highs (shifted) for breakout level
    trigger: "close_above_break"   # or "rebreak_high"

  pullback:
    type: "retest"             # or "mean"

  volume:
    multiple: 2.0              # current 5m / rolling 30d median

  regime_gate:
    symbol: "ETHUSDT"
    timeframe: "4h"
    macd: [12, 26, 9]
    require_hist_positive: true
    require_macd_above_signal: true

filters:
  rs_percentile_min: 70
  liquidity_median_24h_usd_min: 500000

exits:
  sl_atr_mult: 1.8
  final_tp_atr_mult: 8.0
  time_exit_hours: null
```

> To use a new strategy: save it to `strategies/new_strategy.yaml` and set `STRATEGY_SPEC_PATH: "strategies/new_strategy.yaml"` in `config.yaml`. The bot hot-reloads the spec.

**Tips:**

* Keep rule semantics consistent with your backtest research (lookback windows, resampling, shift(1) to avoid look-ahead).
* Prefer **daily Donchian** with **prior-day shift** to prevent peeking.
* Only rely on indicators you compute identically in **live**.

---

## Meta-Model (WinProb) Integration

**Artifacts folder:** `results/meta_export/` with:

* `donch_meta_lgbm.joblib` — trained **LGBMClassifier**
* `ohe.joblib` — **OneHotEncoder** for categoricals
* `feature_names.json` — final column order (numerics + OHE names)
* `config_snapshot.json` — feature-building knobs at train time
* `pstar.txt` — decision threshold (e.g., `0.60`)
* `calibrator.joblib` — optional probability calibrator (isotonic/Platt)

**Loader:** `WinProbScorer(artifact_dir)` initializes on startup.
**Scoring:** `_scan_symbol_for_signal()` builds `meta_row` **at bar close** with features like:

* Numerics: `atr_1h`, `rsi_1h`, `adx_1h`, `atr_pct`, `don_break_len`, `don_break_level`, `don_dist_atr`, `rs_pct`, `hour_sin`, `hour_cos`, `dow`, `vol_mult`, `eth_macd_hist_4h`, `regime_up`, `prior_1d_ret` (optional).
* Categoricals: `entry_rule`, `pullback_type`, `regime_1d` (if available).

If `winprob.is_loaded` is false, the bot assigns `0.0` and continues (but trades will be gated if `META_PROB_THRESHOLD` > 0).

**Upgrading the model:** Drop a new artifact pack into a **new folder** (e.g., `results/meta_export_v2`) and point `WINPROB_ARTIFACT_DIR` to it in `config.yaml`, then **restart** the bot.
**Version locking:** To avoid `InconsistentVersionWarning` from scikit-learn, pin the same scikit-learn version used for training.

---

## Risk, Sizing & Protective Orders

* **Base risk**: fixed USD (`RISK_MODE: fixed`, `RISK_USD`) or % of equity (`RISK_MODE: percent`, `RISK_EQUITY_PCT`).
* **Multipliers** (all optional, controlled by `config.yaml`):

  * **ETH MACD barometer** (resize on unfavorable conditions).
  * **VWAP-stack multiplier** (fraction-in-band + expansion).
  * **WinProb multiplier** (map probability → \[min,max] range).
  * **YAML scalers** (generic feature → multiplier linear map).
* **Stops & Targets:**

  * `SL_ATR_MULT` (default `1.8`).
  * `FINAL_TP_ATR_MULT` (default `8.0`).
  * Optional **Partial TP** and **Trailing** after TP1.
  * Time-based exit optional (`TIME_EXIT_ENABLED`).

---

## Database & Data Retention

Tables (conceptually):

* **`positions`** — lifecycle of each trade (entry/exit, side, ATR, risk, protective order CIDs, etc.).
* **`fills`** — individual fills (entry, TP1, TP2, SL, etc.).
* **`equity_snapshots`** — periodic equity (for drawdown monitoring and reports).

Schema migrations run at startup (`db.migrate_schema()`).

**Backups:** use standard Postgres dumps, e.g.:

```bash
pg_dump $DATABASE_URL --format=custom --file=donch_backup_$(date +%F).dump
```

---

## Maintenance Playbook

* **Logs**: `journalctl -u donch -f` for streaming; configure logrotate if you redirect to files.
* **DB size**: prune old data if needed (e.g., keep 12 months).
* **Listings cache**: `listing_dates.json` generated automatically; safe to delete (it will be rebuilt).
* **Universe context**: built each scan cycle by default. If it becomes a bottleneck, consider external precompute or caching (e.g., cron job to write a JSON daily and let `_build_universe_context` fall back to it).
* **Kill switch**:

  * Loss-streak: trips after `MAX_LOSS_STREAK` losing trades (configurable). `/resume` only works if `risk.can_trade()` is true; otherwise lower `MAX_LOSS_STREAK` or reset via config change.
  * Drawdown pause (DD): if enabled, pauses entries when equity drawdown exceeds `DD_MAX_PCT`.

---

## Upgrading the Bot

1. **Create a maintenance window** (pause bot or stop service).

2. **Pull code** and **update venv**:

   ```bash
   cd ~/apps/donch
   git pull
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Pin ML versions** used to train the current model (`lightgbm`, `scikit-learn`).

4. **Validate**:

   * `python -m live.live_trader` in the foreground.
   * Confirm “WinProb ready … features=NN” and no feature mismatches.

5. **Restart systemd**:

   ```bash
   sudo systemctl restart donch
   ```

**Upgrading meta-model only:** place new artifacts, update `WINPROB_ARTIFACT_DIR`, restart.

---

## Troubleshooting Cookbook

### Strategy YAML not found

```
FileNotFoundError: /root/apps/donch/strategies/donch_pullback_long.yaml
```

* Ensure `STRATEGY_SPEC_PATH` in `config.yaml` points to the correct file.
* Path is relative to repo root (`/root/apps/donch`).

### WinProb failed to load: `libgomp.so.1`

* `sudo apt-get install -y libgomp1`

### Feature mismatch

```
ValueError: X has 29 features, but LGBMClassifier is expecting 31
```

* Two missing features in `meta_row`. Compare with `feature_names.json`.
* Ensure **categoricals** (`entry_rule`, `pullback_type`, `regime_1d`) exist and **OneHotEncoder** names are appended in order.
* Fill absent numerics with 0.0, not NaN.

### InconsistentVersionWarning (scikit-learn)

* Align runtime `scikit-learn` to the version used for **training** (e.g., 1.7.0 vs 1.7.1).
* Pin in `requirements.txt` and reinstall.

### “Not scanning / too quiet”

* Ensure you see: “New scan cycle …”, then “Checking SYMBOL…”.
* If StrategyEngine returns `should_enter=False` a lot, turn on `DEBUG_SIGNAL_DIAG: true` to see diagnostics for **passing** candidates; and temporarily add a debug log just before the `return None` for early vetoes if you need deeper visibility (already logs “skipped by StrategyEngine” at DEBUG).

### Duplicate “Building universe context…”

* Logged twice (caller + callee). Remove one of them or downgrade to DEBUG in the function.

### Bybit leverage/margin errors

* If you see “leverage not modified”, it’s informational; the bot continues.
* Ensure your account type matches (`BYBIT_ACCOUNT_TYPE: UNIFIED` if necessary; currently `_fetch_platform_balance` supports it).

---

## Security Notes

* **API keys** in `.env`; restrict file permissions. Use a **subaccount** with limited permissions.
* **Telegram** bot token grants control; keep it secret, rotate if leaked.
* Consider network egress allow-lists for Bybit/Telegram endpoints if hardening.

---

## Appendix: Useful Commands

### Foreground run (debug):

```bash
cd ~/apps/donch
source .venv/bin/activate
python -m live.live_trader
```

### Systemd:

```bash
sudo systemctl status donch -n 100
journalctl -u donch -f
sudo systemctl restart donch
sudo systemctl stop donch
```

### Quick sanity probe (single symbol):

Create a small script (or Python REPL) to call `_scan_symbol_for_signal` once, as you tested:

```python
import asyncio
from live.live_trader import LiveTrader, load_yaml, Settings

async def one():
    settings = Settings()
    cfg = load_yaml(Path("config.yaml"))
    t = LiveTrader(settings, cfg)
    await t.db.init(); await t.db.migrate_schema()
    await t.exchange._exchange.load_markets()
    t._listing_dates_cache = await t._load_listing_dates()
    reg = await t.regime_detector.get_current_regime()
    eth = await t._get_eth_macd_barometer()
    sig = await t._scan_symbol_for_signal("BTCUSDT", reg, eth)
    print("signal:", bool(sig), getattr(sig, "win_probability", None))

asyncio.run(one())
```

> (Use the exact helpers available in your codebase; above is illustrative.)

---

If you want this manual checked into the repo, save it as `docs/OPERATIONS.md` and link it from your README.
