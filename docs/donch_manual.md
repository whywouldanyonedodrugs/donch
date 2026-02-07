
According to a document from February 7, 2026 (the DONCH live-trader code snapshot and project context available in this Project), the DONCH live bot is a strict-parity Bybit USDT-perps trader built around (1) a YAML rule-engine that proposes entries, (2) a manifest-driven meta-model that scores an offline-exported feature row, (3) scope + `p*` gating, and (4) a probability→risk-multiplier sizing curve. The design is “fail-closed”: if any required parity input is missing/stale/misaligned, the bot must skip rather than approximate. 

──────────────────────────────────────────────────────────────────────────────
DONCH LIVE TRADING BOT MANUAL (STRICT-PARITY)
──────────────────────────────────────────────────────────────────────────────

1. Core principles (what must never change)

A. Data clock (no wall-clock leakage)
All features are computed “as-of” an explicit event timestamp derived from the exchange data stream (not `now()`), called `decision_ts`. A cycle-level timestamp `decision_ts_gov` is computed for governance context (BTC/ETH) to prevent cross-asset lookahead. 

B. Last incomplete bar is never used
The last potentially-forming candle must be dropped deterministically before any indicator/feature computation. Anything else breaks offline→live parity.

C. Strict manifest schema for the meta row
The meta model input row is not “whatever features are convenient”; it must match `feature_manifest.json` exactly: same keys, same order, no extras, no missing. Any mismatch is a hard skip (fail-closed).

D. Deterministic caching keyed by timestamps (not TTL)
Gov context is cached per scan cycle using `decision_ts_gov` as the key; daily sub-features cache by `floor('D')`, Markov by `floor('4h')`, etc., to avoid variability across runs.

2. What DONCH consists of (module map)

Orchestration

* `live_trader.py`: main loop, config reload, exchange IO, strategy evaluation, meta feature construction/scoring, order placement, protective orders, and operational logging.

Offline artifact ingestion

* `artifact_bundle.py`: loads the exported offline bundle directory (manifest, thresholds, sizing curve, optional regime truth paths) and enforces strictness where configured. 

Feature construction (manifest-aligned)

* `feature_builder.py`: builds the meta feature row in strict parity semantics, including deterministic resampling and as-of slicing. It explicitly forbids using wall clock and requires an explicit `decision_ts`.

Market data IO + caching

* `exchange_proxy.py`: CCXT wrapper with paged OHLCV fetching and additional endpoints for OI/funding series.
* `ohlcv_store.py`: disk cache (SQLite) for OHLCV frames; enforces canonical schema and UTC index handling.

Derivatives features (OI/funding)

* `oi_funding.py`: computes OI/funding features strictly as-of `decision_ts` with staleness enforcement; if stale, raise/skip (fail closed). 

Meta model scoring + sizing curve

* `winprob_loader.py`: loads model/calibrator and parses `sizing_curve.csv` into an interpolatable probability→multiplier function; malformed curves fail. 

Online performance state

* `online_state.py`: persists rolling performance stats and can emit “as-of” features into `gov_ctx`.

Regimes parity (optional/when enabled)

* `regime_truth.py`: loads offline truth tables and performs “as-of” mapping with guardrails (no silent forward fill past truth range unless explicitly allowed). 

3. Required files and configuration

A. Environment secrets (from `.env`)
The live trader uses Pydantic settings with explicit env aliases:

* `BYBIT_API_KEY`, `BYBIT_API_SECRET`, `BYBIT_TESTNET`
* `TG_BOT_TOKEN`, `TG_CHAT_ID`
* `DATABASE_URL` (Postgres DSN)
  These are required to start (fail-fast). 

B. Runtime config files

* `config.yaml` (hot-reloaded)
* `symbols.txt`
  Paths are fixed in code defaults unless overridden in your repo/config conventions. 

C. Offline bundle directory (meta artifacts)
Typically `results/meta_export/`, containing at minimum:

* `feature_manifest.json` (ordered features)
* `thresholds.json` (scope + `p*` thresholds)
* `sizing_curve.csv` (prob→multiplier)
* model + calibrator artifacts required by `WinProbScorer`
  Optional but commonly present:
* regime truth parquet files (daily + markov4h)
  Loaded via `load_bundle(..., strict=True)` in strict-parity mode. 

4. Data clock and timestamp semantics (the most important part)

A. `decision_ts` (per symbol)
For each symbol scan, the bot fetches base-timeframe OHLCV (typically 5m), drops the last possibly-incomplete bar, then sets `decision_ts = df.index[-1]` (UTC, tz-aware). This `decision_ts` is the only timestamp allowed to drive features, cyclicals, gating, and order IDs.

B. `decision_ts_gov` (cycle-level governance timestamp)
Gov context fetches BTC and ETH 5m frames, then sets:
`decision_ts_gov = min(last_closed_ts(BTC), last_closed_ts(ETH))`
Both frames are hard-cut to `<= decision_ts_gov` before any cross-asset feature computation, preventing BTC/ETH lookahead mismatch. 

C. Resampling semantics (strict-parity)
Higher TF frames are derived by resampling base TF in a deterministic way (the parity implementation uses right-closed/right-labeled bars with OHLCV aggregation). This is encoded in the parity helpers used by FeatureBuilder and macro parity helpers. 

D. Daily context shift(1)
Cross-asset daily context (e.g., BTC/ETH daily vol-regime level and trend slope) is computed so that today’s value only becomes visible after the daily bar closes (shift(1)), and cached by day key (`floor('D')`). This is explicitly described/implemented in the gov context logic and parity helpers. 

5. Governance context (gov_ctx): what it contains and why

Gov context is computed once per scan cycle and reused across symbols; it includes:

* `decision_ts_gov` itself
* Cross-asset daily context for BTC and ETH (shift(1) daily features)
* BTC/ETH OI and funding features as-of `decision_ts_gov` (subject to staleness rules)
* Markov 4h snapshot (computed on a 4h bucket cache)
* Online performance features as-of `decision_ts_gov` (from `online_state.py`)

The cache key is the timestamp (`decision_ts_gov`), not wall-clock TTL. 

6. Feature construction: from OHLCV to manifest-aligned meta row

A. Rule-engine vs meta-model inputs
The strategy/rule engine uses multi-TF OHLCV and auxiliary context (listing age, liquidity checks, Donch levels, etc.) to decide “should_enter”.
The meta model uses a strictly-defined feature vector built from the same data clock.

B. Strict manifest alignment
`feature_manifest.json` is the source of truth for which features exist. Live must output exactly those features; tests ensure the manifest name extraction is stable.

C. No-lookahead Donch breakout level
The live scanner injects a no-lookahead Donch breakout level computed via a parity helper (daily construction without leaking the current day). This injected value is treated as canonical for diagnostics and downstream use.

D. OI/funding integration
OI/funding features can be used both at gov_ctx level (BTC/ETH) and per-symbol (depending on configuration). These features are explicitly as-of and must pass staleness checks.

7. OI/funding features and staleness rules

A. What’s computed
`oi_funding.py` computes features from OI and funding series aligned to `decision_ts` (e.g., z-scores over a lookback, latest funding, etc., depending on configuration).

B. Staleness enforcement (fail closed)
If the most recent derivative datapoint is older than the configured maximum age, the code raises a staleness error and the caller should skip trading rather than guessing. There is an explicit test ensuring the env-based staleness threshold is respected. 

8. Meta scoring, scope gating, and p* thresholds

A. Scoring
The meta bundle includes a pipeline and often a calibrator; the output probability used for gating is typically `p_cal`.

B. Scope gating + `p*`
`thresholds.json` defines the scope and thresholding logic. Live computes a scope value deterministically (as-of timestamps) and compares `p_cal` to `p*` for that scope. Any out-of-scope or below-threshold signal is skipped (fail closed).

9. Sizing curve (probability → multiplier)

The bot reads `sizing_curve.csv` and supports at least a “point curve” format where:

* a probability column is detected (`p_cal`, `p`, `prob`, etc.)
* a multiplier column is detected (`mult`, `multiplier`, etc.)
  Points are sorted by probability and stored for interpolation; missing columns or too-short curves raise errors rather than silently defaulting. 

10. Order ID stability (idempotency) and Bybit constraints

A. Stable entry clientOrderId
Entry orders use a stable deterministic client order ID derived from `(symbol, decision_ts, side, tag)` and then truncated to satisfy Bybit’s length constraints. This is implemented in `create_entry_cid`, and enforced by unit tests. 

B. Protective order IDs
SL/TP orders use similarly stable IDs with different tags to allow safe retries without duplication.

11. Operations runbook (what the operator does)

A. Start/stop (systemd)
On the deployment host (as used in your logs), typical commands:

* `systemctl start donch`
* `systemctl stop donch`
* `systemctl restart donch`
* `journalctl -u donch -f`

B. Manual run (non-systemd)

* Activate venv
* Ensure repo root is on `PYTHONPATH`
* Run the live entrypoint you use in your repo (commonly the module containing `main()`).

C. Pre-flight checklist

1. `.env` has all required keys (Bybit + Telegram + DB). 
2. Bundle directory exists and loads in strict mode. 
3. `feature_manifest.json` matches FeatureBuilder output exactly (run the manifest tests).
4. OI/funding staleness thresholds are set appropriately and tests pass. 
5. Logs show `decision_ts`/`decision_ts_gov` behavior consistent with the data clock model. 
6. `clientOrderId` stability tests pass. 

12) Troubleshooting (failures you should expect and how to handle them)

A. Feature schema mismatch
Symptom: meta scoring fails due to missing/extra columns.
Action: compare FeatureBuilder output keys against `feature_manifest.json`; fix in builder/inputs, not by “dropping columns” at scoring time. This is a strict-parity failure.

B. “Timestamp/index” errors in OHLCV ingestion
Symptom: KeyError or missing index/timestamp.
Action: fix at the ingestion boundary so OHLCV frames always have a UTC DatetimeIndex and expected columns; do not patch downstream.

C. OI/funding stale or missing
Symptom: staleness error / warnings / missing derivatives.
Action: verify exchange endpoints and pagination; keep staleness enforcement; do not widen thresholds without confirming offline spec. 

D. Regime truth out of range
Symptom: decision_ts beyond truth table max.
Action: refresh offline truth exports or explicitly allow stale truth only for controlled testing; default behavior should not silently forward-fill. 

13. Change management (how to extend safely)

1) Reproduce with pinned `(symbol, decision_ts)` (never debug with “latest”).
2) Create/refresh “golden rows” from offline for those timestamps.
3) Run parity diagnostics comparing live-built features to golden features.
4) Make the smallest change in the correct choke point (OHLCV ingestion, resampling, FeatureBuilder, gov_ctx, OI/funding).
5) Re-run unit tests relevant to that area (manifest, OI/funding, gov ctx signature, clientOrderId, regimes).

