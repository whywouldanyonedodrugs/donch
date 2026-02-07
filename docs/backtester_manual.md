I re-reviewed every Python module available in the current project bundle under `/mnt/data` and extracted the full structure, data contracts, execution logic, training/export pipeline, and the concrete refactor hazards. This manual is grounded in those on-disk files (not the earlier partial excerpts).

Scope note (non-optional): two critical dependencies referenced throughout the code are NOT present in this bundle, so no line-level description of their implementation is possible here. They are:

1. `shared_utils.load_parquet_data` (imported in many modules but not defined in `shared_utils.py` in this bundle).
    
2. `bt_intrabar.resolve_first_touch_1m` (imported by `backtester.py` but `bt_intrabar.py` is not included).
    

Because these are missing from the bundle, the manual includes (a) the exact call contracts inferred from usage, (b) required behavior, and (c) failure modes you must test for when pulling those missing files from the main repo. Everything else below is fully backed by code present in this bundle.

────────────────────────────────────────────────────────────

1. Project purpose (current “bull/optimistic momentum” bot)  
    ────────────────────────────────────────────────────────────  
    The current system is a long-only momentum/breakout strategy designed to perform in optimistic (risk-on) environments. It consists of:
    

A) Signal generation (“scout”):

- Uses a daily Donchian breakout confirmation (daily close > prior N-day high; no look-ahead via shifting).
    
- Requires a pullback “retest” around the breakout level within a lookback window, plus close above the level.
    
- Adds a volume spike gate, micro-volatility (ATR/price) filter, RS percentile gate, and liquidity gate.
    
- Produces a partitioned Parquet dataset of signals (hive-style: `signals/symbol=XYZ/part-....parquet`) with a large feature panel merged onto each signal timestamp.
    

B) Trade simulation (“backtester”):

- Reads signals (streamed from partitioned parquet or a single file).
    
- Applies throughput guards (dedup window, cooldown, max trades/day, max open).
    
- Optional meta gating and meta sizing:
    
    - Offline: merge `meta_p` from a predictions parquet via `merge_asof`.
        
    - Online: compute `meta_p` using the exported model bundle via `bt_meta_online` + `winprob_loader`.
        
- Simulates long trades with ATR-based SL/TP (and optional partial TP / trailing / AVWAP exit basis).
    
- Outputs trades to CSV/parquet and equity curves; can optionally log decision traces and lock timelines.
    

C) Research/training pipeline (scripts `00_...` through `09_...`):

- Takes cleaned trade exports (`results/trades.clean.csv`) and builds targets, regimes, feature screens, CV models, calibration/EV tables, TP/SL diagnostics, deployment artifacts, and HTML/PDF-like reports.
    

D) Enrichment scripts:

- Add funding/OI-derived features to trades post-hoc (Bybit), backfill additional context features, and compute entry-quality panels.
    

The bear-market team’s job is not just “invert signals”; the codebase is structurally long-biased in multiple locations: signal logic, simulation, derived features (e.g., `don_dist_atr`), and several regime definitions. Those long-only assumptions are enumerated explicitly below.

────────────────────────────────────────────────────────────  
2) Repository layout and module responsibilities (all files)  
────────────────────────────────────────────────────────────  
Core runtime:

- `config.py`: global configuration (paths, strategy params, risk, meta, throttles, regimes, caches).
    
- `shared_utils.py`: symbol list + blacklist helpers (but missing `load_parquet_data` in this bundle).
    
- `indicators.py`: MTF-safe indicator utilities (ATR, MACD hist, RSI, ADX, Donchian upper, rolling median/quantile, resampling).
    
- `scout.py`: generates long breakout/retest signals, builds per-symbol feature panels (OI/funding, cross-asset context, sentiment index), writes partitioned parquet.
    
- `backtester.py`: reads signals, applies throttle + meta gating/sizing, simulates long trades, writes outputs.
    
- `bt_meta_online.py`: online scoring helper that enforces strict feature schema and calls `WinProbScorer`.
    
- `winprob_loader.py`: loads model pipeline + calibrator artifacts and computes calibrated win probabilities.
    
- `regime_detector.py`: daily trend+vol regime (uses smoothed Markov probabilities = look-ahead risk) and 4h Markov regime (filtered, past-only).
    

Auxiliary feature generation / enrichment:

- `fill_entry_quality_features.py`: computes “entry quality” feature panel per symbol and provides `compute_entry_quality_panel` used by backtester.
    
- `enrich_trades_funding_bybit.py`: pulls Bybit funding history and enriches trade logs.
    
- `backfill_trade_features.py`: backfills additional trade-level features (BTC/ETH vol regime, slopes, leverage proxy, etc.).
    
- `sentiment_index.py`: builds a cross-sectional “sentiment index” parquet used optionally by `scout.py`.
    

Research pipeline:

- `00_load_qa.py`: QA report for `trades.clean.csv`.
    
- `01_make_targets.py`: generates target labels from trades.
    
- `02_make_regimes.py`: builds regime library and regime-set codes.
    
- `03_univariate_screens.py`: univariate screens inside regimes (cutpoints).
    
- `04_models_cv.py`: cross-validated modeling (purged/embargo CV).
    
- `05_calibration_ev.py`: calibration + expected value tables + sizing curves.
    
- `06_tp_sl_diagnostics.py`: TP/SL diagnostics across regimes.
    
- `07_export_deployment_artifacts.py`: exports model bundle (model, calibration.json, thresholds.json, sizing_curve.csv, feature_manifest.json, checksums).
    
- `08_make_executive_report.py`: executive HTML report and coverage diagnostics.
    
- `09_generate_report.py`: orchestrated report generator.
    

Parameter sweep / automation:

- `sweep_policy_settings_v2.py`: runs multiple variants via subprocess, loads scoped signals robustly (handles hive partitioning, Arrow tz comparisons).
    

────────────────────────────────────────────────────────────  
3) Data contracts (critical: this is what the bear team must preserve)  
────────────────────────────────────────────────────────────  
3.1 OHLCV parquet inputs (5m, optional 1m)  
The system expects per-symbol parquet files in:

- `cfg.PARQUET_DIR / f"{SYMBOL}.parquet"` (default `PROJECT_ROOT/parquet`).  
    Optional intrabar:
    
- `cfg.PARQUET_1M_DIR / f"{SYMBOL}.parquet"` (default `PROJECT_ROOT/parquet_1m`).
    

Required columns (5m):

- `open, high, low, close, volume`  
    Optional columns:
    
- `open_interest`, `funding_rate`
    

Timestamp handling (invariant):

- All modules normalize to tz-aware UTC timestamps.
    
- Several components assume a DatetimeIndex named `timestamp` OR a `timestamp` column convertible to UTC.
    

3.2 Signals dataset output (scout)  
Output location:

- `cfg.SIGNALS_DIR` (default `PROJECT_ROOT/signals`), written as hive partitions:
    
    - `signals/symbol=BTCUSDT/part-....parquet`  
        Some parts may not physically include a `symbol` column (partition-only); readers must restore it.
        

Signal base columns produced by `detect_signals_for_symbol`:

- `timestamp` (UTC)
    
- `symbol`
    
- `entry` (currently close price at signal bar)
    
- `atr` (ATR aligned to cfg.ATR_TIMEFRAME)
    
- `don_break_len` (days)
    
- `don_break_level`
    
- `pullback_type` (currently “retest”)
    
- `entry_rule` (currently “donch_yaml_v1”)
    
- `vol_spike` (bool)
    
- `rs_pct` (float or None)
    

Then `_process_one_symbol` merges `_build_feature_panel(sym)` on `(timestamp,symbol)` adding many more features, including:  
1h context:

- `atr_1h, rsi_1h, adx_1h, vol_mult, atr_pct, close, don_upper`  
    OI/funding (if available):
    
- `oi_level, oi_notional_est, oi_pct_1h, oi_pct_4h, oi_pct_1d, oi_z_7d, oi_chg_norm_vol_1h, oi_price_div_1h`
    
- `funding_rate, funding_abs, funding_z_7d, funding_rollsum_3d, funding_oi_div`
    
- `crowded_long, crowded_short, crowd_side, est_leverage`  
    Additional MTF indicators:
    
- `asset_rsi_15m, asset_rsi_4h`
    
- `asset_macd_line_1h, asset_macd_signal_1h, asset_macd_hist_1h, asset_macd_slope_1h`
    
- `asset_macd_line_4h, asset_macd_signal_4h, asset_macd_hist_4h, asset_macd_slope_4h`
    
- `asset_vol_1h, asset_vol_4h`
    
- `gap_from_1d_ma, prebreak_congestion`  
    Cross-asset context (BTC/ETH) columns are prefixed:
    
- `btcusdt_vol_regime_level, btcusdt_trend_slope, ethusdt_vol_regime_level, ethusdt_trend_slope`
    
- plus `btcusdt_oi_*` and `btcusdt_funding_*` if BTC has those columns; similarly for ETH.  
    Optional sentiment merge (if `results/sentiment_index.parquet` exists):
    
- `sent_rets_1h_z, sent_rets_1d_z, sent_oi_chg_1h_z, sent_oi_chg_1d_z, sent_funding_mean_1d, sent_funding_z_1d, sent_beta_risk_on`
    

ETH MACD overlay:

- `_eth_macd_full_4h_to_5m` appends:
    
    - `eth_macd_line_4h, eth_macd_signal_4h, eth_macd_hist_4h`
        
    - `eth_macd_hist_slope_4h, eth_macd_hist_slope_1h`
        
    - `eth_macd_both_pos_4h` (int)
        

Derived feature:

- `don_dist_atr = (close - don_upper)/ATR_scale` (long-oriented sign convention).
    

3.3 Trades output (backtester)  
`backtester.Trade` dataclass defines the canonical trade record fields written to `results/trades.parquet` and `results/trades.csv`.

Core:

- `trade_id, symbol, entry_ts, exit_ts, entry, exit, qty, side, sl, tp, exit_reason, atr_at_entry, regime_up, rs_pct, pullback_type, entry_rule, don_break_len, fees, pnl, pnl_R`
    

Diagnostics:

- `mae_over_atr, mfe_over_atr`
    

Regime diagnostics (daily regime + 4h Markov):

- `markov_state_4h, markov_prob_up_4h`
    
- `trend_regime_1d, vol_regime_1d, vol_prob_low_1d, regime_code_1d, regime_1d`
    

Snapshot of entry features (subset aligned to model training needs):

- `atr_1h, rsi_1h, adx_1h, vol_mult, atr_pct, days_since_prev_break, consolidation_range_atr, prior_1d_ret, rv_3d, don_break_level, don_dist_atr`
    
- asset indicators listed above
    
- eth macd overlay listed above
    
- oi/funding, crowding, leverage proxy
    
- btc/eth context fields in non-prefixed names:
    
    - `btc_funding_rate, btc_oi_z_7d, btc_vol_regime_level, btc_trend_slope`
        
    - `eth_funding_rate, eth_oi_z_7d, eth_vol_regime_level, eth_trend_slope`
        

This schema is what research scripts expect (especially after cleaning).

────────────────────────────────────────────────────────────  
4) Long-only assumptions that must be removed/parameterized for bear bot  
────────────────────────────────────────────────────────────  
A) Signal logic is “upper breakout then retest then close above level”:

- Daily breakout: `daily["close"] > prior N-day high` (scout).
    
- Retest checks a band around the upper level and requires current close > level.  
    Bear bot needs:
    
- Daily breakout: `daily["close"] < prior N-day low` (new Donchian lower).
    
- Retest: price revisits the level from below, and close < level on trigger.
    
- A symmetric `donchian_lower_days_no_lookahead` helper and equivalent “touch” logic.
    

B) Feature sign conventions are long-oriented:

- `don_dist_atr` is currently `(entry - don_break_level)/ATR` (in `winprob_loader._derive`) and `(close - don_upper)/ATR` (in scout).  
    Bear bot must define or standardize:
    
- Either invert sign for shorts or add a separate `don_dist_atr_signed` that depends on side.  
    If you reuse the same meta infrastructure, inconsistent sign conventions will silently break portability.
    

C) Execution simulator is long-only:

- `_simulate_long_with_partials_trail` and associated logic check SL using bar lows and TP using bar highs in a way specific to longs.
    
- “Immediate stop on entry bar” is long-only (`low <= sl`).  
    Bear bot needs a `_simulate_short...` implementation with:
    
- SL checks against highs; TP checks against lows.
    
- Spread modeling must invert bid/ask logic (current implementation is written as “bid_*” for long).
    
- Trailing logic and partial TP levels must invert.
    

D) Risk sizing helper `_size_from_risk` is direction-agnostic (uses absolute risk per unit), but trade PnL, MAE/MFE tracking, and exit conditions are direction-specific in simulation.  
Bear refactor should isolate:

- Position direction logic (entry→exit, fee application, pnl calculation, MAE/MFE) into a “side engine”.
    

────────────────────────────────────────────────────────────  
5) Regimes: what exists and what is unsafe  
────────────────────────────────────────────────────────────  
5.1 ETH MACD regime gate (used for trading)  
`backtester.RegimeGate` loads ETH 5m, resamples to `cfg.REGIME_TIMEFRAME` (default 4h), computes MACD and histogram, and defines “up” as:

- If `cfg.REGIME_REQUIRE_BOTH_POSITIVE`: `macd>signal AND hist>0`
    
- Else: `hist>0`  
    This is used inside the backtester to gate entries (`regime_up = self.regime.is_up(ts)`).
    

Bear bot likely wants the inverse:

- Trade only when hist<0 (and/or macd<signal), with optional “both negative” requirement.
    

5.2 Daily combined regime (trend + vol): LOOK-AHEAD PROBLEM  
`regime_detector.compute_daily_combined_regime`:

- Trend: triangular moving average + ATR channel (ok if computed from past).
    
- Vol regime: statsmodels MarkovRegression on daily returns BUT uses:
    
    - `results.smoothed_marginal_probabilities`  
        This is a forward-looking smoother (uses future data). If these daily regime outputs are used as features for meta modeling or gating, it introduces leakage.
        

In this codebase:

- Backtester computes daily regimes in `__init__` and then:
    
    - Uses them for per-trade regime diagnostics and for regime-set features in meta replay paths.  
        Therefore:
        
- If any meta training includes columns derived from this daily vol regime, the training set is contaminated.  
    Bear team must change this to filtered probabilities (past-only), analogous to how `compute_markov_regime_4h` already uses `filtered_marginal_probabilities`.
    

5.3 4h Markov regime (past-only) is safe  
`regime_detector.compute_markov_regime_4h`:

- Uses MarkovRegression on 4h log returns and explicitly uses `filtered_marginal_probabilities`.
    
- Then determines the “up” state by weighted mean return.  
    This is safe from look-ahead (subject to the usual resampling alignment).
    

────────────────────────────────────────────────────────────  
6) Meta model scoring: offline merge vs online scorer  
────────────────────────────────────────────────────────────  
6.1 Offline merge (backtester `_attach_meta_if_needed`)  
If `META_PROB_THRESHOLD` is set OR `META_SIZING_ENABLED` is True and signals lack `meta_p`, backtester:

- Loads `cfg.META_PRED_PATH` parquet
    
- Normalizes prediction columns to `(ts,sym,meta_p)`
    
- Floors timestamps to `cfg.META_MERGE_ROUND` (default `"5min"`)
    
- Deduplicates `(ts,sym)` by mean
    
- `merge_asof` backward with tolerance `cfg.META_MERGE_TOL` (default `"10min"`)
    

Bear bot note:

- The merge is side-agnostic, but if you train a bear model, you must keep the same merge semantics (rounding/tolerance) or your live/backtest parity will drift.
    

6.2 Online scoring (`bt_meta_online.py` + `winprob_loader.py`)  
If `cfg.BT_META_ONLINE_ENABLED` is True, backtester can compute `meta_p` on the fly by:

- Loading `model.joblib`, `feature_manifest.json`, and `calibration.json` from `cfg.META_MODEL_DIR` (`results/meta_export` default).
    
- Building a “meta_row” from the signal row + extra replay values.
    
- Enforcing strict schema when `cfg.META_STRICT_SCHEMA=True`:
    
    - Missing _keys_ (not NaN) cause a hard exception.
        
- Scoring with `WinProbScorer.score()` and applying the exported calibration method:
    
    - none: clip
        
    - sigmoid: `sigmoid(a*logit(p)+b)`
        
    - isotonic: sklearn isotonic regressor
        

Refactor hazard (bear):

- `winprob_loader._derive` currently derives `don_dist_atr` as `(entry - don_break_level)/ATR`, which is long-signed. If you keep the same feature name for shorts, the online scorer will compute a semantically inverted feature unless you redesign the schema.
    

────────────────────────────────────────────────────────────  
7) Scout feature engineering (what is added and why it matters to refactor)  
────────────────────────────────────────────────────────────  
7.1 RS / liquidity table is misnamed and daily (not weekly)  
`build_weekly_rs` actually builds DAILY rows:

- resamples to 1D
    
- computes 7-day return and 7-day rolling median USD volume proxy
    
- ranks cross-sectionally per day and stores `rs_pct` 0..100  
    Outputs:
    
- `results/rs_weekly.parquet` (misleading name)  
    Gates:
    
- `RS_MIN_PERCENTILE` and `RS_LIQ_MIN_USD_24H` enforced in `detect_signals_for_symbol`.
    

Bear team decision:

- Decide whether RS should favor “relative weakness” instead of strength (invert percentile) for short selection.
    
- If inverted, also invert liquidity filtering? (probably not; liquidity should remain a hard minimum).
    

7.2 OI/funding features  
`scout.add_oi_funding_features` creates:

- OI level, notional estimate, OI % change over 1h/4h/1d, OI z-score 7d.
    
- OI change normalized by 1h volume sum.
    
- `oi_price_div_1h = sign(ret_1h)*oi_pct_1h`.
    
- Funding z-score 7d, funding rollsum 3d, funding*oi interaction.
    
- “Crowded long/short” flags based on `cfg.CROWD_Z_HIGH/LOW`.
    

Bear model likely needs:

- “Crowded short” as a bullish contrarian vs bearish continuation depends on your thesis; keep both flags and let model learn.
    

7.3 Cross-asset context  
`_merge_cross_asset_context` adds for BTC and ETH:

- A daily ATR% regime level and trend slope (shift(1) to avoid same-day leakage intraday).
    
- Optional OI/funding features if those columns exist.  
    Bear team should keep:
    
- BTC/ETH regime as strong predictors of short edge (risk-off clustering).
    

7.4 Sentiment index integration is optional but wired  
If `results/sentiment_index.parquet` exists, scout merges selected sentiment columns on timestamp.  
Bear team can:

- Keep the same infrastructure and build a “risk-off” sentiment beta or inverted measure, but do not break schema if strict online scoring is used.
    

────────────────────────────────────────────────────────────  
8) Backtester: throughput guards, execution, and major caveats  
────────────────────────────────────────────────────────────  
8.1 Throughput guards (entry suppression)  
Backtester tracks per-symbol:

- `_lock_until`, `_open_until`, `_cooldown_until`, `_last_entry`  
    and per-day trade counts. It can also write `results/lock_timeline.csv`.
    

Configured by:

- `DEDUP_WINDOW_HOURS` (used in backtester)
    
- `SYMBOL_COOLDOWN_MINUTES`
    
- `MAX_TRADES_PER_DAY`
    
- `MAX_OPEN_POSITIONS`
    

Note: `DEDUP_BUSY_WINDOW_MIN` is used in scout dedup; `DEDUP_WINDOW_HOURS` is used in backtester dedup. They are separate and easy to confuse.

8.2 Execution simulation is long-specific

- Entry is aligned to the next available 5m bar (with some asof logic) and then slippage is applied as `entry_price *= 1.002` (hardcoded).
    
- SL/TP are computed from either price-ATR basis or AVWAP basis (configurable).
    
- Optional partial TP1, move SL to BE, and trailing.
    
- Intrabar tie-breaking machinery exists (`_resolve_intrabar` + `resolve_first_touch_1m`) but must be verified: the long simulator’s order of checks is deterministic and may not always use the intrabar resolver in the way you expect.
    

Bear refactor must:

- Implement a symmetric simulator; do not attempt to “hack” by negating prices.
    

8.3 Known noisy warning + performance hazard  
In backtester’s OI/funding feature extraction at entry, it computes:

- `mu = np.nanmean(win_fr)` even when `win_fr` is all-NaN, which triggers:
    
    - “RuntimeWarning: Mean of empty slice”  
        This was observed as repeated warnings in sweeps and can materially slow runs and spam logs. Fix is to guard with `np.isfinite(win_fr).any()` before nanmean/nanstd.
        

8.4 Daily regime table may be empty  
Backtester prints warning if daily regime build returns empty, which then sets all daily regime diagnostics to NaN.  
Root causes include missing benchmark parquet, timestamp issues, or the missing `load_parquet_data` implementation.

8.5 Two critical missing dependencies (must be restored from main repo)

- `shared_utils.load_parquet_data` is required for virtually everything (scout, backtester, regime detector, sentiment index, enrichment scripts).  
    Required behavior (inferred from callsites):
    
    - Parameters: `(symbol, start_date, end_date, drop_last_partial, columns)`
        
    - Must return a DataFrame with either a DatetimeIndex or a `timestamp` column.
        
    - Must respect date slicing and `drop_last_partial=True` to avoid partial day leakage.
        
    - Must support optional column selection (and should tolerate missing optional columns).
        
- `bt_intrabar.resolve_first_touch_1m(symbol, bar_ts, inside_levels_dict)`:
    
    - Should resolve which level (sl/tp/tp1/trail) is hit first using 1m bars.
        
    - Must be consistent with backtester’s tie-breaking options.
        

────────────────────────────────────────────────────────────  
9) Research pipeline (00–09) and deployment bundle  
────────────────────────────────────────────────────────────  
The research pipeline is structured to take `results/trades.clean.csv` (not created by these scripts; you must provide it) and produce model artifacts and reports.

Outputs and intent:

- `00_load_qa.py`: HTML QA report, schema checks.
    
- `01_make_targets.py`: targets parquet (win/time/tp/sl/quality labels).
    
- `02_make_regimes.py`: regimes parquet + regime-set codes:
    
    - risk_on is defined as: (regime_up==1) & (btc_trend_up==1) & (btc_vol_high==0).  
        Bear team should define risk_off analogs explicitly and ensure they do not rely on look-ahead regime components.
        
- `03_univariate_screens.py`: cutpoints by regime slices; fails fast if no cutpoints (writes diagnostics).
    
- `04_models_cv.py`: purged/embargo CV; writes `oof_predictions.parquet` and model comparison tables.
    
- `05_calibration_ev.py`: calibrators + EV tables + sizing curve CSV.
    
- `06_tp_sl_diagnostics.py`: regime-conditional TP/SL outcome analyses.
    
- `07_export_deployment_artifacts.py`: creates `meta_export/` bundle:
    
    - `model.joblib`
        
    - `feature_manifest.json` (authoritative training schema for online scoring)
        
    - `calibration.json`
        
    - `thresholds.json`
        
    - `sizing_curve.csv`
        
    - `deployment_config.json`
        
    - `checksums_sha256.json`
        
- `08_make_executive_report.py`: HTML report + unscored trade detection.
    
- `09_generate_report.py`: orchestration.
    

Bear team: if you train a bear-only model, export it into a separate model directory (e.g., `results/meta_export_bear`) and ensure the runtime points to it (`cfg.META_MODEL_DIR`), otherwise you will silently score bear signals with bull priors.

────────────────────────────────────────────────────────────  
10) Config: duplicates, missing keys, and full listing appendix  
────────────────────────────────────────────────────────────  
10.1 Duplicated assignments (later wins)  
The following keys are assigned multiple times in `config.py`:

- `ATR_LEN`
    
- `NOTIONAL_CAP_PCT_OF_EQUITY`
    
- `MAX_LEVERAGE`
    
- `DEDUP_BUSY_WINDOW_MIN`
    
- `SYMBOL_COOLDOWN_MINUTES`
    
- `MAX_TRADES_PER_DAY`
    
- `USE_INTRABAR_1M`
    
- `MAX_TRADES_PER_VARIANT`
    
- `MIN_EQUITY_FRACTION_BEFORE_ABORT`
    

Some of these are intentional (LABELING_MODE overrides), but others can cause confusion when refactoring. Treat the final effective value as the one after the `LABELING_MODE` block evaluation.

10.2 Config keys referenced in code but NOT defined in `config.py`  
These are accessed via `getattr(cfg, ...)` so they won’t crash, but they are “hidden knobs”:

- `BT_SIGNAL_BATCH_SIZE`
    
- `BT_CACHE_EQ_MAX_SYMBOLS`
    
- `META_TRADE_FEATURES_PATH`
    
- `REGIME_BENCHMARK_SYMBOL`
    
- `REGIME_MA_PERIOD`
    
- `REGIME_ATR_PERIOD`
    
- `REGIME_ATR_MULT`
    
- `RS_N_WORKERS`
    
- `SENTIMENT_INDEX_PATH`
    
- `SIMULATE_SPREAD_ENABLED`
    
- `SPREAD_PCT`
    
- `VOL_CAP_BARS`
    

Bear team should either add them explicitly to config or keep the defaults stable.

10.3 Full config assignment listing (line-ordered)  
Below is the complete set of variable assignments detected in `config.py` in this bundle. For duplicates, the entries are suffixed `#2` etc in occurrence order.

(If you need, I can also provide an “effective config” list showing only the last assignment per variable, but this is the no-omission version.)

APPENDIX A — `config.py` assignments (152 lines)  
L007 PROJECT_ROOT = Path(**file**).resolve().parent  
L008 PARQUET_DIR = PROJECT_ROOT / "parquet"  
L009 PARQUET_1M_DIR = PROJECT_ROOT / "parquet_1m"  
L010 SIGNALS_DIR = PROJECT_ROOT / "signals"  
L011 RESULTS_DIR = PROJECT_ROOT / "results"  
L012 SYMBOLS_FILE = PROJECT_ROOT / "symbols.txt"  
L016 START_DATE = "2023-01-01"  
L017 END_DATE = "2025-11-15"  
L020 N_WORKERS = max(1, (os.cpu_count() or 2) - 1)  
L021 SCOUT_BACKEND = "process"  
L022 IO_MEMORY_MAP = True  
L025 SCOUT_STREAMING = True  
L026 SCOUT_CLEAN_OUTPUT_DIR = True  
L027 SCOUT_ROW_GROUP_SIZE = 100_000  
L028 SCOUT_PARTITION_COLS = ["symbol"]  
L030 MAX_OPEN_POSITIONS = 9999  
L033 RS_ENABLED = True  
L034 RS_MIN_PERCENTILE = 70  
L035 RS_REBALANCE_ANCHOR_WEEKDAY = 0  
L036 RS_LIQ_MIN_USD_24H = 100000.0  
L037 MICRO_VOL_MIN = 0.0001  
L041 DONCH_BASIS = "days"  
L042 DON_N_DAYS = 20  
L043 DON_N_BARS = 5760  
L044 DON_CONFIRM_CLOSE_ABOVE = True  
L047 VOL_SPIKE_ENABLED = True  
L048 VOL_SPIKE_MODE = "multiple"  
L049 VOL_LOOKBACK_DAYS = 30  
L050 VOL_MULTIPLE = 2.0  
L051 VOL_QUANTILE_Q = 0.95  
L054 PULLBACK_MODEL = "retest"  
L055 PULLBACK_WINDOW_BARS = 12  
L056 PULLBACK_WINDOW_HOURS = 24  
L057 RETEST_EPS_PCT = 0.003  
L058 RETEST_LOOKBACK_BARS = 288  
L059 MEAN_MA_LEN = 20  
L060 MEAN_BAND_ATR_MULT = 0.5  
L061 ENTRY_RULE = "close_above_break"  
L064 ATR_TIMEFRAME = "1h"  
L065 ATR_LEN#1 = 14  
L068 REGIME_FILTER_ENABLED = True  
L069 REGIME_ASSET = "ETHUSDT"  
L070 REGIME_TIMEFRAME = "4h"  
L071 REGIME_MACD_FAST = 12  
L072 REGIME_MACD_SLOW = 26  
L073 REGIME_MACD_SIGNAL = 9  
L074 REGIME_REQUIRE_BOTH_POSITIVE = False  
L075 REGIME_BLOCK_WHEN_DOWN = False  
L076 REGIME_SIZE_WHEN_DOWN = 0.2  
L079 INITIAL_CAPITAL = 2000  
L081 RISK_MODE = "cash"  
L082 RISK_PCT = 0.01  
L083 FIXED_RISK_CASH = 100.0  
L084 NOTIONAL_CAP_PCT_OF_EQUITY#1 = 0.25  
L086 MAX_LEVERAGE#1 = 10.0  
L087 FEE_RATE = 0.00055  
L088 ATR_LEN#2 = 14  
L089 SL_ATR_MULT = 2.0  
L090 TP_ATR_MULT = 8.0  
L091 TIME_EXIT_HOURS = 72  
L094 DYN_EXITS_ENABLED = False  
L095 DYN_MACD_HIST_THRESH = 0.0  
L096 DYN_TP_MULT_POS = 1.15  
L097 DYN_SL_MULT_POS = 0.90  
L098 DYN_TP_MULT_NEG = 0.85  
L099 DYN_SL_MULT_NEG = 1.15  
L102 PARTIAL_TP_ENABLED = False  
L103 PARTIAL_TP_RATIO = 0.5  
L104 PARTIAL_TP1_ATR_MULT = 5.0  
L105 MOVE_SL_TO_BE_ON_TP1 = False  
L107 TRAIL_AFTER_TP1 = False  
L108 TRAIL_ATR_MULT = 1.0  
L109 TRAIL_USE_HIGH_WATERMARK = False  
L112 EXIT_BASIS = "price_atr"  
L113 AVWAP_MODE = "static"  
L114 AVWAP_ANCHOR = "breakout"  
L115 AVWAP_SL_MULT = 2  
L116 AVWAP_TP_MULT = 8.0  
L117 AVWAP_USE_ENTRY_ATR = True  
L121 DEDUP_BUSY_WINDOW_MIN#1 = 120  
L122 SYMBOL_COOLDOWN_MINUTES#1 = 240  
L123 DEDUP_WINDOW_HOURS = 2  
L124 MAX_TRADES_PER_DAY#1 = None  
L127 USE_INTRABAR_1M#1 = False  
L128 TIE_BREAKER = "sl_wins"  
L132 BT_CACHE_5M_MAX_SYMBOLS = 6  
L133 BT_CACHE_1M_MAX_SYMBOLS = 2  
L136 BT_DOWNCAST_FLOAT32 = True  
L139 BT_DECISION_LOG_ENABLED = False  
L142 MAX_TRADES_PER_VARIANT#1 = 10000000  
L143 MIN_EQUITY_FRACTION_BEFORE_ABORT#1 = 0.05  
L144 MIN_ATR_PCT_OF_PRICE = 0.0001  
L147 SAVE_TRADES_CSV = True  
L148 SAVE_EQUITY_CSV = True  
L149 AGGREGATE_BY = ["symbol","pullback_type","entry_rule","don_break_len","regime_up"]  
L152 RSI_LEN = 14  
L153 ADX_LEN = 14  
L154 RSI_TIMEFRAME = "1h"  
L155 ADX_TIMEFRAME = "1h"  
L156 DON_DIST_IN_ATR = True  
L159 MARKOV_TIMEFRAME = "4h"  
L160 MARKOV_P_STAY = 0.95  
L161 MARKOV_RANDOM_STATE = 0  
L162 MARKOV_PROB_EWMA_ALPHA = 0.2  
L164 MARKOV4H_PROB_EWMA_ALPHA = 0.2  
L167 LABELING_MODE = False  
L170 DEDUP_BUSY_WINDOW_MIN#2 = 120  
L171 SYMBOL_COOLDOWN_MINUTES#2 = 240  
L172 MAX_TRADES_PER_DAY#2 = 100  
L175 USE_INTRABAR_1M#2 = False  
L180 DEDUP_BUSY_WINDOW_MIN#3 = 0  
L181 SYMBOL_COOLDOWN_MINUTES#3 = 0  
L182 MAX_TRADES_PER_DAY#3 = None  
L183 MAX_TRADES_PER_VARIANT#2 = 10_000_000  
L184 MIN_EQUITY_FRACTION_BEFORE_ABORT#2 = 0.0  
L185 USE_INTRABAR_1M#3 = False  
L188 NOTIONAL_CAP_PCT_OF_EQUITY#2 = 1.0  
L189 MAX_LEVERAGE#2 = 1000.0  
L193 META_PROB_THRESHOLD = None  
L194 META_SIZING_ENABLED = True  
L195 META_SIZING_P0 = 0.60  
L196 META_SIZING_P1 = 0.90  
L197 META_SIZING_MIN = 0.01  
L198 META_SIZING_MAX = 1.00  
L199 SIZE_MIN_CAP = 0.01  
L200 SIZE_MAX_CAP = 1.00  
L201 REGIME_DOWNSIZE_MULT = 1.0  
L206 META_GATE_SCOPE = "all"  
L207 META_GATE_FAIL_CLOSED = False  
L212 BTC_VOL_HI = 0.753777980804443  
L213 RISK_OFF_PROBE_MULT = 0.01  
L216 WEEK_PATTERN_ENABLED = False  
L217 WEEK_PATTERN = "10"  
L220 LIVE_DERISK_ENABLED = False  
L221 DERISK_TARGET_MULT = 0.65  
L222 DERISK_DOWNSHIFT_ONLY = False  
L223 DERISK_HYST = 0.02  
L224 DERISK_MIN_QTY_FRAC = 0.10  
L225 DERISK_COOLDOWN_BARS = 12  
L229 META_PRED_PATH = RESULTS_DIR / 'meta_export' / 'oos_predictions_calibrated.parquet'  
L233 META_MERGE_ROUND = "5min"  
L234 META_MERGE_TOL = "10min"  
L239 PORTFOLIO_RISK_CAP_PCT = 0.1  
L240 GROSS_EXPOSURE_CAP_MULT = 3.0  
L242 ON_CAP_BREACH = "scale"  
L249 CROWD_Z_HIGH = 1.0  
L250 CROWD_Z_LOW = -1.0  
L254 BT_META_ONLINE_ENABLED = True  
L258 META_MODEL_DIR = RESULTS_DIR / "meta_export"  
L265 REGIME_SLOPE_FILTER_ENABLED = False  
L266 REGIME_SLOPE_MIN = 0.0  
L268 META_STRICT_SCHEMA = True

────────────────────────────────────────────────────────────  
11) Concrete refactor guidance (bear bot) — minimum safe plan  
────────────────────────────────────────────────────────────  
If you want a refactor that does not create hidden long-bias regressions:

Step 1: Make direction a first-class concept everywhere.

- Add `side ∈ {"long","short"}` to signals and to all derived features with directional meaning.
    
- Split or parameterize:
    
    - Donchian breakout detector (upper vs lower).
        
    - Retest condition (above/below).
        
    - `don_dist_atr` and other distance features (signed).
        
    - Execution simulator (long vs short).
        

Step 2: Fix regime leakage before training bear meta models.

- Replace daily vol regime’s `smoothed_marginal_probabilities` with filtered.
    
- Rebuild regimes parquet and ensure no forward-looking columns leak into training features.
    

Step 3: Keep schema parity stable for online scorer.

- For bear model export, produce a separate `feature_manifest.json` and keep strict schema enabled.
    
- Do not reuse bull schema with inverted semantics under the same feature names.
    

Step 4: Validate parity with controlled tests.

- Use a small dataset window and:
    
    - Run offline-merged meta and online-scored meta; verify `meta_p` matches within tolerance.
        
    - Verify signal timestamps align after flooring/merge_asof.
        
    - Validate that partitioned signals missing physical `symbol` are reconstructed in all readers.
        

────────────────────────────────────────────────────────────  
12) Remaining incompleteness (what cannot be completed from this bundle)  
────────────────────────────────────────────────────────────  
Two items cannot be fully documented line-by-line because the source is missing from the provided files:

13. `shared_utils.load_parquet_data`: must be pulled from the main repo; this manual provides contract + required behavior but not code specifics.
    
14. `bt_intrabar.py`: required for 1m tie resolution; absent here.
    

If you provide those two missing modules (or paste their contents), I can incorporate them into this manual with the same level of detail as the rest, including exact pitfalls and bear-specific refactor notes.