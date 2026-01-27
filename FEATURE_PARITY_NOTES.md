# FEATURE_PARITY_NOTES.md
Strict-Parity Live Trading (DONCH) — Feature Semantics and Parity Contracts

This document is the single source of truth for “what a feature means” in the live stack, and what constraints must hold to claim parity with offline research exports.

Versioning note:
- Always identify which artifact bundle produced a given behavior. In live logs, use `bundle=<bundle_id>` and the SHA256 lines from startup validation.
- Any semantics that depend on thresholds must explicitly reference bundle artifacts (e.g., `deployment_config.json`, `regimes_report.json`, `thresholds.json`).

------------------------------------------------------------
1) Global invariants (non-negotiable)
------------------------------------------------------------

1.1 Decision timestamp (as-of semantics)
- The decision is made at `decision_ts`, defined in live as: `decision_ts = df5.index[-1]` where `df5` is the last completed 5-minute OHLCV dataframe used for the decision.
- IMPORTANT: In the current live implementation, OHLCV indices are the exchange-provided candle timestamps (typically bar OPEN time) and the final potentially-incomplete bar is dropped. Therefore `decision_ts` is the timestamp of the last completed bar index in that convention.
- Every feature must be computed strictly using information available as-of `decision_ts`. No wall-clock `now()` is allowed for features that feed the model.

1.2 No-lookahead and deterministic alignment
- Any higher-timeframe values (1h / 4h / 1d) used at 5m decision time must be mapped “as-of last completed HTF bar”.
- Rolling values must be computed as rolling series first, then read at `decision_ts` (never compute a scalar using future data).

1.3 Manifest-driven raw schema
- The model consumes a raw feature dict with exactly the keys in `feature_manifest.json` (currently 73 raw features in the deployed bundle).
- Live must fail closed on missing required raw keys. The scorer is strict: missing keys or type mismatches must prevent trading.

1.4 Bundle immutability
- Live must refuse to start if required artifacts are missing.
- Live must log SHA256 for each artifact at startup and attach bundle ID to decisions and trade logs.

------------------------------------------------------------
2) Meta-model scoring and gating semantics
------------------------------------------------------------

2.1 Raw → model matrix → probabilities
- The WinProb scorer:
  - Validates the raw feature dict against `feature_manifest.json` (required keys, dtypes/categories).
  - Builds the model input matrix in the fixed order expected by the trained model.
  - Produces `p_raw` and a calibrated `p_cal` (via `calibration.json` and optional isotonic file if present).
- If schema validation fails, meta gating must fail closed (no trade) and log the error.

2.2 Decision threshold and scope source
- Live reads:
  - `p*` from `deployment_config.json["decision"]["threshold"]`.
  - `decision.scope` from `deployment_config.json["decision"]["scope"]`.
- The META_DECISION log must include:
  - `p_cal`, `pstar`, `pstar_scope`, `scope_ok`, and raw scope inputs.

2.3 Scope semantics (confirmed with offline team)
Scope value selection when `decision.scope == "RISK_ON_1"`:
- Use `risk_on_1` if the column/key exists in the row.
- Otherwise (only if the key is absent), fallback to `risk_on`.
- There is NO row-wise fallback when `risk_on_1` exists but is NaN. NaN is treated as 0 → out-of-scope.
- Type handling:
  - Coerce to numeric with `to_numeric(errors="coerce")`, then `.fillna(0)`, then compare `== 1`.
  - Examples in-scope: 1, 1.0, "1", and typically True.
  - Examples out-of-scope: 0, 0.0, "0", 2, "true"/"false" (coerce to NaN → 0), None/NaN.
- Fail-closed:
  - Missing/unparseable values become 0 → scope_ok False.
  - If neither `risk_on_1` nor `risk_on` exists at all and scope is `RISK_ON_1`, offline raises an error; live must not silently pass.

------------------------------------------------------------
3) Feature families and exact live semantics (Sprint 1 status)
------------------------------------------------------------

This section documents the semantics currently implemented in live for the feature families that materially affect Sprint 1 acceptance (feature-building, scope inputs, and decision gating).

Note:
- Some feature computations call helpers located in other modules. This document specifies the callsite semantics (inputs, “as-of” constraints, and outputs). For full mathematical definitions, consult the implementation of those helper functions in the codebase.

3.1 Donchian / breakout-alignment features
- `don_break_level`:
  - Preferred source: injected `ctx["don_break_level"]` when finite (StrategyEngine patched to prefer injected level).
  - Fallback: derived from daily Donchian logic based on resampled daily bars and rolling max shifted by 1 day.
- `don_break_len`:
  - Comes from strategy context when available; live is robust to `None` (treated as non-tradable or defaulted at feature level depending on builder).
- `don_dist_atr`:
  - `(close_5m - don_break_level) / atr_1h`, with non-finite handling fail-closed (live currently uses a safe numeric fallback, typically 0.0 when not finite).

3.2 Entry-quality derived features (computed from frames as-of decision_ts)
The entry-quality feature builder computes a set of features using only historical bars up to `decision_ts`:
- Inputs:
  - `df5` (5m OHLCV), `df1h` (1h OHLCV), `decision_ts`, `cfg`.
- Output keys include (non-exhaustive; see feature_manifest.json for required subset):
  - `atr_1h`: ATR on 1h bars, length `cfg.ATR_LEN` (EMA-style).
  - `rsi_1h`: RSI on 1h closes, length `cfg.RSI_LEN` (EMA-style).
  - `adx_1h`: ADX on 1h bars, length `cfg.ADX_LEN`.
  - `atr_pct`: `atr_1h / close_5m`.
  - `vol_mult`: based on rolling median of 5m volume over a trailing lookback window, read at decision_ts.
  - `don_break_level`, `don_break_len`, `don_dist_atr`.
  - Other “entry quality” keys as provided by the builder.

Indicator definitions (as implemented in the shared indicator utilities):
- ATR:
  - True range: `max(high-low, abs(high-prev_close), abs(low-prev_close))`
  - ATR: EMA of TR with alpha = 1/len (adjust=False)
- RSI:
  - delta = diff(close)
  - up = clip(delta, lower=0), down = -clip(delta, upper=0)
  - EMA(up, alpha=1/len), EMA(down, alpha=1/len)
  - RSI = 100 - (100 / (1 + EMA(up)/EMA(down)))
- ADX:
  - Uses EMA smoothing with alpha=1/len for TR, +DM, -DM and then computes DX and ADX.

3.3 Time features (must be derived from decision_ts, not wall-clock)
- `hour_sin`, `hour_cos`:
  - Derived from `decision_ts.hour` with a 24-hour cycle.
- `dow`:
  - Day-of-week numeric derived from `decision_ts.dayofweek`.
- Any other calendrical/session features must follow the same rule: based only on decision_ts.

3.4 Regime-set augmentation and risk_on inputs (Sprint 1: scope correctness + observability)
- Live performs a post-assembly augmentation step that must run BEFORE manifest-row materialization:
  - `_augment_meta_with_regime_sets(meta_full)` is called after all prerequisite fields are placed into `meta_full` and before `meta_row = {k: meta_full.get(k, np.nan) ...}`.
- Current live augmentation produces:
  - `regime_trend_1d` from `regime_code_1d` and the trend regime thresholds (numeric, default 0).
  - `regime_vol_1d` from `vol_prob_low_1d` and the vol regime thresholds (numeric, default 0).
  - `risk_on` computed as:
    - 1 only when BOTH `(regime_trend_1d == 1)` AND `(regime_vol_1d == 1)`, else 0.
  - `risk_on_1` currently set as an alias of `risk_on` in the augmentation step.
- NOTE (Sprint 2): Exact parity for regime probabilities (`markov_prob_up_4h`, `vol_prob_low_1d`) and thresholds must be validated against offline golden exports. Sprint 1 acceptance only required that scope behavior matches offline and is observable.

3.5 VWAP stack features (callsite semantics)
- Live calls `vwap_stack_features(df5, lookback_bars=..., band_pct=...)` and writes:
  - `vwap_stack_frac`, `vwap_stack_expansion_pct`, `vwap_stack_slope_pph`.
- As-of requirement:
  - The df5 passed to vwap_stack_features must not include future bars beyond decision_ts.

3.6 Cross-asset ETH barometer features (watch item)
- Live computes ETH MACD(4h) for logging and/or feature inputs.
- Parity requirement:
  - If ETH MACD fields are model features, they must be computed strictly as-of decision_ts from ETH frames aligned to completed 4h bars.
- If any ETH feature is computed off wall-clock “latest fetch” rather than as-of decision_ts, that is a parity violation to be addressed in Sprint 2.

------------------------------------------------------------
4) Observability requirements (decision logs)
------------------------------------------------------------

4.1 META_DECISION log
Must include:
- `bundle`, `symbol`, `decision_ts`
- `schema_ok`, `p_cal`, `pstar`, `pstar_scope`
- Raw scope inputs: `risk_on_1` (raw), `risk_on` (raw)
- Scope evaluation debug:
  - `scope_val` (numeric value actually used after coercion)
  - `scope_src` (which column/key drove scope: risk_on_1 or risk_on)
  - `scope_ok` boolean
- Final decision:
  - `meta_ok`, `strat_ok`, `reason`, `err`

4.2 Safe-mode / fail-closed reasons
“No trade” must always carry a machine-readable reason code, such as:
- `schema_fail:<details>`
- `scope_fail:<scope_name>`
- `below_threshold`
- `scorer_error`
- `stale_data:<feed>`

------------------------------------------------------------
5) Tests that enforce parity contracts
------------------------------------------------------------

5.1 Unit parity tests
- `tests/test_parity.py`:
  - Ensures synthetic parity between offline replication and live computations for the covered subset.
- `tests/test_meta_scope.py`:
  - Validates the exact offline truth table for scope evaluation, including:
    - numeric/float/string coercion
    - alias fallback ONLY if risk_on_1 key is missing
    - no row-wise fallback when risk_on_1 exists but is NaN
    - missing both columns triggers error for RISK_ON_1 scope
    - unknown scope fails closed

5.2 Shadow scan tools (Sprint 1 acceptance)
- `tools/shadow_scan_meta_scope.py`:
  - Reads a parquet of rows with (symbol, timestamp/decision_ts, risk_on/risk_on_1, p_cal) and prints examples:
    - in-scope timestamps
    - above/below threshold examples near p*
  - Used to prove the threshold branch is evaluated in-scope (historically) without waiting for live market regime changes.

------------------------------------------------------------
6) Known gaps / Sprint 2 focus
------------------------------------------------------------

Sprint 2 must close parity for missing or incomplete feature families:
- Regime detector parity (daily regime + 4h Markov + vol regime probabilities).
- Regime-set codes and exact `risk_on` / `risk_on_1` parity with frozen thresholds and golden exports.
- OI/Funding feature family with staleness enforcement and as-of alignment.
- Cross-asset context features (BTC/ETH) computed strictly as-of decision_ts.
- Golden-row parity integration tests comparing live-built features at exported timestamps vs `golden_features.parquet` (categoricals exact; numerics within tolerance).

End of file.
