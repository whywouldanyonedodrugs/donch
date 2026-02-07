Feature set (from `feature_manifest.json`). Cat: `regime_code_1d`, `markov_state_4h`, `regime_up`, `funding_regime_code`, `oi_regime_code`, `btc_risk_regime_code`, `risk_on`, `risk_on_1`, `S1_regime_code_1d`, `S2_markov_x_vol1d`, `S3_funding_x_oi`, `S4_crowd_x_trend1d`, `S5_btcRisk_x_regimeUp`, `S6_fresh_x_compress`. Numeric: `atr_at_entry`, `rs_pct`, `markov_prob_up_4h`, `vol_prob_low_1d`, `atr_1h`, `rsi_1h`, `adx_1h`, `vol_mult`, `atr_pct`, `days_since_prev_break`, `consolidation_range_atr`, `prior_1d_ret`, `rv_3d`, `don_break_level`, `don_dist_atr`, `asset_rsi_15m`, `asset_rsi_4h`, `asset_macd_*`, `asset_vol_1h`, `asset_vol_4h`, `gap_from_1d_ma`, `prebreak_congestion`, `eth_macd_*`, `oi_*`, `funding_*`, `btc_*`, `eth_*`, `recent_winrate_*`.

Below: only features with explicit code evidence are specified. Anything not fully evidenced is flagged at the end.

---

DONCHIAN BREAKOUT (entry signal context)

1. `don_break_level` (numeric)  
    Computed from 5m bars grouped into calendar days: daily high = max(high), daily close = last(close). Prior N-day Donchian upper = rolling max of daily high over `DON_N_DAYS`, shifted by 1 day (no look-ahead). Daily breakout flag = (daily close > prior N-day high). Then, to mirror live behavior (using the last completed daily bar), both the Donchian upper and breakout flag are shifted forward by +1 day and joined onto intraday bars for day D via the intraday day index; `don_break_level` is the shifted prior-N-day-high for that day.
    
2. `don_dist_atr` (numeric)  
    Not evidenced in the retrieved code chunks (flagged as missing).
    

---

ENTRY-QUALITY FEATURES (per-signal; OHLCV-only)

Common resampling alignment used here:

- 5m -> 1h resample: `resample_ohlcv(df, "1h")` uses OHLC: first/max/min/last and volume sum with `label="right", closed="right"` (this matters for timestamp alignment).
    
- Indicators are computed on the resampled frame and then forward-filled or mapped onto 5m timestamps (see below per feature).
    

3. `atr_1h` (numeric)  
    Computed as ATR on 1h OHLCV using `atr(df, ATR_LEN)` where true range is max of: (high-low), abs(high-prev_close), abs(low-prev_close), then Wilder-style EMA (`ewm(alpha=1/len, adjust=False, min_periods=len)`). Mapped/forward-filled back to 5m timestamps for signal rows.
    
4. `rsi_1h` (numeric)  
    Computed on 1h close using `rsi(close, RSI_LEN)` with delta split into gains/losses, Wilder EMA of gains/losses (`ewm(alpha=1/len, adjust=False, min_periods=len)`), RSI = 100 - 100/(1+RS). Forward-filled to 5m timestamps for signal rows.
    
5. `adx_1h` (numeric)  
    Computed on 1h OHLCV using `adx(df, ADX_LEN)`. DM+/DM- and TR are built from high/low moves and true range; smoothed via Wilder EMA; DI+/DI- formed; DX computed and smoothed to ADX via Wilder EMA. Forward-filled to 5m timestamps for signal rows.
    
6. `vol_mult` (numeric)  
    On 5m bars: rolling median of volume over `VOL_LOOKBACK_DAYS` (default 30) using `bars_per_day=288`, window `lb_bars = 288 * lookback_days`, `min_periods=max(5, lb_bars//10)`. `vol_mult = volume / rolling_median(volume)` (median zeros replaced with NaN).
    
7. `atr_pct` (numeric)  
    On 5m bars: `atr_pct = atr_1h_ffill / close` (ATR% of price).
    
8. `days_since_prev_break` (numeric)  
    On 5m bars: compute daily high series, compute daily Donchian upper = rolling max(daily high, N=`DON_N_DAYS`) shifted by 1 day; forward-fill this daily upper to 5m timestamps. Define `touch = (high_5m >= don_upper_5m)`. Track last touch timestamp by forward-filling the timestamps where touch is true; then `days_since_prev_break = (t - last_touch_time) / 86400`.
    
9. `consolidation_range_atr` (numeric), 10) `prior_1d_ret` (numeric), 11) `rv_3d` (numeric)  
    These are computed in the “entry quality” panel code (shared parity logic) and include:
    

- `consolidation_range_atr`: rolling (hi-lo) range over a pullback window, normalized by ATR(1h) mapped to 5m.
    
- `prior_1d_ret`: 1-day (288-bar) return computed from 5m close with look-ahead protection.
    
- `rv_3d`: rolling 3-day realized volatility computed from 5m log returns over 3*288 bars.  
    Exact formulas/shift conventions are in `fill_entry_quality_features.py` / `compute_entry_quality_panel` code path.
    

---

ASSET TECHNICAL FEATURES (per-symbol; OHLCV-only)

12. `asset_vol_1h` (numeric), 13) `asset_vol_4h` (numeric)  
    Compute on resampled closes:
    

- 1h: `ret1h = log(close_1h).diff()`, `vol1h = ret1h.rolling(20).std() * sqrt(20)`.
    
- 4h: `ret4h = log(close_4h).diff()`, `vol4h = ret4h.rolling(20).std() * sqrt(20)`.  
    Then aligned to 5m timestamps via `map_to_left_index(df5.index, volX)` and stored as the feature values.
    

14. `gap_from_1d_ma` (numeric)  
    On 5m: `ma1d = close_5m.rolling(288).mean()`. Gap is normalized by ATR(1h) mapped to 5m: `(close_5m - ma1d) / atr_1h_5m` with zero ATR treated as NaN.
    
15. `prebreak_congestion` (numeric)  
    On 5m: `close.pct_change().rolling(3*288).std()` (3-day rolling stdev of 5m simple returns).
    
16. `asset_rsi_15m`, `asset_rsi_4h`, `asset_macd_line_1h`, `asset_macd_signal_1h`, `asset_macd_hist_1h`, `asset_macd_slope_1h`, `asset_macd_line_4h`, `asset_macd_signal_4h`, `asset_macd_hist_4h`, `asset_macd_slope_4h`  
    Not evidenced in the retrieved code chunks (flagged as missing). (Note: MACD math exists in `_macd_components`, but the specific mapping to these feature names/timeframes was not present in the retrieved evidence.)
    

---

ETH 4H MACD CONTEXT (merged onto 5m signal rows)

17. `eth_macd_line_4h`, 18) `eth_macd_signal_4h`, 19) `eth_macd_hist_4h` (numeric)  
    From ETH parquet: 4h close = resample 4h (`label="right", closed="right"`, taking last close). MACD uses `fast=12`, `slow=26`, `signal=9`:
    

- `ema_fast = close_4h.ewm(span=fast, adjust=False, min_periods=fast).mean()`
    
- `ema_slow = close_4h.ewm(span=slow, adjust=False, min_periods=slow).mean()`
    
- `macd_line = ema_fast - ema_slow`
    
- `macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()`
    
- `macd_hist = macd_line - macd_signal`
    

20. `eth_macd_hist_slope_4h` (numeric)  
    Defined as Δhist per 4h bar: `macd_hist.diff()` on 4h bars, then forward-filled to 5m after merge (per docstring).
    
21. `eth_macd_hist_slope_1h` (numeric)  
    Defined analogously as Δhist per 1h bar, then forward-filled to 5m after merge (per docstring). The detailed 1h computation section was present in the retrieved chunks earlier in this thread.
    

---

OPEN INTEREST + FUNDING (per-symbol snapshot, and also used for BTC/ETH context when available)

22. `oi_level` (numeric)  
    If `open_interest` exists and `oi_level` not already set, `oi_level = open_interest` (type float).
    
23. `oi_notional_est` (numeric)  
    `oi_notional_est = oi_level * close` (both cast to float).
    
24. `oi_pct_1h`, 25) `oi_pct_4h`, 26) `oi_pct_1d` (numeric)  
    Percent changes of `oi_level` over fixed bar windows corresponding to 1h/4h/1d in 5m bars (constants `WIN_1H`, `WIN_4H`, `WIN_1D` in the same function).
    
25. `oi_z_7d` (numeric)  
    Z-score of `oi_pct_1d` using rolling 7d mean/std (windows `WIN_7D`, `min_periods=WIN_1D`): `(oi_pct_1d - mu_7d) / (sd_7d + 1e-12)`.
    
26. `oi_chg_norm_vol_1h` (numeric)  
    Normalized OI change by absolute 1h price move: `oi_pct_1h / (abs(close.pct_change(WIN_1H)) + 1e-12)`.
    
27. `oi_price_div_1h` (numeric)  
    OI-vs-price divergence over 1h: `oi_pct_1h - close.pct_change(WIN_1H)`.
    
28. `funding_rate` (numeric)  
    Funding is forward-filled: `funding_rate = funding_rate.astype(float).ffill()`.
    
29. `funding_abs` (numeric)  
    `abs(funding_rate)`.
    
30. `funding_z_7d` (numeric)  
    Rolling 7d z-score of `funding_rate`: `(funding_rate - mean_7d) / (std_7d + 1e-12)` with `min_periods=WIN_1D`.
    
31. `funding_rollsum_3d` (numeric)  
    Rolling 3d sum of `funding_rate`: `funding_rate.rolling(WIN_3D, min_periods=WIN_1D).sum()`.
    
32. `funding_oi_div` (numeric)  
    Interaction term: `funding_oi_div = funding_z_7d * oi_z_7d`.
    
33. `est_leverage` (numeric)  
    “Leverage proxy” computed from absolute z-scores: `(abs(oi_z_7d) + 0.5) * (abs(funding_z_7d) + 0.5)`.
    
34. `crowded_long`, 37) `crowded_short` (numeric int flags), 38) `crowd_side` (numeric int in {-1,0,1})  
    Uses config thresholds `CROWD_Z_HIGH` and `CROWD_Z_LOW` (defaults 1.0 and -1.0):
    

- `crowded_long = (oi_z_7d >= high) & (funding_z_7d >= high)`
    
- `crowded_short = (oi_z_7d >= high) & (funding_z_7d <= low)`
    
- `crowd_side = 1` if crowded_long, `-1` if crowded_short, else `0`
    

---

CROSS-ASSET CONTEXT (BTC/ETH attached to per-symbol panel)

39. `btc_vol_regime_level`, 40) `eth_vol_regime_level` (numeric)  
    Inside `_merge_cross_asset_context`:
    

- Resample BTC/ETH OHLCV to daily.
    
- Compute `atr1d = atr(daily, 20)` and `atr_pct_1d = atr1d / close`.
    
- Baseline = expanding median of `atr_pct_1d` with `min_periods=50`.
    
- `vol_regime_level = (atr_pct_1d / (baseline + 1e-12)).shift(1)` (explicit look-ahead protection via shift(1)).  
    This becomes `btcusdt_vol_regime_level` / `ethusdt_vol_regime_level` after prefixing (and is later read into model feature names).
    

41. `btc_trend_slope`, 42) `eth_trend_slope` (numeric)  
    Not fully evidenced: the code shown constructs daily `ma20` and `ma50` on daily close, but the final slope calculation and mapping to intraday timestamps were not present in the retrieved evidence chunk.
    
42. `btc_funding_rate`, `btc_oi_z_7d`, `eth_funding_rate`, `eth_oi_z_7d`  
    Only partially evidenced: `_merge_cross_asset_context` states it “adds btcusdt_/ethusdt_ prefixed oi_* and funding_* columns (when available)”, and the per-asset OI/funding computations are defined in `add_oi_funding_features`, but the exact merge/mapping lines for these specific prefixed columns were not present in the retrieved evidence chunk.
    

---

DAILY REGIME + MARKOV REGIME FEATURES (meta-model regimes)

44. `vol_prob_low_1d` (numeric), 45) `vol_regime_1d` (string used to derive codes), 46) `trend_regime_1d` (string), 47) `regime_code_1d` (categorical)  
    In `compute_daily_combined_regime` (daily bars):
    

- Volatility: `atr_pct = ATR(daily, atr_len) / close`; baseline = expanding median (min periods `base_days`); `vol_level = atr_pct / (baseline + 1e-12)`. A smooth probability is computed with a logistic transform around `vol_thr` and `vol_alpha`: `vol_prob_low = 1 / (1 + exp((vol_level - vol_thr) * vol_alpha))`. Then `vol_regime = LOW` if `vol_prob_low >= 0.5` else `HIGH`.
    
- Trend: compute fast/slow MAs over `ma_fast_days` / `ma_slow_days`, and use `trend_thr` on the fast MA slope; assigns regimes `UP`, `DOWN`, `FLAT`.
    
- Combined `regime_code` is built from (trend_regime, vol_regime) mapping defined in the same function.
    

48. `markov_state_4h` (categorical int), 49) `markov_prob_up_4h` (numeric)  
    In `compute_markov_regime_4h`, the function returns the last computed `state` and `prob_up` from an internally computed Markov regime table (`out`), exposed as `(state_up, prob_up)` (with safe fallbacks). The exact transition estimation details beyond “state/prob_up are computed into `out`” were not fully present in the retrieved chunk, so only the output semantics are strictly evidenced here.
    
49. `regime_up` (categorical)  
    Not evidenced: it is consumed as an input boolean to regime-set building, but its derivation (from daily regime outputs or otherwise) was not present in the retrieved evidence.
    

---

REGIME-SET / PRODUCT-SLICE CATEGORICAL FEATURES (used by the meta-model)

All of these are generated in `_meta_build_regime_sets` (backtester-side feature construction).

51. `oi_regime_code` (categorical in {-1,0,1})  
    Chooses `oi_val` as either `oi_z_7d` or `oi_pct_1d` depending on `oi_source` threshold config. Then buckets by terciles using thresholds `oi_q33`, `oi_q66`: `<=q33 -> -1`, `>=q66 -> +1`, else `0`.
    
52. `btc_risk_regime_code` (categorical in {0,1,2,3})  
    Compute `btc_trend_up = 1 if btc_trend_slope > 0 else 0`, `btc_vol_high = 1 if btc_vol_level >= btc_vol_hi else 0`, then `btc_risk_regime_code = btc_trend_up*2 + btc_vol_high`.
    
53. `risk_on` and 54) `risk_on_1` (categorical)  
    With `ru = 1 if regime_up else 0`, `risk_on = 1` iff `(ru==1 and btc_trend_up==1 and btc_vol_high==0)`, else `0`. `risk_on_1` is an alias equal to `risk_on`.
    
54. `S1_regime_code_1d` (categorical)  
    `S1 = float(regime_code_1d)` (direct copy).
    
55. `S2_markov_x_vol1d` (categorical)  
    Let `vol_code` be derived from `vol_regime_1d` (LOW→0, HIGH→1), else inferred from `vol_prob_low_1d` (<0.5→1 else 0), else inferred from `regime_code_1d` (codes in (0,2)→1 else 0). Then `S2 = markov_state_4h * 2 + vol_code`.
    
56. `S3_funding_x_oi` (categorical)  
    `S3 = (funding_regime_code + 1)*3 + (oi_regime_code + 1)` (3x3 coding). Requires both codes finite.
    
57. `S4_crowd_x_trend1d` (categorical)  
    `S4 = (crowd_side + 1)*2 + trend_code` (3x2 coding). The exact derivation of `trend_code` (mapping from daily trend regime strings to {0,1,2} or similar) was not present in the retrieved evidence chunk, so only the set-coding formula is strictly evidenced.
    
58. `S5_btcRisk_x_regimeUp` (categorical)  
    `S5 = btc_risk_regime_code*2 + ru`.
    
59. `S6_fresh_x_compress` (categorical)  
    Compute tercile buckets:
    

- `freshness_code = bucket_terciles(days_since_prev_break, fresh_q33, fresh_q66)` with outputs {0,1,2}
    
- `compression_code = bucket_terciles(consolidation_range_atr, comp_q33, comp_q66)` with outputs {0,1,2}  
    Then `S6 = freshness_code*3 + compression_code`.
    

Independent confirmation: the same S1–S6 regime-set definitions (and `risk_on`) are also constructed in the offline regime builder script, consistent with the backtester implementation.

61. `funding_regime_code` (categorical)  
    Not evidenced in retrieved chunks (the set-coding uses it, but the code that assigns it from `funding_rate`/thresholds was not present in the retrieved evidence).
    

---

RECENT PERFORMANCE FEATURES (online state-derived)

62. `recent_winrate_20`, 63) `recent_winrate_50`, 64) `recent_winrate_ewm_20` (numeric)  
    Defined in backtester meta-feature generation; computed from the online win/loss state with explicit `shift(1)` to avoid using the current trade outcome. The exact computation lines were present earlier in this thread’s retrieved evidence (function `_meta_recent_winrate_features`).
    

---

FEATURES WITH INSUFFICIENT / INCOMPLETE EVIDENCE (DO NOT IMPLEMENT FROM THIS NOTE)

Numeric:

- `atr_at_entry` (no evidenced assignment shown)
    
- `rs_pct` (RS table construction is evidenced, but the final `rs_pct` derivation + how it is joined to signals is not fully evidenced)
    
- `don_dist_atr`
    
- `asset_rsi_15m`, `asset_rsi_4h`
    
- `asset_macd_line_1h`, `asset_macd_signal_1h`, `asset_macd_hist_1h`, `asset_macd_slope_1h`
    
- `asset_macd_line_4h`, `asset_macd_signal_4h`, `asset_macd_hist_4h`, `asset_macd_slope_4h`
    
- `btc_trend_slope`, `eth_trend_slope` (only partial pre-computation evidenced)
    
- `btc_funding_rate`, `btc_oi_z_7d`, `eth_funding_rate`, `eth_oi_z_7d` (per-asset computations evidenced; cross-asset merge/mapping not fully evidenced)
    
- `markov_prob_up_4h` internal estimation details (only output semantics evidenced)
    

Categorical:

- `regime_up` (consumed; derivation not evidenced)
    
- `funding_regime_code` (used; derivation not evidenced)
    
- `S4_crowd_x_trend1d` trend-code mapping (only set-coding formula evidenced)
    

If you want this to be “single source of truth” for live implementation, the next step is to extract the missing feature blocks directly from the relevant functions (notably: RS join logic, `atr_at_entry` assignment at trade entry, `_build_feature_panel` for RSI/MACD features, `_merge_cross_asset_context` tail for trend slope + mapping, and the funding regime bucketing).


------


Additionally:

Here is the answer from the offline team:

"1. `funding_regime_code` (definition, thresholds, lookahead, storage)

Definition used by the offline pipeline (research/`02_make_regimes.py::_funding_regime` and mirrored in `backtester.py`):

* Input: `funding_rate` at the decision timestamp (5m grid).
* Threshold: `eps = funding_neutral_eps`.
* Buckets:

  * `-1` if `funding_rate <= -eps`
  * `0` if `-eps < funding_rate < eps`
  * `+1` if `funding_rate >= eps`
  * `NA` if missing/unparseable.

How `funding_neutral_eps` is computed (research/`02_make_regimes.py::pass1_estimate_cutpoints` + `pass2_build_regimes`):

* Pass1 collects a bounded random sample of `funding_rate` values from the trade store and computes quantiles on `abs(funding_rate)`.
* `eps = q20(abs(funding_rate))`, then clamped: `eps = max(eps, 1e-8)`.

Where the threshold is stored:

* **Not** in `thresholds.json`.
* Stored in `regimes_report.json` under:

  * `thresholds.funding_neutral_eps`

Lookahead/shift policy:

* `02_make_regimes.py` does not shift `funding_rate`; it assumes the `funding_rate` column in the trade store already follows the backtest “as-of” semantics at decision time.
* In the enrichment/build logic, funding is mapped “as-of” onto the 5m grid via forward-fill (details under #3).

To print the actual number used for this bundle/run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("results/meta_export/regimes_report.json")
thr = json.loads(p.read_text())["thresholds"]
print("funding_neutral_eps =", thr["funding_neutral_eps"])
PY
```

2. `oi_regime_code` (source selection, thresholds, storage)

Definition used by the offline pipeline (research/`02_make_regimes.py::_oi_regime_from_continuous` and mirrored in `backtester.py`):

* Input: a single continuous OI signal chosen by `oi_source`.
* Thresholds: `q33`, `q66`.
* Buckets:

  * `-1` if `oi_val <= q33`
  * `0` if `q33 < oi_val < q66`
  * `+1` if `oi_val >= q66`
  * `NA` if missing/unparseable.

Source selection (`oi_source`) is determined at regime-build time (research/`02_make_regimes.py::pass2_build_regimes`):

* Prefer `oi_z_7d` if pass1 was able to compute q33/q66 for it.
* Else fall back to `oi_pct_1d` if pass1 was able to compute q33/q66 for it.

Where the thresholds are stored:

* **Not** in `thresholds.json`.
* Stored in `regimes_report.json` under:

  * `thresholds.oi_source` (string: `"oi_z_7d"` or `"oi_pct_1d"`)
  * `thresholds.oi_q33`
  * `thresholds.oi_q66`

How q33/q66 are computed:

* Pass1 samples the chosen column (`oi_z_7d` and/or `oi_pct_1d`) from the trade store and computes the 0.33 / 0.66 quantiles.

To print the actual values used:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("results/meta_export/regimes_report.json")
thr = json.loads(p.read_text())["thresholds"]
print("oi_source =", thr["oi_source"])
print("oi_q33    =", thr["oi_q33"])
print("oi_q66    =", thr["oi_q66"])
PY
```

3. Prefixed BTC/ETH derivatives mapping onto per-symbol 5m decision rows

What exists offline (as produced by the feature builder, `scout.py`):

* The cross-asset context columns are written as prefixed series:

  * `btcusdt_funding_rate`, `btcusdt_oi_z_7d`, … (and other `btcusdt_oi_*`, `btcusdt_funding_*`)
  * `ethusdt_funding_rate`, `ethusdt_oi_z_7d`, … (and other `ethusdt_oi_*`, `ethusdt_funding_*`)

How they are produced and aligned (offline, `scout.py::_merge_cross_asset_context`):

* Load BTCUSDT/ETHUSDT parquet from `cfg.PARQUET_DIR/{BTCUSDT,ETHUSDT}.parquet`.
* Compute OI/funding features on the BTC/ETH 5m series via `add_oi_funding_features()`:

  * OI windows are in 5m bars (`WIN_1D=288`, `WIN_7D=7*288`, etc.).
  * Funding is explicitly forward-filled to avoid gaps: `df["funding_rate"] = df["funding_rate"].ffill()`.
* Merge onto each per-symbol feature timestamp (`ts5 = feat["timestamp"]`) using:

  * `map_to_left_index(ts5, series)` which is `series.reindex(ts5, method="ffill")`.

So the as-of rule is:

* For each decision timestamp `t` on the 5m grid, use the **last known BTC/ETH value at or before `t`** (pure forward-fill / as-of).

Decision timestamp convention used by backtest/meta replay:

* `backtester.py::_meta_store_load` floors the stored entry timestamp to 5m:

  * `entry_ts_5m = entry_ts.dt.floor("5min")`
* That is the timestamp key used for parity/replay lookups, i.e., “decision ts” is on the 5m grid (floor/open-time convention).

Notes on `btc_*` / `eth_*` unprefixed names:

* The online scorer layer (`bt_meta_online.py::_apply_aliases`) aliases:

  * `btcusdt_funding_rate -> btc_funding_rate`, `btcusdt_oi_z_7d -> btc_oi_z_7d`, etc.
  * `ethusdt_* -> eth_*`
* This is naming only; the time alignment semantics are as above.

4. `btc_risk_regime_code` + trend slopes (definitions, thresholds, feature-name mapping)

A) `btcusdt_trend_slope` / `btcusdt_vol_regime_level` (offline definition in `scout.py::_merge_cross_asset_context`)

Computed from BTCUSDT (and similarly for ETHUSDT) using only OHLCV:

* Build daily bars via `indicators.resample_ohlcv(ohlc, "1D")`.

  * Important: this resampler does not specify `label`/`closed`, so pandas defaults apply (label/closed = left/left).
* Compute:

  * `atr1d = atr(daily, 20)`
  * `atr_pct_1d = atr1d / daily["close"]`
  * `base = atr_pct_1d.expanding(min_periods=50).median()`
  * `vol_regime_level = (atr_pct_1d / (base + 1e-12)).shift(1)`
  * `ma20 = daily["close"].rolling(20).mean()`
  * `ma50 = daily["close"].rolling(50).mean()`
  * `trend_slope = (ma20 - ma50).diff().shift(1)`
* Map both daily series onto the 5m timestamps with forward-fill (`map_to_left_index`).

Written column names in the feature panel:

* `btcusdt_vol_regime_level`, `btcusdt_trend_slope`
* `ethusdt_vol_regime_level`, `ethusdt_trend_slope`

B) `btc_risk_regime_code` (offline definition in `02_make_regimes.py::_btc_risk_regime` and mirrored in `backtester.py`)

Inputs:

* BTC trend slope: read from either `btc_trend_slope` or (fallback) `btcusdt_trend_slope`.
* BTC vol regime level: read from either `btc_vol_regime_level` or (fallback) `btcusdt_vol_regime_level`.

Threshold:

* `btc_vol_hi` from `regimes_report.json["thresholds"]["btc_vol_hi"]`.

Buckets:

* `btc_trend_up = 1` if `btc_trend_slope > 0`, else `0`.
* `btc_vol_high = 1` if `btc_vol_regime_level >= btc_vol_hi`, else `0`.
* `btc_risk_regime_code = btc_trend_up*2 + btc_vol_high`, mapping:

  * `0 = DOWN_LOW`
  * `1 = DOWN_HIGH`
  * `2 = UP_LOW`
  * `3 = UP_HIGH`

How `btc_vol_hi` is computed and stored (`02_make_regimes.py`):

* Pass1 samples `btc_vol_regime_level` and computes quantiles.
* Pass2 chooses:

  * `btc_vol_hi = q66(btc_vol_regime_level)` if available, else `q50`, else `1.0`.
* Stored in `regimes_report.json` under:

  * `thresholds.btc_vol_hi`

To print the exact `btc_vol_hi` used:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("results/meta_export/regimes_report.json")
thr = json.loads(p.read_text())["thresholds"]
print("btc_vol_hi =", thr["btc_vol_hi"])
PY
```
"
