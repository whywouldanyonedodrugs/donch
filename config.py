# config.py
"""
Strategy & risk defaults.
Edit in Git – for one‑off tweaks use config.yaml or Telegram `/set KEY value`.
"""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_CSV_DIR  = PROJECT_ROOT / "raw_csv"
PARQUET_DIR  = PROJECT_ROOT / "parquet"
SIGNALS_DIR  = PROJECT_ROOT / "signals"
RESULTS_DIR  = PROJECT_ROOT / "results"
SYMBOLS_FILE = PROJECT_ROOT / "symbols.txt"
MERGED_CSV_DIR = PROJECT_ROOT / "data_merged"

BYBIT_ACCOUNT_TYPE    = "UNIFIED"  # or "STANDARD"

START_DATE = "2022-01-01"
END_DATE = "2026-01-01"

EMA_FILTER_ENABLED = False
EMA_TIMEFRAME   = "4h" 
EMA_FAST_PERIOD = 20   # 20
EMA_SLOW_PERIOD = 200 # 200

RSI_PERIOD      = 14
RSI_TIMEFRAME   = "1h"
ADX_PERIOD      = 14
ADX_TIMEFRAME   = "1h"

RSI_ENTRY_MIN   = 30
RSI_ENTRY_MAX   = 95
RSI_FILTER_ENABLED = True
RSI_EPS           = 0.5

ADX_FILTER_ENABLED = False  # enable in YAML/Telegram if desired
ADX_MIN           = 10       # skip if trendless
ADX_MAX           = 25       # skip if strongly trending

PRICE_BOOM_PERIOD_H = 24
PRICE_BOOM_PCT      = 0.15
PRICE_SLOWDOWN_PERIOD_H = 4
PRICE_SLOWDOWN_PCT      = 0.01

# ── risk sizing ────────────────────────────────────────────────────────────
RISK_MODE          = "FIXED"     # or "FIXED"
FIXED_RISK_USDT    = 10.0

RISK_PCT           = 0.05         # 0.5 %
SLIPPAGE_BUFFER_PCT = 0.0005

#WINPROB_SIZE_FLOOR = 1.00
#WINPROB_SIZE_CAP   = 1.00

# ── stop / targets ─────────────────────────────────────────────────────────
SL_ATR_MULT             = 2.0
FINAL_TP_ENABLED        = True
FINAL_TP_ATR_MULT       = 8.0
PARTIAL_TP_ENABLED      = False
PARTIAL_TP_PCT          = 8.0       
PARTIAL_TP_ATR_MULT     = 1.0
TRAIL_ENABLED           = False
TRAIL_START_ATR_MULT    = 1.0
TRAIL_DISTANCE_ATR_MULT = 1.0
TRAIL_MIN_MOVE_PCT      = 0.001


TIME_EXIT_HOURS_ENABLED = True     # Set to True to enable hourly exit
TIME_EXIT_HOURS         = 4        # The number of hours a trade can run
TIME_EXIT_ENABLED       = True     # This now acts as a fallback if hourly is disabled
TIME_EXIT_DAYS          = 10       # Fallback: close runner after N days

# --- Coin Age Filter (Applied in filters.py) ---
MIN_COIN_AGE_DAYS = 14  # Min days since listing
MAX_COIN_AGE_DAYS = 9999 # Max days since listing (e.g., 5 years)

# --- Structural Trend Filter (Long-Term) ---
STRUCTURAL_TREND_FILTER_ENABLED = False 
STRUCTURAL_TREND_DAYS    = 30
STRUCTURAL_TREND_RET_PCT = 0.01

# ── VWAP Stack ─────────────────────────────────────────────
# --- Legacy ---
GAP_FILTER_ENABLED = False
GAP_VWAP_HOURS  = 2
GAP_MAX_DEV_PCT = 1.00
GAP_MIN_BARS    = 3
# VWAP_LEN = 15

# --- Working ---
VWAP_STACK_SIZING_ENABLED = True

# how we compute the features
VWAP_STACK_LOOKBACK_BARS = 12        # on 5m bars ≈ 1 hour
VWAP_STACK_BAND_PCT = 0.004          # ±0.40% “near value” band
VWAP_STACK_EXPANSION_ABS_MIN = 0.006 # 0.6% – below this, treat as very weak
VWAP_STACK_EXPANSION_GOOD = 0.015    # 1.5% – good expansion
VWAP_STACK_FRAC_MIN = 0.50           # at least half the window near VWAP
VWAP_STACK_FRAC_GOOD = 0.70          # solid consolidation

# convert “signal quality” → risk scaling
VWAP_STACK_MIN_MULTIPLIER = 0.80     # never risk below 20% of base (still “sizing filter”, not a veto)
VWAP_STACK_MAX_MULTIPLIER = 1.00     # cap at base risk for now (can raise later to 1.2)
VWAP_STACK_FRAC_WEIGHT = 0.6         # weight on consolidation quality
VWAP_STACK_EXP_WEIGHT = 0.4          # weight on expansion

# --- Volume Filter (Applied during Scouting) ---
VOL_FILTER_ENABLED  = False
MIN_VOL_USD         = 10000    # Min 4-hour USD volume
MAX_VOL_USD         = 5000000000  # Max 4-hour USD volume

# --- Volatility Filter (Applied during Scouting) ---
VOLATILITY_FILTER_ENABLED = False
MIN_ATR_PCT               = 0.005 # Min ATR as a percentage of price

# ── portfolio / kill‑switches ─────────────────────────────────────────────
MAX_OPEN            = 30          # max concurrent positions
MAX_LOSS_STREAK     = 10
KILL_EQUITY_LEVEL   = 200.0        # USDT
MIN_EQUITY_USDT     = 100.0
MAX_LEVERAGE        = 10
MIN_NOTIONAL        = 0.0000001         # exchange min order size
MIN_STOP_DIST_USD   = 0.0002
MIN_STOP_DIST_PCT   = 0.000008      # 0.08 %

# draw‑down pause
DD_PAUSE_ENABLED    = True     # set False to disable drawdown protection  # <--- ADD THIS
DD_MAX_PCT          = 50.0     # Pause trading if equity drops 10% from peak # <--- ADD THIS
DD_COOLDOWN_PCT     = 50.0     # % from equity peak
DD_COOLDOWN_DURATION_H = 6


# ── BTC / ALT EMA trend veto ──────────────────────────────────────────────
BTC_FAST_FILTER_ENABLED = False
BTC_FAST_TIMEFRAME      = "5m"
BTC_FAST_EMA_PERIOD     = 20
BTC_SLOW_FILTER_ENABLED = False
BTC_SLOW_TIMEFRAME      = "1h"
BTC_SLOW_EMA_PERIOD     = 200

ALT_SYMBOL              = "ETHUSDT"
ALT_FAST_FILTER_ENABLED = False
ALT_FAST_TIMEFRAME      = "5m"
ALT_FAST_EMA_PERIOD     = 20
ALT_SLOW_FILTER_ENABLED = False
ALT_SLOW_TIMEFRAME      = "1h"
ALT_SLOW_EMA_PERIOD     = 200

# ── Boom‑Volume rule ───────────────────────────────────────────────────────
BV_RULE_ENABLED   = False
BV_BUCKETS        = 5             # 1..5 buckets from tester
BV_MIN_VOL_SPIKE  = [1.2, 1.4, 1.6, 1.8, 2.0]
BV_MAX_VOL_SPIKE  = [6.0, 4.5, 3.5, 2.5, 2.0]

# ── Open‑interest confirmation ────────────────────────────────────────────
OI_FILTER_ENABLED        = False
OI_LOOKBACK_PERIOD_BARS  = 6       # history bars stored in cache +1
OI_MIN_CHANGE_PCT        = -5.0
OI_MAX_CHANGE_PCT        = 10.0

# ── runtime / misc ────────────────────────────────────────────────────────
SCAN_INTERVAL_SEC = 60
TIMEFRAME = '5m' 
ENTRY_ORDER_TYPE  = "MARKET"        # or "MARKET"

# ─── Re‑entry cool‑down per symbol ──────────────────────────────────────────
SYMBOL_COOLDOWN_HOURS = 4    # skip new signals for the same symbol during X h

      
# ── market regime detection ──────────────────────────────────────────────────
# Parameters for the live market regime detector.

# The benchmark asset to determine the overall market state (e.g., BTCUSDT, ETHUSDT).
REGIME_BENCHMARK_SYMBOL = "BTCUSDT"

# How many minutes to cache the calculated regime before re-calculating.
# A value of 60 means it will run the calculation at most once per hour.
REGIME_CACHE_MINUTES = 60

# --- Parameters for the Trend Regime (TMA + Keltner Channel) ---
REGIME_MA_PERIOD = 100       # Long-term moving average period.
REGIME_ATR_PERIOD = 20       # ATR period for Keltner Channel width.
REGIME_ATR_MULT = 2.0        # ATR multiplier for the Keltner bands.


# --- ETH MACD BAROMETER FILTER (NEW) ---
ETH_BAROMETER_ENABLED = True
UNFAVORABLE_MODE = "RESIZE" 
UNFAVORABLE_RISK_RESIZE_FACTOR = 0.6 

# Debounce finalize of manual closes
FINALIZE_BACKOFF_SEC        = 120   # don't retry finalize more often than this

CANCEL_CLEANUP_BACKOFF_SEC  = 180




