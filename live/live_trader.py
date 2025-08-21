"""
live_trader.py – v4.2 (Clean, YAML sizing + meta gate, robust listing-age)
=============================================================================
- Robust UTC-safe listing-age helper (uses cache, then earliest OHLCV bar).
- Removes utcnow() deprecation warnings (timezone-aware everywhere).
- Fixes NameError/UnboundLocal (no stray 'sym', no undefined listing_age_days).
- Filters get the correct listing_age_days from the built signal.
- Keeps StrategyEngine, YAML scalers, meta-prob gate, DB/Telegram/Bybit V5, etc.
- ETH MACD barometer uses CLOSED 4h bar (no forming-bar repaint).
- Correlation guard is a proper method; safe scoping; optional soft sizing.
- TP1 double-fill bug removed; trailing activation idempotent.
- _activate_trailing defines tp_mult locally; no NameError.
- Finalize-zero helper calls finalize with the correct signature.
- Universe context persists rs_z and turnover_z for meta/use.
- Ticker price retrieval is robust (last/mark/index/close/bid/ask).
- Markov vol regime hardened with smoothed-prob reading + quantile fallback.
"""

from __future__ import annotations

import secrets
import asyncio
import dataclasses
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional
import subprocess
import math

import aiohttp
import asyncpg
import ccxt.async_support as ccxt
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config as cfg
from . import indicators as ta
from .indicators import vwap_stack_features
from . import filters
from .shared_utils import is_blacklisted
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from .exchange_proxy import ExchangeProxy
from .database import DB
from .telegram import TelegramBot
from .winprob_loader import WinProbScorer

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings
from .strategy_engine import StrategyEngine

import logging
logging.getLogger("watchdog").setLevel(logging.WARNING)
logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)

UNIVERSE_CACHE_PATH = Path("universe_cache.json")
# ──────────────────────────────────────────────────────────────────────────────
# Shims/utilities that may be referenced by joblib'd pipelines from research
# ──────────────────────────────────────────────────────────────────────────────

class ModelBundle:
    """Shim so joblib can unpickle research ModelBundle saved elsewhere."""
    pass

def _hour_cyc(X):
    """Cyclical hour encoder: returns [sin(hour*2π/24), cos(hour*2π/24)]."""
    h = np.asarray(X).astype(float).reshape(-1, 1)
    sin = np.sin(2 * np.pi * h / 24.0)
    cos = np.cos(2 * np.pi * h / 24.0)
    return np.hstack([sin, cos])

def _dow_onehot(X):
    """Day-of-week one-hot encoder: 7 columns, Mon=0..Sun=6."""
    d = np.asarray(X).astype(int).reshape(-1, 1)
    out = np.zeros((d.shape[0], 7), dtype=float)
    m = (d >= 0) & (d < 7)
    out[np.arange(d.shape[0])[m.ravel()], d[m].ravel()] = 1.0
    return out

def triangular_moving_average(series: pd.Series, period: int) -> pd.Series:
    """Simplified TMA (double SMA)."""
    return series.rolling(window=period, min_periods=period).mean().rolling(window=period, min_periods=period).mean()


LISTING_PATH = Path("listing_dates.json")

###############################################################################
# 0 ▸ LOGGING #################################################################
###############################################################################

LOG = logging.getLogger("live_trader")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("ccxt").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

###############################################################################
# 1 ▸ SETTINGS pulled from ENV (.env) #########################################
###############################################################################

class Settings(BaseSettings):
    """Secrets & env flags."""
    bybit_api_key: str = Field(..., validation_alias="BYBIT_API_KEY")
    bybit_api_secret: str = Field(..., validation_alias="BYBIT_API_SECRET")
    bybit_testnet: bool = Field(False, validation_alias="BYBIT_TESTNET")
    tg_bot_token: str = Field(..., validation_alias="TG_BOT_TOKEN")
    tg_chat_id: str = Field(..., validation_alias="TG_CHAT_ID")
    pg_dsn: str = Field(..., validation_alias="DATABASE_URL")
    default_leverage: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = 'ignore'

###############################################################################
# 2 ▸ PATHS & YAML LOADER #####################################################
###############################################################################

CONFIG_PATH = Path("config.yaml")
SYMBOLS_PATH = Path("symbols.txt")

def load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(p)
    return yaml.safe_load(p.read_text()) or {}

###############################################################################
# 3 ▸ HOT-RELOAD WATCHER ######################################################
###############################################################################

class _Watcher(FileSystemEventHandler):
    def __init__(self, path: Path, cb):
        self.path = path.resolve()
        self.cb = cb
        obs = Observer()
        obs.schedule(self, self.path.parent.as_posix(), recursive=False)
        obs.daemon = True
        obs.start()

    def on_modified(self, e):
        if Path(e.src_path).resolve() == self.path:
            self.cb()

###############################################################################
# 4 ▸ RISK MANAGER ############################################################
###############################################################################

class RiskManager:
    def __init__(self, cfg_dict: Dict[str, Any]):
        self.cfg = cfg_dict
        self.loss_streak = 0
        self.kill_switch = False

    async def on_trade_close(self, pnl: float, telegram: TelegramBot):
        if pnl < 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0
        if self.loss_streak >= self.cfg.get("MAX_LOSS_STREAK", 3):
            self.kill_switch = True
            await telegram.send("❌ Kill-switch: max loss streak")

    def can_trade(self) -> bool:
        return not self.kill_switch

###############################################################################
# 5 ▸ REGIME DETECTOR #########################################################
###############################################################################

class RegimeDetector:
    """
    Calculates and caches the market regime based on a live benchmark asset.
    """
    def __init__(self, exchange, config: dict):
        self.exchange = exchange
        self.cfg = config
        self.benchmark_symbol = self.cfg.get("REGIME_BENCHMARK_SYMBOL", "BTCUSDT")
        self.cache_duration = timedelta(minutes=self.cfg.get("REGIME_CACHE_MINUTES", 60))
        self.cached_regime = "UNKNOWN"
        self.last_calculation_time = None
        LOG.info(
            "RegimeDetector initialized for benchmark %s with a %d-minute cache.",
            self.benchmark_symbol, self.cfg.get("REGIME_CACHE_MINUTES", 60)
        )

    async def _fetch_benchmark_data(self) -> pd.DataFrame | None:
        """Fetches the last ~500 days of daily OHLCV data for the benchmark symbol."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(self.benchmark_symbol, '1d', limit=500)
            if len(ohlcv) < 200:
                LOG.warning("Not enough historical data for %s to calculate regime (%d bars).",
                            self.benchmark_symbol, len(ohlcv))
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            LOG.error("Failed to fetch benchmark data for regime detection: %s", e)
            return None

    def _calculate_vol_regime(self, daily_returns: pd.Series) -> pd.Series:
        """Volatility regime via Markov Switching (2 states) with robust fallback."""
        try:
            dr = daily_returns.dropna()
            if len(dr) < 60:
                raise RuntimeError("not enough returns for Markov")
            model = sm.tsa.MarkovRegression(dr, k_regimes=2, switching_variance=True)
            results = model.fit(disp=False)
            probs = results.smoothed_marginal_probabilities  # DataFrame-like, index = regime id
            # Assign regime with highest probability each time step
            assign = probs.idxmax(axis=0)
            tmp = pd.DataFrame({"ret": dr, "reg": assign})
            var0 = tmp.loc[tmp["reg"] == 0, "ret"].var()
            var1 = tmp.loc[tmp["reg"] == 1, "ret"].var()
            low_vol = 0 if (var0 < var1) else 1
            lab = np.where(probs.loc[low_vol] > 0.5, "LOW_VOL", "HIGH_VOL")
            return pd.Series(lab, index=dr.index, name="vol_regime")
        except Exception as e:
            LOG.warning("Markov vol regime failed: %s. Defaulting to realized-vol quantile.", e)
            rv = daily_returns.rolling(20).std()
            q = rv.quantile(0.5)
            return pd.Series(np.where(rv <= q, "LOW_VOL", "HIGH_VOL"),
                             index=daily_returns.index, name="vol_regime")

    def _calculate_trend_regime(self, df_daily: pd.DataFrame) -> pd.Series:
        """Trend regime via TMA ± Keltner (ATR×mult)."""
        df_daily['tma'] = triangular_moving_average(df_daily['close'], self.cfg.get("REGIME_MA_PERIOD", 100))
        atr_series = ta.atr(df_daily, period=self.cfg.get("REGIME_ATR_PERIOD", 20))
        df_daily['keltner_upper'] = df_daily['tma'] + (atr_series * self.cfg.get("REGIME_ATR_MULT", 2.0))
        df_daily['keltner_lower'] = df_daily['tma'] - (atr_series * self.cfg.get("REGIME_ATR_MULT", 2.0))
        df_daily.dropna(inplace=True)

        trend = pd.Series(np.nan, index=df_daily.index, dtype="object")
        for i in range(1, len(df_daily)):
            if df_daily['close'].iloc[i] > df_daily['keltner_upper'].iloc[i]:
                trend.iloc[i] = "BULL"
            elif df_daily['close'].iloc[i] < df_daily['keltner_lower'].iloc[i]:
                trend.iloc[i] = "BEAR"
            else:
                trend.iloc[i] = trend.iloc[i-1]
        return trend.ffill().bfill()

    async def get_current_regime(self) -> str:
        now = datetime.now(timezone.utc)
        if self.last_calculation_time and (now - self.last_calculation_time) < self.cache_duration:
            return self.cached_regime

        LOG.info("Regime cache expired/empty. Recomputing…")
        self.last_calculation_time = now

        df_daily = await self._fetch_benchmark_data()
        if df_daily is None:
            self.cached_regime = "UNKNOWN"
            return self.cached_regime

        daily_returns = df_daily['close'].pct_change()
        df_daily['vol_regime'] = self._calculate_vol_regime(daily_returns)
        df_daily['trend_regime'] = self._calculate_trend_regime(df_daily)
        df_daily.dropna(subset=['vol_regime','trend_regime'], inplace=True)
        if df_daily.empty:
            self.cached_regime = "UNKNOWN"
            return self.cached_regime

        self.cached_regime = f"{df_daily['trend_regime'].iloc[-1]}_{df_daily['vol_regime'].iloc[-1]}"
        LOG.info("New market regime: %s", self.cached_regime)
        return self.cached_regime

###############################################################################
# 6 ▸ SIGNAL DATACLASS ########################################################
###############################################################################

@dataclasses.dataclass
class Signal:
    # Required
    symbol: str
    entry: float
    atr: float
    rsi: float
    adx: float
    atr_pct: float
    market_regime: str
    price_boom_pct: float
    price_slowdown_pct: float
    vwap_dev_pct: float
    ret_30d: float
    ema_fast: float
    ema_slow: float
    listing_age_days: int
    vwap_z_score: float
    is_ema_crossed_down: bool
    vwap_consolidated: bool
    session_tag: str
    day_of_week: int
    hour_of_day: int
    side: str = "short"
    # Optional
    win_probability: float = 0.0

    def __post_init__(self):
        self.side = str(self.side).upper()
        if self.side not in ("LONG", "SHORT"):
            self.side = "SHORT"
        LOG.info(f"Successfully created Signal object for {self.symbol} using correct class definition.")

###############################################################################
# 7 ▸ MAIN TRADER #############################################################
###############################################################################

class LiveTrader:
    def __init__(self, settings: Settings, cfg_dict: Dict[str, Any]):
        self.settings = settings

        # Merge python config defaults (config.py) with YAML overrides into self.cfg
        self.cfg: Dict[str, Any] = {}
        for key in dir(cfg):
            if key.isupper():
                self.cfg[key] = getattr(cfg, key)
        self.cfg.update(cfg_dict)
        for k, v in self.cfg.items():  # keep config.py attributes in sync too
            setattr(cfg, k, v)

        self.db = DB(settings.pg_dsn)
        self.tg = TelegramBot(settings.tg_bot_token, settings.tg_chat_id)
        self.risk = RiskManager(self.cfg)

        # Exchange & regime detector
        self.exchange = ExchangeProxy(self._init_ccxt())
        self.regime_detector = RegimeDetector(self.exchange, self.cfg)

        self.symbols = self._load_symbols()
        self._universe_ctx: dict[str, dict] = {}
        self._universe_ctx_ts: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self.open_positions: Dict[int, Dict[str, Any]] = {}
        self.peak_equity: float = 0.0
        self._listing_dates_cache: Dict[str, datetime.date] = {}
        self.last_exit: Dict[str, datetime] = {}
        self._zero_finalize_backoff: Dict[int, datetime] = {}

        self._zero_finalize_backoff: Dict[int, datetime] = {}
        self._cancel_cleanup_backoff: Dict[str, datetime] = {}

        self.strategy_engine = StrategyEngine(
            self.cfg.get("STRATEGY_SPEC_PATH", "strategy/donch_pullback_long.yaml"),
            cfg=self.cfg
        )
        self._strategy_spec_path = Path(self.strategy_engine.spec_path).resolve()
        _Watcher(self._strategy_spec_path, self._reload_strategy)
        LOG.info("StrategyEngine loaded from %s; requires TFs: %s",
                 self._strategy_spec_path, sorted(self.strategy_engine.required_timeframes()))

        # Hot reloads
        _Watcher(CONFIG_PATH, self._reload_cfg)
        _Watcher(SYMBOLS_PATH, self._reload_symbols)

        self.paused = False
        self.tasks: List[asyncio.Task] = []
        self.api_semaphore = asyncio.Semaphore(10)
        self.symbol_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Meta / win-prob scorer
        from live.winprob_loader import WinProbScorer
        self.winprob = WinProbScorer(os.getenv("DONCH_WINPROB_DIR", "results/meta_export"))

        art_dir = self.cfg.get("WINPROB_ARTIFACT_DIR", "results/meta_export")
        self.winprob = WinProbScorer(art_dir)

        try:
            self.winprob.strict_parity = bool(self.cfg.get("META_REQUIRED", False))
        except Exception:
            pass

        if self.winprob.is_loaded:
            # Be robust to older WinProbScorer without .kind or .expected_features
            kind = getattr(self.winprob, "kind", "lgbm+ohe")
            feats = (getattr(self.winprob, "feature_order", None)
                    or getattr(self.winprob, "expected_features", None)
                    or [])
            nfeats = len(feats) if feats is not None else 0
            pstar = getattr(self.winprob, "pstar", None)
            if isinstance(pstar, (int, float)):
                LOG.info("WinProb ready (kind=%s, features=%d, p*=%.2f)", kind, nfeats, float(pstar))
            else:
                LOG.info("WinProb ready (kind=%s, features=%d)", kind, nfeats)
        else:
            LOG.warning("WinProb not loaded; using wp=0.0")

    def _score_winprob_safe(self, symbol: str, meta_row: dict) -> Optional[float]:
        wp = getattr(self, "winprob", None)
        if not wp or not getattr(wp, "is_loaded", False):
            return None
        try:
            p = float(wp.score(meta_row))
            return p if np.isfinite(p) and 0.0 <= p <= 1.0 else None
        except Exception as e:
            LOG.debug("WinProb error for %s: %s", symbol, e)
            return None

    def _load_universe_cache_if_fresh(self) -> Optional[dict]:
        """Return {symbol: {rs_pct, median_24h_turnover_usd}} if cache exists & fresh & symbols match."""
        if not bool(self.cfg.get("UNIVERSE_CACHE_ENABLED", True)):
            return None
        p = Path(self.cfg.get("UNIVERSE_CACHE_PATH", UNIVERSE_CACHE_PATH))
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text())
            ttl_min = int(self.cfg.get("UNIVERSE_CACHE_TTL_MIN", 1440))  # default: 24h
            ts = raw.get("computed_at")
            if not ts:
                return None
            # tolerate "Z" suffix or naive
            try:
                computed_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                computed_at = datetime.fromisoformat(ts)
            if computed_at.tzinfo is None:
                computed_at = computed_at.replace(tzinfo=timezone.utc)

            if datetime.now(timezone.utc) - computed_at > timedelta(minutes=ttl_min):
                return None

            if bool(self.cfg.get("UNIVERSE_CACHE_REQUIRE_SYMBOLS_MATCH", True)):
                cached_syms = list(map(str, raw.get("symbols", [])))
                if sorted(cached_syms) != sorted(map(str, self.symbols)):
                    return None

            data = raw.get("data")
            return data if isinstance(data, dict) and data else None
        except Exception as e:
            LOG.warning("Universe cache load failed: %s", e)
            return None

    def _save_universe_cache(self, data: dict) -> None:
        if not bool(self.cfg.get("UNIVERSE_CACHE_ENABLED", True)):
            return
        p = Path(self.cfg.get("UNIVERSE_CACHE_PATH", UNIVERSE_CACHE_PATH))
        try:
            blob = {
                "computed_at": datetime.now(timezone.utc).isoformat(),
                "symbols": list(self.symbols),
                "data": data,
            }
            p.write_text(json.dumps(blob))
            LOG.info("Universe context cached to %s (%d symbols).", p, len(data))
        except Exception as e:
            LOG.warning("Failed to write universe cache: %s", e)

    async def _build_universe_context(self) -> dict[str, dict]:
        """
        Compute:
        - weekly RS percentile (based on 1d closes: last / close[-7d])
        - median 24h USD turnover (approx from 1h bars; median of rolling 24h sums over last 7d)
        Returns { symbol: {"rs_pct": float, "median_24h_turnover_usd": float, "rs_z": float, "turnover_z": float} }
        """
        syms = list(self.symbols)
        LOG.info("Building universe context for %d symbols…", len(syms))

        # Run the two big batches without holding a single semaphore token over all of them.
        tasks_1d = {s: self.exchange.fetch_ohlcv(s, '1d', limit=10) for s in syms}
        tasks_1h = {s: self.exchange.fetch_ohlcv(s, '1h', limit=168) for s in syms}
        res_1d = await asyncio.gather(*tasks_1d.values(), return_exceptions=True)
        res_1h = await asyncio.gather(*tasks_1h.values(), return_exceptions=True)

        one_d = {s: r for s, r in zip(tasks_1d.keys(), res_1d)}
        one_h = {s: r for s, r in zip(tasks_1h.keys(), res_1h)}

        weekly_ret = {}
        med24 = {}
        for s in syms:
            # weekly return from 1d
            d = one_d.get(s)
            rs = 0.0
            try:
                if isinstance(d, list) and len(d) >= 8:
                    close_series = pd.Series([row[4] for row in d], index=pd.to_datetime([row[0] for row in d], unit='ms', utc=True))
                    last = float(close_series.iloc[-1])
                    # 7 days ago close (approx; daily granularity)
                    prev = float(close_series.iloc[-8])  # T-7d
                    rs = (last / prev - 1.0) if prev > 0 else 0.0
            except Exception:
                rs = 0.0
            weekly_ret[s] = rs

            # median 24h turnover from 1h bars
            h = one_h.get(s)
            med = 0.0
            try:
                if isinstance(h, list) and len(h) >= 48:
                    dfh = pd.DataFrame(h, columns=['ts','o','h','l','c','v'])
                    dfh['ts'] = pd.to_datetime(dfh['ts'], unit='ms', utc=True)
                    dfh.set_index('ts', inplace=True)
                    # USD notional per hour ~ close * volume
                    notional = dfh['c'] * dfh['v']
                    # rolling 24h sum (24 bars)
                    roll_24h = notional.rolling(24).sum().dropna()
                    # median over last 7 windows if available
                    med = float(roll_24h.tail(7).median()) if len(roll_24h) >= 1 else 0.0
            except Exception:
                med = 0.0
            med24[s] = med

        # percentile ranks
        rets = np.array([weekly_ret[s] for s in syms], dtype=float)
        ranks = np.argsort(np.argsort(rets))  # 0..N-1
        pct = 100.0 * ranks / max(1, len(syms) - 1)

        rs_arr = np.asarray(rets, dtype=float)
        to_arr = np.asarray([med24[s] for s in syms], dtype=float)
        mu_rs = float(np.nanmean(rs_arr)); sd_rs = float(np.nanstd(rs_arr));  sd_rs = sd_rs if sd_rs > 0 else 1.0
        mu_to = float(np.nanmean(to_arr)); sd_to = float(np.nanstd(to_arr));  sd_to = sd_to if sd_to > 0 else 1.0
        rs_z  = (rs_arr - mu_rs) / sd_rs
        to_z  = (to_arr - mu_to) / sd_to

        out = {}
        for i, s in enumerate(syms):
            out[s] = {
                "rs_pct": float(pct[i]),
                "median_24h_turnover_usd": float(med24[s]),
                "rs_z": float(rs_z[i]),
                "turnover_z": float(to_z[i]),
            }

        LOG.info("Universe context ready (%d symbols).", len(out))
        return out

    # ---- Prior Donchian breakout quality (fail count & rate) ---------------
    def _prior_breakout_stats(
        self,
        df1d: pd.DataFrame,
        period: int = 20,
        lookback_days: int = 60,
        fallback_days: int = 3,
    ) -> tuple[int, int, float]:
        """
        Count breakouts in last `lookback_days` where close[t] crossed above the
        prior-N-day Donch upper (shifted) and within `fallback_days` the close
        fell back below that upper (failed breakout).
        Returns: (num_breakouts, num_failures, fail_rate)
        """
        try:
            if df1d is None or len(df1d) < period + fallback_days + 5:
                return 0, 0, 0.0
            dd = df1d.sort_index()
            highs = dd["high"].rolling(period).max().shift(1)
            close = dd["close"].astype(float)
            # limit to lookback window (exclude last 1 bar to avoid lookahead)
            sub = dd.iloc[-(lookback_days + fallback_days + 1):-1]
            if sub.empty: return 0, 0, 0.0

            highs_sub = highs.loc[sub.index]
            cls_sub   = close.loc[sub.index]

            brk_idx = cls_sub > highs_sub
            breakout_days = list(np.where(brk_idx.to_numpy())[0])
            n_b = 0; n_f = 0
            for idx in breakout_days:
                n_b += 1
                upper = float(highs_sub.iloc[idx])
                # window of next k daily closes (guard tail)
                nxt = cls_sub.iloc[idx+1: idx+1+fallback_days]
                if len(nxt) and np.nanmin(nxt.to_numpy()) < upper:
                    n_f += 1
            fail_rate = float(n_f / n_b) if n_b > 0 else 0.0
            return int(n_b), int(n_f), float(fail_rate)
        except Exception:
            return 0, 0, 0.0

    async def _cancel_reducing_orders(self, symbol: str, pos: dict):
        """
        Cancel any leftover reduce-only/close orders (TP/SL etc) for this symbol.
        Uses both order id and clientOrderId when available.
        """
        try:
            open_orders = await self._all_open_orders(symbol) or []
        except Exception as e:
            LOG.warning("Cancel sweep: failed to fetch open orders for %s: %s", symbol, e)
            return

        # Known client ids we created for this position
        known_cids = {
            pos.get("tp_final_cid"), pos.get("tp2_cid"), pos.get("tp1_cid"),
            pos.get("sl_trail_cid"), pos.get("sl_cid")
        }

        for o in open_orders:
            try:
                oid  = o.get("id") or o.get("orderId")
                coid = o.get("clientOrderId") or o.get("client_order_id")
                # Reduce-only flags vary by venue/ccxt version
                ro   = bool(o.get("reduceOnly") or o.get("reduce_only") or
                            (o.get("info", {}).get("reduceOnly") if isinstance(o.get("info"), dict) else False))
                # Heuristic: any order we created with a known CID is a child; cancel it.
                is_child = (coid in known_cids)
                if ro or is_child:
                    try:
                        if oid:
                            await self.exchange.cancel_order(oid, symbol, {"clientOrderId": coid, "acknowledged": True})
                        elif coid:
                            # Some venues allow cancel by client id only
                            await self.exchange.cancel_order(None, symbol, {"clientOrderId": coid, "acknowledged": True})
                        LOG.info("Canceled stale child order %s (cid=%s) on %s", oid, coid, symbol)
                    except Exception as ce:
                        LOG.warning("Cancel failed for %s (cid=%s) on %s: %s", oid, coid, symbol, ce)
            except Exception as e:
                LOG.warning("Cancel sweep loop error on %s: %s", symbol, e)

    def _listing_age_days(self, symbol: str, dfs: dict[str, pd.DataFrame] | None = None) -> int | None:
        """
        Order of preference:
          1) cached listing_dates.json
          2) earliest timestamp in fetched dataframes (prefer '1d' if present)
          3) None if unknown
        """
        today = datetime.now(timezone.utc).date()

        # 1) cache
        d = self._listing_dates_cache.get(symbol)
        if d:
            try:
                return (today - d).days
            except Exception:
                pass

        # 2) infer from dataframes (earliest bar)
        if dfs:
            try:
                # prefer '1d', otherwise earliest among all TFs
                if '1d' in dfs and not dfs['1d'].empty:
                    earliest = dfs['1d'].index[0].date()
                else:
                    earliest = None
                    for df in dfs.values():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            ts = df.index[0].date()
                            if earliest is None or ts < earliest:
                                earliest = ts
                if earliest:
                    return (today - earliest).days
            except Exception:
                pass

        # 3) unknown
        return None

    def _reload_strategy(self):
        try:
            self.strategy_engine.reload()
            LOG.info("Strategy reloaded from %s", self.strategy_engine.spec_path)
            try:
                asyncio.create_task(self.tg.send(f"♻️ Strategy reloaded: {self.strategy_engine.spec_path}"))
            except Exception:
                pass
        except Exception as e:
            LOG.error("Strategy reload failed: %s", e)

    def _init_ccxt(self):
        ex = ccxt.bybit({
            "apiKey": self.settings.bybit_api_key,
            "secret": self.settings.bybit_api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        if self.settings.bybit_testnet:
            ex.set_sandbox_mode(True)
        return ex

    @staticmethod
    def _load_symbols():
        return SYMBOLS_PATH.read_text().split()

    def _reload_cfg(self):
        try:
            new_cfg = load_yaml(CONFIG_PATH)
            self.cfg.update(new_cfg)
            for k, v in self.cfg.items():
                setattr(cfg, k, v)
            LOG.info("Config reloaded from %s", CONFIG_PATH)
        except Exception as e:
            LOG.error("Failed to reload config: %s", e)

    def _reload_symbols(self):
        try:
            self.symbols = self._load_symbols()
            LOG.info("Symbols reloaded – %d symbols", len(self.symbols))
        except Exception as e:
            LOG.error("Failed to reload symbols: %s", e)

    def _normalize_cfg_key(self, key: str) -> str:
        """Alias old → new config keys so /set works with either."""
        key_up = key.upper()
        aliases = {
            "FIXED_RISK_USDT": "RISK_USD",
            "RISK_USDT": "RISK_USD",
            "RISK_PCT": "RISK_EQUITY_PCT",
            "WINPROB_SIZE_FLOOR": "WINPROB_PROB_FLOOR",
            "WINPROB_SIZE_CAP":   "WINPROB_PROB_CAP",
        }
        return aliases.get(key_up, key_up)

    @staticmethod
    def _cid(pid: int, tag: str) -> str:
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        return f"bot_{pid}_{tag}_{timestamp_ms}"[:36]

    async def _ensure_leverage(self, symbol: str):
        try:
            await self.exchange.set_margin_mode("cross", symbol)
            await self.exchange.set_leverage(self.settings.default_leverage, symbol)
        except ccxt.ExchangeError as e:
            if "leverage not modified" in str(e):
                LOG.debug("Leverage for %s already set to %dx.", symbol, self.settings.default_leverage)
            else:
                LOG.warning("Leverage setup failed for %s: %s", symbol, e)
                raise e
        except Exception as e:
            LOG.warning("Unexpected leverage setup error for %s: %s", symbol, e)
            raise e

    # ───────────────────── Correlation guard (as a method) ────────────────────

    async def _corr_guard_stats(self, target_symbol: str, *, timeframe: str = "1h",
                                lookback: int = 500, method: str = "spearman") -> dict:
        """
        Corr stats of target vs currently open symbols using log-return Spearman.
        Returns: {'avg': float, 'max': float, 'n': int, 'by_symbol': {sym: rho}}
        """
        try:
            others = [row["symbol"] for row in self.open_positions.values()
                      if isinstance(row, dict) and row.get("symbol") and row["symbol"] != target_symbol]
            if not others:
                return {"avg": 0.0, "max": 0.0, "n": 0, "by_symbol": {}}

            async def _closes(sym):
                o = await self.exchange.fetch_ohlcv(sym, timeframe, limit=lookback)
                if not o: return None
                df = pd.DataFrame(o, columns=["ts","o","h","l","c","v"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                df.set_index("ts", inplace=True)
                if len(df) < 3:
                    return None
                df = df.iloc[:-1]  # drop forming bar
                return pd.Series(df["c"].astype(float), index=df.index)

            tgt = await _closes(target_symbol)
            if tgt is None or len(tgt) < 50:
                return {"avg": 0.0, "max": 0.0, "n": 0, "by_symbol": {}}

            series = {"__TARGET__": np.log(tgt).diff().dropna()}
            for s in others:
                cs = await _closes(s)
                if cs is not None:
                    series[s] = np.log(cs).diff().dropna()

            df = pd.concat(series.values(), axis=1, join="inner")
            df.columns = list(series.keys())
            if df.shape[0] < 50 or df.shape[1] < 2:
                return {"avg": 0.0, "max": 0.0, "n": 0, "by_symbol": {}}

            corr = df.corr(method=method)
            by = corr.loc["__TARGET__"].drop("__TARGET__", errors="ignore").dropna()
            if by.empty:
                return {"avg": 0.0, "max": 0.0, "n": 0, "by_symbol": {}}
            return {"avg": float(by.abs().mean()), "max": float(by.abs().max()), "n": int(by.size),
                    "by_symbol": {k: float(v) for k, v in by.items()}}
        except Exception as e:
            LOG.warning("corr_guard failed for %s: %s", target_symbol, e)
            return {"avg": 0.0, "max": 0.0, "n": 0, "by_symbol": {}}

    # ───────────────────── Indicator scanning ─────────────────────

    async def _scan_symbol_for_signal(
        self,
        symbol: str,
        market_regime: str,
        eth_macd: Optional[dict],
        gov_ctx: Optional[dict] = None  # placeholder
    ) -> Optional[Signal]:
        LOG.info("Checking %s...", symbol)
        try:
            # ---------------- Timeframes to fetch ----------------
            base_tf = str(self.cfg.get('TIMEFRAME', '5m'))
            ema_tf  = str(self.cfg.get('EMA_TIMEFRAME', '4h'))
            rsi_tf  = str(self.cfg.get('RSI_TIMEFRAME', '1h'))
            adx_tf  = str(self.cfg.get('ADX_TIMEFRAME', '1h'))

            required_tfs = {base_tf, ema_tf, rsi_tf, adx_tf, '1d', '1h'}
            required_tfs |= set(self.strategy_engine.required_timeframes() or [])

            # ---------------- OHLCV fetch ------------------------
            async with self.api_semaphore:
                tasks = {
                    tf: self.exchange.fetch_ohlcv(symbol, tf, limit=1500 if tf == base_tf else 500)
                    for tf in sorted(required_tfs)
                }
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                ohlcv_data = dict(zip(tasks.keys(), results))

            dfs: dict[str, pd.DataFrame] = {}
            for tf, data in ohlcv_data.items():
                if isinstance(data, Exception) or not data:
                    LOG.debug("Could not fetch OHLCV for %s on %s.", symbol, tf)
                    return None
                df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)

                # Stale/illiquid guard on last 100 completed bars
                recent = df.tail(100)
                if not recent.empty:
                    zero_vol_pct = (recent['volume'] == 0).sum() / len(recent)
                    if zero_vol_pct > 0.25:
                        LOG.warning("DATA_ERROR %s %s: stale (%.0f%% zero vol).", symbol, tf, zero_vol_pct * 100)
                        return None

                # Drop possibly incomplete last bar; ensure non-empty
                if len(df) < 3:
                    return None
                df = df.iloc[:-1]
                if df.empty:
                    return None

                dfs[tf] = df

            if base_tf not in dfs:
                LOG.warning("Base TF %s not available for %s.", base_tf, symbol)
                return None
            df5 = dfs[base_tf]

            # ---------------- Indicators aligned to base index ---
            # EMA on ema_tf (for optional heuristics)
            df5['ema_fast'] = ta.ema(dfs[ema_tf]['close'], cfg.EMA_FAST_PERIOD).reindex(df5.index, method='ffill')
            df5['ema_slow'] = ta.ema(dfs[ema_tf]['close'], cfg.EMA_SLOW_PERIOD).reindex(df5.index, method='ffill')

            # Training features are 1h ATR/RSI/ADX
            atr_len = int(self.cfg.get("ATR_LEN", 14))
            df1h = dfs.get('1h')
            if df1h is None or df1h.empty:
                LOG.warning("No 1h data for %s; cannot compute ATR/RSI/ADX features.", symbol)
                return None
            atr_1h = ta.atr(df1h, atr_len)
            rsi_1h = ta.rsi(df1h['close'], atr_len)
            adx_1h = ta.adx(df1h, atr_len)

            df5['atr_1h'] = atr_1h.reindex(df5.index, method='ffill')
            df5['rsi_1h'] = rsi_1h.reindex(df5.index, method='ffill')
            df5['adx_1h'] = adx_1h.reindex(df5.index, method='ffill')

            # Legacy aliases used downstream
            df5['atr'] = df5['atr_1h']
            df5['adx'] = df5['adx_1h']
            df5['rsi'] = df5['rsi_1h']

            # For the meta row & diag we'll need a clean "last" of required cols
            needed = ['close','volume','atr_1h','rsi_1h','adx_1h','ema_fast','ema_slow']
            df5_req = df5[needed].dropna()
            if df5_req.empty:
                return None
            last = df5_req.iloc[-1]

            # ---------------- Listing age ------------------------
            age_opt = self._listing_age_days(symbol, dfs)
            listing_age_days = int(age_opt) if age_opt is not None else 9999

            # ---------------- Strategy verdict (rule engine) -----
            univ = (getattr(self, "_universe_ctx", {}) or {}).get(symbol, {})  # e.g., {"rs_pct": 78, "liq_ok": True}
            eth_ctx = {"eth_macd": dict(eth_macd or {})}
            ctx = {
                "symbol": symbol,
                "market_regime": market_regime,
                "listing_age_days": listing_age_days,
                "last_exit_dt": self.last_exit.get(symbol),
                "is_symbol_blacklisted": is_blacklisted(symbol),
                **univ, **eth_ctx,
            }
            verdict = self.strategy_engine.evaluate(dfs, ctx)
            should_enter = bool(getattr(verdict, "should_enter", False))

            # Resolve side EARLY so heuristics below can use it safely
            side = self._resolve_side(verdict)  # "long" | "short"

            # ---------------- Light features used by meta/diag ----
            # ret_30d (daily trend proxy)
            df1d = dfs.get('1d')
            ret_30d = 0.0
            if df1d is not None and len(df1d) > cfg.STRUCTURAL_TREND_DAYS:
                ret_30d = (df1d['close'].iloc[-1] / df1d['close'].iloc[-cfg.STRUCTURAL_TREND_DAYS] - 1)

            tf_minutes = 5  # base timeframe minutes (used for vol_mult window)
            # Donch breakout info
            don_len = int(
                getattr(verdict, "don_break_len",
                        ((getattr(self.strategy_engine, "_spec", {}) or {}).get("params", {}) or {})
                        .get("DONCH_PERIOD", self.cfg.get("DON_N_DAYS", 20)))
            )

            don_level = None
            try:
                if df1d is not None and len(df1d) >= (don_len + 2):
                    don_upper = df1d['high'].rolling(don_len).max().shift(1)  # prior N full days
                    don_level = float(don_upper.iloc[-1])
            except Exception:
                pass
            if getattr(verdict, "don_break_level", None) is not None:
                don_level = float(verdict.don_break_level)
            if don_level is None:
                don_level = float(last['close'])

            # volume multiple: current 5m volume / rolling ~30d median
            try:
                bars_needed = int(self.cfg.get("VOL_LOOKBACK_DAYS", 30) * 24 * (60 // tf_minutes))
                bars_needed = max(96, min(bars_needed, len(df5)))
                vol_med = df5['volume'].tail(bars_needed).median()
                vol_mult = float(last['volume'] / vol_med) if vol_med > 0 else 0.0
            except Exception:
                vol_mult = 0.0

            rs_pct = float(univ.get("rs_pct", 0.0) or 0.0)

            # ETH MACD hist & above-signal
            eth_hist = float((eth_macd or {}).get("hist", 0.0) or 0.0)
            eth_above = 1.0 if float((eth_macd or {}).get("macd", 0.0)) > float((eth_macd or {}).get("signal", 0.0)) else 0.0
            regime_up = 1.0 if (eth_hist > 0 and eth_above == 1.0) else 0.0

            # time cyclicals
            now_utc = datetime.now(timezone.utc)
            hour_of_day = now_utc.hour
            day_of_week = now_utc.weekday()
            hour_sin = math.sin(2 * math.pi * hour_of_day / 24.0)
            hour_cos = math.cos(2 * math.pi * hour_of_day / 24.0)

            # derived distances
            atr_pct = (last['atr_1h'] / last['close']) if last['close'] > 0 else 0.0
            don_dist_atr = (float(last['close']) - float(don_level)) / float(last['atr_1h'] if last['atr_1h'] > 0 else np.nan)
            if not np.isfinite(don_dist_atr):
                don_dist_atr = 0.0

            # -------------- Meta-model feature row (canonical) ----
            entry_rule = getattr(verdict, "entry_rule", None) or self.cfg.get("ENTRY_RULE", "close_above_break")
            pullback_type = getattr(verdict, "pullback_type", None) or self.cfg.get("PULLBACK_TYPE", "retest")
            regime_1d = getattr(verdict, "regime_1d", None) or ""  # optional

            # --- NEW: AVWAP (rolling VWAP) stack features on base tf ----------
            try:
                vwap_lb   = int(self.cfg.get("VWAP_LOOKBACK_BARS", 12))
                vwap_band = float(self.cfg.get("VWAP_BAND_PCT", 0.004))
                vwap_feats = vwap_stack_features(df5, lookback_bars=vwap_lb, band_pct=vwap_band)
            except Exception:
                vwap_feats = {"vwap_frac_in_band": 0.0, "vwap_expansion_pct": 0.0, "vwap_slope_pph": 0.0}

            # --- NEW: perp basis (mark-index) and funding rate ----------------
            basis_pct = 0.0
            funding_8h = 0.0
            try:
                tk = await self.exchange.fetch_ticker(symbol)
                mark  = float(tk.get("markPrice", tk.get("mark", tk.get("last", 0.0)))) if tk else 0.0
                index = float(tk.get("indexPrice", tk.get("index", 0.0))) if tk else 0.0
                if isinstance(tk, dict) and "info" in tk:
                    info = tk["info"] or {}
                    mark  = float(info.get("markPrice", mark))
                    index = float(info.get("indexPrice", index))
                if index and np.isfinite(mark) and np.isfinite(index) and index > 0:
                    basis_pct = float((mark - index) / index)
            except Exception:
                pass
            try:
                fr = await self.exchange.fetch_funding_rate(symbol)
                funding_8h = float(fr.get("fundingRate", 0.0)) if isinstance(fr, dict) else 0.0
                if not np.isfinite(funding_8h):
                    funding_8h = 0.0
            except Exception:
                pass

            # --- NEW: prior breakout quality from 1d --------------------------
            prior_b, prior_f, prior_fail_rate = (0, 0, 0.0)
            try:
                prior_b, prior_f, prior_fail_rate = self._prior_breakout_stats(
                    df1d, period=don_len, lookback_days=int(self.cfg.get("PRIOR_BRK_LOOKBACK_DAYS", 60)),
                    fallback_days=int(self.cfg.get("PRIOR_BRK_FALLBACK_DAYS", 3))
                )
            except Exception:
                pass

            # --- NEW: pull RS/turnover z-scores from universe snapshot --------
            rs_z = float(univ.get("rs_z", 0.0) or 0.0)
            turnover_z = float(univ.get("turnover_z", 0.0) or 0.0)

            meta_row = {
                # numerics
                "atr_1h": float(last['atr_1h']),
                "rsi_1h": float(last['rsi_1h']),
                "adx_1h": float(last['adx_1h']),
                "atr_pct": float(atr_pct),
                "don_break_len": float(don_len),
                "don_break_level": float(don_level),
                "don_dist_atr": float(don_dist_atr),
                "rs_pct": float(rs_pct),
                "hour_sin": float(hour_sin),
                "hour_cos": float(hour_cos),
                "dow": float(day_of_week),
                "vol_mult": float(vol_mult),
                "eth_macd_hist_4h": float(eth_hist),
                "regime_up": float(regime_up),
                "prior_1d_ret": float(ret_30d),

                # --- NEW: AVWAP stack
                "vwap_frac_in_band": float(vwap_feats.get("vwap_frac_in_band", 0.0)),
                "vwap_expansion_pct": float(vwap_feats.get("vwap_expansion_pct", 0.0)),
                "vwap_slope_pph": float(vwap_feats.get("vwap_slope_pph", 0.0)),

                # --- NEW: basis / funding
                "basis_pct": float(basis_pct),
                "funding_8h": float(funding_8h),

                # --- NEW: prior breakout quality
                "prior_breakout_count": float(prior_b),
                "prior_breakout_fail_count": float(prior_f),
                "prior_breakout_fail_rate": float(prior_fail_rate),

                # --- NEW: crowding proxies
                "rs_z": float(rs_z),
                "turnover_z": float(turnover_z),

                # categoricals
                "entry_rule": str(entry_rule),
                "pullback_type": str(pullback_type),
                "regime_1d": str(regime_1d),
            }

            # -------------- Score meta (even on rejects) ----------
            p = self._score_winprob_safe(symbol, meta_row)
            pstar = float(getattr(self.winprob, "pstar", None) or self.cfg.get("META_PSTAR", 0.60))

            if p is None or not np.isfinite(p):
                meta_ok = True  # don’t block if model unavailable
                p_disp = "n/a"
            else:
                meta_ok = (p >= pstar)
                p_disp = f"{p:.3f}"

            LOG.debug("  META: p*=%0.2f,  p=%s → %s", pstar, p_disp, "✅" if meta_ok else "❌")

            # ---------- DIAGNOSTICS (always) ----------------------
            if bool(self.cfg.get("DEBUG_SIGNAL_DIAG", True)):
                # Liquidity gate from universe snapshot
                liq_usd = float(univ.get("median_24h_turnover_usd", 0.0) or 0.0)
                liq_thr = float(self.cfg.get("LIQ_MIN_24H_USD", 500_000.0))
                liq_ok_univ = univ.get("liq_ok", None)
                g_liq = (bool(liq_ok_univ) if liq_ok_univ is not None else (liq_usd >= liq_thr))

                # Spec-like gates
                rs_min = float(self.cfg.get("RS_MIN_PERCENTILE", 70))
                vol_needed = float(self.cfg.get("VOL_MULTIPLE", 2.0))
                regime_block = bool(self.cfg.get("REGIME_BLOCK_WHEN_DOWN", True))
                g_rs = rs_pct >= rs_min
                g_vol = vol_mult >= vol_needed
                g_regime = (regime_up == 1.0) or (not regime_block)
                g_micro = (atr_pct >= float(self.cfg.get("ENTRY_MIN_ATR_PCT", 0.0)))
                g_meta = (p is not None and p >= pstar)

                # Try to expose "why" from StrategyEngine when present
                why = None
                for attr in ("reason_tags", "tags", "reasons", "why", "debug", "failures"):
                    if hasattr(verdict, attr):
                        why = getattr(verdict, attr)
                        break

                def _ok(b): return "✅" if b else "❌"
                LOG.debug(
                    (
                        f"\n--- {symbol} | {df5.index[-1].strftime('%Y-%m-%d %H:%M')} UTC ---\n"
                        f"[Strategy verdict] should_enter={should_enter}"
                        f"{'  why=' + str(why) if why is not None else ''}\n"
                        f"[Inputs]\n"
                        f"  Price: {last['close']:.6f}\n"
                        f"  ATR1h: {last['atr_1h']:.6f} ({atr_pct*100:.3f}%)  RSI1h: {last['rsi_1h']:.2f}  ADX1h: {last['adx_1h']:.2f}\n"
                        f"  RS pct: {rs_pct:.1f}%  Liquidity(24h med USD): {liq_usd:,.0f} (thr {liq_thr:,.0f})\n"
                        f"  ETH MACD(4h) hist: {eth_hist:.4f}  macd>signal: {bool(eth_above)}  → regime_up={bool(regime_up)}\n"
                        f"  Donch({don_len}d prev) upper: {don_level:.6f}  dist_atr={don_dist_atr:+.3f}\n"
                        f"  Vol mult (median {int(self.cfg.get('VOL_LOOKBACK_DAYS',30))}d): x{vol_mult:.2f}\n"
                        f"  Pullback/Entry: {pullback_type} + {entry_rule}\n"
                        f"[Gates]\n"
                        f"  RS≥{rs_min:.0f} ............. {_ok(g_rs)}\n"
                        f"  Liquidity≥{liq_thr:,.0f} ..... {_ok(g_liq)}\n"
                        f"  RegimeUp / not blocked ...... {_ok(g_regime)} (block_when_down={regime_block})\n"
                        f"  Volume spike x{vol_needed:.1f} .... {_ok(g_vol)}\n"
                        f"  Micro-ATR min ............... {_ok(g_micro)} (min={float(self.cfg.get('ENTRY_MIN_ATR_PCT',0.0)):.5f})\n"
                        f"  META: p*={pstar:.2f},  p={(p if p is not None else float('nan')):.3f} → {_ok(g_meta)}\n"
                        f"===================================================="
                    )
                )

            # If rules say NO, stop here (we already printed p/p*)
            if not should_enter:
                return None

            # ---------------- VWAP-stack diagnostics -------------
            lookback = int(self.cfg.get("VWAP_STACK_LOOKBACK_BARS", 12))
            band_pct = float(self.cfg.get("VWAP_STACK_BAND_PCT", 0.004))
            try:
                vw = vwap_stack_features(
                    df5[['open','high','low','close','volume']].copy(),
                    lookback_bars=lookback, band_pct=band_pct
                )
                vwap_frac  = float(vw.get("vwap_frac_in_band", 0.0))
                vwap_exp   = float(vw.get("vwap_expansion_pct", 0.0))
                vwap_slope = float(vw.get("vwap_slope_pph", 0.0))
            except Exception as e:
                LOG.error("VWAP-stack calc failed for %s: %s", symbol, e)
                vwap_frac = vwap_exp = vwap_slope = 0.0

            # ---------------- Legacy research feats --------------
            boom_bars = int((cfg.PRICE_BOOM_PERIOD_H * 60) / tf_minutes)
            slowdown_bars = int((cfg.PRICE_SLOWDOWN_PERIOD_H * 60) / tf_minutes)
            df5['price_boom_ago'] = df5['close'].shift(boom_bars)
            df5['price_slowdown_ago'] = df5['close'].shift(slowdown_bars)

            # VWAP dev/z (legacy)
            vwap_bars = int((cfg.GAP_VWAP_HOURS * 60) / tf_minutes)
            vwap_num = (df5['close'] * df5['volume']).shift(1).rolling(vwap_bars).sum()
            vwap_den = df5['volume'].shift(1).rolling(vwap_bars).sum()
            df5['vwap'] = vwap_num / vwap_den
            vwap_dev_raw = df5['close'] - df5['vwap']
            df5['vwap_dev_pct'] = vwap_dev_raw / df5['vwap']
            df5['price_std'] = df5['close'].rolling(vwap_bars).std()
            df5['vwap_z_score'] = vwap_dev_raw / df5['price_std']
            df5['vwap_ok'] = df5['vwap_dev_pct'].abs() <= cfg.GAP_MAX_DEV_PCT
            df5['vwap_consolidated'] = df5['vwap_ok'].rolling(cfg.GAP_MIN_BARS).min().fillna(0).astype(bool)

            df5.dropna(inplace=True)
            if df5.empty:
                return None

            last = df5.iloc[-1]  # refresh after adding vwap cols

            boom_ret_pct = (last['close'] / last['price_boom_ago'] - 1)
            slowdown_ret_pct = (last['close'] / last['price_slowdown_ago'] - 1)
            is_ema_crossed_down = last['ema_fast'] < last['ema_slow']

            # -------------- Optional entry heuristics ------------
            enter_ok = True
            if bool(self.cfg.get("ENTRY_REQUIRE_EMA_CROSS", False)):
                enter_ok = enter_ok and ((last['ema_fast'] > last['ema_slow']) if side == "long" else (last['ema_fast'] < last['ema_slow']))
            if bool(self.cfg.get("ENTRY_REQUIRE_VWAP_CONSOL", False)):
                enter_ok = enter_ok and bool(last.get('vwap_consolidated', False))
            min_atr_pct = float(self.cfg.get("ENTRY_MIN_ATR_PCT", 0.0))
            if min_atr_pct > 0:
                enter_ok = enter_ok and (atr_pct >= min_atr_pct)
            if not enter_ok:
                if bool(self.cfg.get("DEBUG_SIGNAL_DIAG", True)):
                    LOG.debug("— %s vetoed by local entry heuristics.", symbol)
                return None

            # -------------- Compose Signal object ----------------
            signal_obj = Signal(
                symbol=symbol,
                entry=float(last['close']),
                atr=float(last['atr_1h']),
                rsi=float(last['rsi_1h']),
                adx=float(last['adx_1h']),
                atr_pct=float(atr_pct) * 100.0,  # DB stores as %
                market_regime=market_regime,
                price_boom_pct=float(boom_ret_pct),
                price_slowdown_pct=float(slowdown_ret_pct),
                vwap_dev_pct=float(last.get('vwap_dev_pct', 0.0)),
                vwap_z_score=float(last.get('vwap_z_score', 0.0)),
                ret_30d=float(ret_30d),
                ema_fast=float(last['ema_fast']),
                ema_slow=float(last['ema_slow']),
                listing_age_days=int(listing_age_days),
                session_tag=("ASIA" if 0 <= hour_of_day < 8 else "EUROPE" if 8 <= hour_of_day < 16 else "US"),
                day_of_week=day_of_week,
                hour_of_day=hour_of_day,
                vwap_consolidated=bool(last.get('vwap_consolidated', False)),
                is_ema_crossed_down=bool(is_ema_crossed_down),
                side=side,
            )
            signal_obj.vwap_stack_frac = vwap_frac
            signal_obj.vwap_stack_expansion_pct = vwap_exp
            signal_obj.vwap_stack_slope_pph = vwap_slope

            signal_obj.rs_pct = rs_pct
            signal_obj.liq_ok = bool(univ.get("liq_ok", True))
            signal_obj.vol_mult = vol_mult

            signal_obj.don_break_len = don_len
            signal_obj.don_break_level = don_level
            signal_obj.don_dist_atr = don_dist_atr

            signal_obj.eth_macd_hist_4h = eth_hist
            signal_obj.eth_macd_above_signal = bool(eth_above)
            signal_obj.regime_up_flag = bool(regime_up)

            signal_obj.entry_rule = entry_rule
            signal_obj.pullback_type = pullback_type
            signal_obj.regime_1d = regime_1d or ""

            # -------------- Win-prob: reuse p if available -------
            try:
                if p is not None and np.isfinite(p):
                    signal_obj.win_probability = max(0.0, min(1.0, float(p)))
                else:
                    # one more attempt on the fully-computed row, same protocol
                    p2 = self._score_winprob_safe(symbol, meta_row)
                    signal_obj.win_probability = max(0.0, min(1.0, float(p2))) if (p2 is not None and np.isfinite(p2)) else 0.0
            except Exception as e:
                LOG.warning("Failed to score signal for %s: %s", symbol, e)
                signal_obj.win_probability = 0.0

            # -------------- Final INFO line ----------------------
            LOG.info(
                "SIGNAL %s @ %.6f | regime=%s | wp=%.2f%%",
                symbol, float(last['close']), market_regime, signal_obj.win_probability * 100.0
            )
            return signal_obj

        except ccxt.BadSymbol:
            LOG.warning("Invalid symbol on exchange: %s", symbol)
        except Exception as e:
            LOG.error("Error scanning symbol %s: %s", symbol, e, exc_info=True)
        return None

    # ───────────────────── Sizing helpers ─────────────────────

    def _vwap_stack_multiplier(self, stack_frac: float | None, expansion_pct: float | None) -> float:
        """
        VWAP stack multiplier controlled by YAML.
        """
        cfgd = self.cfg
        if not bool(cfgd.get("VWAP_STACK_SIZING_ENABLED", True)):
            return 1.0

        m_lo = float(cfgd.get("VWAP_STACK_MIN_MULTIPLIER", 1.00))
        m_hi = float(cfgd.get("VWAP_STACK_MAX_MULTIPLIER", 1.00))
        if abs(m_hi - m_lo) < 1e-12:
            return m_lo

        frac_min = float(cfgd.get("VWAP_STACK_FRAC_MIN", 0.50))
        frac_good = float(cfgd.get("VWAP_STACK_FRAC_GOOD", 0.70))
        exp_abs_min = float(cfgd.get("VWAP_STACK_EXPANSION_ABS_MIN", 0.006))
        exp_good = float(cfgd.get("VWAP_STACK_EXPANSION_GOOD", 0.015))
        w_frac = float(cfgd.get("VWAP_STACK_FRAC_WEIGHT", 0.6))
        w_exp  = float(cfgd.get("VWAP_STACK_EXP_WEIGHT", 0.4))
        if (w_frac + w_exp) <= 0:
            w_frac, w_exp = 1.0, 0.0

        frac = float(stack_frac) if stack_frac is not None and np.isfinite(stack_frac) else 0.0
        exp  = float(expansion_pct) if expansion_pct is not None and np.isfinite(expansion_pct) else 0.0

        def lin01(x, lo, hi):
            if hi <= lo:
                hi = lo + 1e-6
            return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

        s_frac = lin01(frac, frac_min, frac_good)
        s_exp  = lin01(abs(exp), exp_abs_min, exp_good)

        score = (w_frac * s_frac + w_exp * s_exp) / (w_frac + w_exp)
        return float(m_lo + score * (m_hi - m_lo))

    def _winprob_multiplier(self, wp: float) -> float:
        """
        Map win-probability 'wp' to size multiplier, controlled by YAML.
        """
        cfgd = self.cfg
        if not bool(cfgd.get("WINPROB_SIZING_ENABLED", True)):
            return 1.0
        try:
            wp = float(wp)
        except Exception:
            return 1.0
        if not np.isfinite(wp) or wp <= 0.0:
            return 1.0
        wp = float(np.clip(wp, 0.0, 1.0))

        prob_lo = float(cfgd.get("WINPROB_PROB_FLOOR", cfgd.get("WINPROB_SIZE_FLOOR", 0.50)))
        prob_hi = float(cfgd.get("WINPROB_PROB_CAP",   cfgd.get("WINPROB_SIZE_CAP",   0.90)))
        if prob_hi <= prob_lo:
            prob_hi = prob_lo + 1e-6

        mult_lo = float(cfgd.get("WINPROB_MIN_MULTIPLIER", 0.70))
        mult_hi = float(cfgd.get("WINPROB_MAX_MULTIPLIER", 1.30))
        if abs(mult_hi - mult_lo) < 1e-12:
            return mult_lo

        x = (np.clip(wp, prob_lo, prob_hi) - prob_lo) / (prob_hi - prob_lo)
        return float(mult_lo + x * (mult_hi - mult_lo))

    def _yaml_sizing_multiplier(self, sig: Signal) -> float:
        """
        Apply a sequence of YAML scalers to produce a single size multiplier.
        """
        scalers = self.cfg.get("SIZING_SCALERS", []) or []
        if not isinstance(scalers, list) or not scalers:
            return 1.0

        data = dict(vars(sig))
        data.setdefault("win_probability", float(getattr(sig, "win_probability", 0.0) or 0.0))
        data.setdefault("vwap_stack_frac", getattr(sig, "vwap_stack_frac", None))
        data.setdefault("vwap_stack_expansion_pct", getattr(sig, "vwap_stack_expansion_pct", None))
        data.setdefault("vwap_stack_slope_pph", getattr(sig, "vwap_stack_slope_pph", None))

        def _one_scaler(s: dict) -> float:
            try:
                feat = s.get("feature") or s.get("field")
                if not feat:
                    return 1.0
                x = data.get(feat, s.get("default", 0.0))
                x = float(x) if x is not None and np.isfinite(x) else float(s.get("default", 0.0))

                in_lo = float(s.get("in_min", s.get("min", 0.0)))
                in_hi = float(s.get("in_max", s.get("max", 1.0)))
                out_lo = float(s.get("mult_min", 1.0))
                out_hi = float(s.get("mult_max", 1.0))
                if in_hi <= in_lo:
                    in_hi = in_lo + 1e-9

                x_clamped = min(max(x, in_lo), in_hi)
                t = (x_clamped - in_lo) / (in_hi - in_lo)
                m = out_lo + t * (out_hi - out_lo)

                if "cap_min" in s or "cap_max" in s:
                    cap_min = float(s.get("cap_min", -1e18))
                    cap_max = float(s.get("cap_max", +1e18))
                    m = min(max(m, cap_min), cap_max)
                return float(m)
            except Exception:
                return 1.0

        mult = 1.0
        for s in scalers:
            if isinstance(s, dict):
                mult *= _one_scaler(s)

        gmin = float(self.cfg.get("SIZING_MULT_MIN", 0.0))
        gmax = float(self.cfg.get("SIZING_MULT_MAX", 1e18))
        return float(min(max(mult, gmin if gmin > 0 else 0.0), gmax))

    async def _get_eth_macd_barometer(self) -> Optional[dict]:
        """Latest ETHUSDT 4h MACD dict from the last CLOSED bar: {'macd','signal','hist'}."""
        try:
            eth_ohlcv = await self.exchange.fetch_ohlcv('ETHUSDT', '4h', limit=200)
            if not eth_ohlcv:
                return None
            df = pd.DataFrame(eth_ohlcv, columns=['ts','open','high','low','close','volume'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
            df.set_index('ts', inplace=True)
            if len(df) < 2:
                return None
            df = df.iloc[:-1]  # drop the forming bar
            macd_df = ta.macd(df['close'])
            latest = macd_df.iloc[-1]
            return {"macd": float(latest['macd']),
                    "signal": float(latest['signal']),
                    "hist": float(latest['hist'])}
        except Exception as e:
            LOG.warning("ETH barometer unavailable: %s", e)
            return None

    # ───────────────────── Entry & position lifecycle ─────────────────────

    async def _open_position(self, sig: Signal) -> None:
        """
        Open a LONG or SHORT with:
        - base risk (fixed or % equity)
        - multipliers: ETH barometer, VWAP-stack, WinProb, YAML scalers
        - optional corr guard & soft corr sizing
        - OCO protection: SL + (optional) TP1 + final TP
        - full DB journaling + Telegram
        """
        # --- Block duplicate positions (bot + exchange sanity) ---
        if any(row["symbol"] == sig.symbol for row in self.open_positions.values()):
            return
        try:
            positions = await self.exchange.fetch_positions(symbols=[sig.symbol])
            if positions and positions[0] and float(positions[0].get("info", {}).get("size", 0)) > 0:
                LOG.warning("Pre-flight: found existing exchange position for %s. Abort new entry.", sig.symbol)
                return
        except Exception as e:
            LOG.error("Pre-flight check failed for %s: %s", sig.symbol, e)
            return

        try:
            dd_hours = float(self.cfg.get("DEDUP_WINDOW_HOURS", 8.0))
            recent_open = await self.db.pool.fetchval(
                "SELECT MAX(opened_at) FROM positions WHERE symbol=$1", sig.symbol
            )
            if recent_open and (datetime.now(timezone.utc) - recent_open) < timedelta(hours=dd_hours):
                LOG.info("De-dup window: last %s open at %s < %sh → skip.", sig.symbol, recent_open, dd_hours)
                return
        except Exception as _e:
            LOG.warning("De-dup check failed for %s: %s (continuing)", sig.symbol, _e)

        # --- Win-probability gate (optional) ---
        thr = float(getattr(self.winprob, "pstar", None)
                    or self.cfg.get("MIN_WINPROB_TO_TRADE",
                                    self.cfg.get("META_PROB_THRESHOLD", 0.0)))

        wp  = float(getattr(sig, "win_probability", 0.0) or 0.0)
        if wp < thr:
            LOG.info("Skip %s: WinProb %.3f < threshold %.3f", sig.symbol, wp, thr)
            return

        # --- Resolve side & direction-dependent helpers ---
        side = (getattr(sig, "side", "short") or "short").lower()
        is_long = (side == "long")
        entry_side = "buy" if is_long else "sell"          # entry market order side
        sl_close_side = "sell" if is_long else "buy"       # SL close side
        tp_close_side = "sell" if is_long else "buy"       # TP close side
        sl_trigger_dir = 2 if is_long else 1               # long: falling, short: rising
        tp_trigger_dir = 1 if is_long else 2               # long: rising, short: falling
        side_label = "LONG" if is_long else "SHORT"

        # --- Base risk selection (fixed | percent of equity) ---
        mode = str(self.cfg.get("RISK_MODE", "fixed")).lower()
        fixed_risk = float(self.cfg.get("RISK_USD", 10.0))
        base_risk_usd = fixed_risk
        if mode == "percent":
            try:
                latest_eq = await self.db.latest_equity()
                latest_eq = float(latest_eq) if latest_eq is not None else 0.0
            except Exception:
                latest_eq = 0.0
            pct = float(self.cfg.get("RISK_EQUITY_PCT", 0.01))  # default 1%
            base_risk_usd = max(0.0, latest_eq * pct) if latest_eq > 0 else fixed_risk

        # --- Multipliers ---
        # 1) ETH 4h MACD barometer (closed bar)
        eth_mult = 1.0
        if bool(self.cfg.get("ETH_BAROMETER_ENABLED", True)):
            try:
                eth = await self._get_eth_macd_barometer()
                hist = float(eth.get("hist", 0.0)) if eth else 0.0
                if is_long:
                    cutoff = float(self.cfg.get("ETH_MACD_HIST_CUTOFF_NEG", 0.0))
                    unfavorable = (hist < cutoff)
                else:
                    cutoff = float(self.cfg.get("ETH_MACD_HIST_CUTOFF_POS", 0.0))
                    unfavorable = (hist > cutoff)
                if unfavorable:
                    eth_mult = float(self.cfg.get("UNFAVORABLE_RISK_RESIZE_FACTOR", 0.2))
                    LOG.info("ETH barometer unfavorable → resize x%.2f (hist=%.3f cutoff=%.3f side=%s)",
                             eth_mult, hist, cutoff, side_label)
            except Exception as e:
                LOG.warning("ETH barometer unavailable (%s). Proceeding with base risk.", e)

        # 2) VWAP stack multiplier (from scan diagnostics)
        vw_mult = self._vwap_stack_multiplier(
            getattr(sig, "vwap_stack_frac", None),
            getattr(sig, "vwap_stack_expansion_pct", None),
        )

        # 3) Calibrated win-prob multiplier (0..1 → mult range)
        wp_mult = self._winprob_multiplier(wp)

        # 4) YAML scalers
        def _apply_yaml_scalers(sig_obj: Signal) -> float:
            scalers = self.cfg.get("SIZING_SCALERS", []) or []
            if not isinstance(scalers, list) or not scalers:
                return 1.0
            data = dict(vars(sig_obj))
            # ensure common extras exist
            data.setdefault("win_probability", float(getattr(sig_obj, "win_probability", 0.0) or 0.0))
            data.setdefault("vwap_stack_frac", getattr(sig_obj, "vwap_stack_frac", None))
            data.setdefault("vwap_stack_expansion_pct", getattr(sig_obj, "vwap_stack_expansion_pct", None))
            data.setdefault("vwap_stack_slope_pph", getattr(sig_obj, "vwap_stack_slope_pph", None))

            def linmap(x, in_lo, in_hi, out_lo, out_hi):
                in_hi = in_hi if in_hi > in_lo else (in_lo + 1e-9)
                x = float(x)
                x = min(max(x, in_lo), in_hi)
                t = (x - in_lo) / (in_hi - in_lo)
                return out_lo + t * (out_hi - out_lo)

            mult = 1.0
            for s in scalers:
                if not isinstance(s, dict):
                    continue
                feat = s.get("feature") or s.get("field")
                if not feat:
                    continue
                x = data.get(feat, s.get("default", 0.0))
                try:
                    x = float(x) if x is not None and np.isfinite(x) else float(s.get("default", 0.0))
                except Exception:
                    x = float(s.get("default", 0.0))
                in_lo = float(s.get("in_min", s.get("min", 0.0)))
                in_hi = float(s.get("in_max", s.get("max", 1.0)))
                out_lo = float(s.get("mult_min", 1.0))
                out_hi = float(s.get("mult_max", 1.0))
                m = linmap(x, in_lo, in_hi, out_lo, out_hi)
                if "cap_min" in s or "cap_max" in s:
                    cap_min = float(s.get("cap_min", -1e18))
                    cap_max = float(s.get("cap_max", +1e18))
                    m = min(max(m, cap_min), cap_max)
                mult *= float(m)

            # optional global clamp
            gmin = float(self.cfg.get("SIZING_MULT_MIN", 0.0))
            gmax = float(self.cfg.get("SIZING_MULT_MAX", 1e9))
            mult = float(min(max(mult, gmin if gmin > 0 else 0.0), gmax))
            return mult

        yaml_mult = _apply_yaml_scalers(sig)

        # --- Correlation guard / sizing ---------------------------------------
        stats = {"avg": 0.0, "max": 0.0, "n": 0, "by_symbol": {}}
        want_guard = bool(self.cfg.get("CORR_GUARD_ENABLED", True))
        want_size  = bool(self.cfg.get("CORR_SIZING_ENABLED", False))
        if want_guard or want_size:
            stats = await self._corr_guard_stats(
                sig.symbol,
                timeframe=str(self.cfg.get("CORR_TIMEFRAME","1h")),
                lookback=int(self.cfg.get("CORR_LOOKBACK_BARS", 500)),
                method=str(self.cfg.get("CORR_METHOD","spearman")),
            )

        if want_guard:
            rho_thr = float(self.cfg.get("CORR_MAX_AVG", 0.65))
            if stats["n"] > 0 and stats["avg"] >= rho_thr:
                LOG.info("Correlation guard: avg=%.2f ≥ %.2f with %d open → veto %s",
                         stats["avg"], rho_thr, stats["n"], sig.symbol)
                return  # veto entry

        corr_mult = 1.0
        if want_size and stats["n"] > 0:
            lo = float(self.cfg.get("CORR_SIZING_LO", 0.40))   # multiplier at high corr
            hi = float(self.cfg.get("CORR_SIZING_HI", 1.00))   # multiplier at low corr
            t0 = float(self.cfg.get("CORR_SIZING_RHO_LO", 0.30))
            t1 = float(self.cfg.get("CORR_SIZING_RHO_HI", 0.70))
            r  = np.clip((stats["avg"] - t0) / max(1e-9, (t1 - t0)), 0.0, 1.0)
            corr_mult = float(hi + (lo - hi) * r)

        # --- Combine multipliers & clamp final risk -----------------------------------
        # Ensure floats (avoids accidental Decimal/str propagation)
        base_risk_usd = float(base_risk_usd)
        eth_mult      = float(eth_mult)
        vw_mult       = float(vw_mult)
        wp_mult       = float(wp_mult)
        yaml_mult     = float(yaml_mult)
        corr_mult     = float(corr_mult)

        pre_risk_usd = base_risk_usd * eth_mult * vw_mult * wp_mult * yaml_mult * corr_mult

        risk_usd = pre_risk_usd
        min_cap  = self.cfg.get("RISK_USD_MIN")
        max_cap  = self.cfg.get("RISK_USD_MAX")
        clamped  = None  # "min=…" | "max=…"

        if min_cap is not None and risk_usd < float(min_cap):
            risk_usd = float(min_cap)
            clamped = f"min={risk_usd:.2f}"

        if max_cap is not None and risk_usd > float(max_cap):
            risk_usd = float(max_cap)
            clamped = f"max={risk_usd:.2f}"

        sig.risk_usd = float(risk_usd)

        # Log with deferred formatting (best practice for logging)
        if clamped:
            LOG.info(
                "Sizing(%s) base=%.2f · ETH×=%.2f · VWAP×=%.2f · WP×=%.2f · YAML×=%.2f · CORR×=%.2f ⇒ pre=%.2f → risk=%.2f (clamped: %s)",
                side_label, base_risk_usd, eth_mult, vw_mult, wp_mult, yaml_mult, corr_mult, pre_risk_usd, risk_usd, clamped
            )
        else:
            LOG.info(
                "Sizing(%s) base=%.2f · ETH×=%.2f · VWAP×=%.2f · WP×=%.2f · YAML×=%.2f · CORR×=%.2f ⇒ risk=%.2f",
                side_label, base_risk_usd, eth_mult, vw_mult, wp_mult, yaml_mult, corr_mult, risk_usd
            )

        # --- Compute size from live price / ATR distance ---
        try:
            ticker = await self.exchange.fetch_ticker(sig.symbol)
            px = float(
                ticker.get("last")
                or ticker.get("mark")
                or (ticker.get("info", {}) or {}).get("markPrice")
                or ticker.get("close")
                or ticker.get("bid")
                or ticker.get("ask")
            )
        except Exception as e:
            LOG.error("Failed to fetch live ticker for %s: %s", sig.symbol, e)
            return
        if not np.isfinite(px) or px <= 0:
            LOG.error("No usable price for %s (ticker=%s).", sig.symbol, ticker)
            return

        sl_mult = float(self.cfg.get("SL_ATR_MULT", 1.8))
        stop_preview = px - sl_mult * float(sig.atr) if is_long else px + sl_mult * float(sig.atr)
        stop_dist = abs(px - stop_preview)
        if stop_dist <= 0:
            LOG.warning("Stop distance is zero for %s. Skip.", sig.symbol)
            return
        intended_size = max(risk_usd / stop_dist, 0.0)

        try:
            mkt = self.exchange._exchange.market(sig.symbol)  # unified market
        except Exception:
            mkt = None

        min_amt = None
        min_cost = None
        step_amt = None
        if mkt:
            lims = (mkt.get("limits") or {})
            amt_lims = (lims.get("amount") or {})
            cost_lims = (lims.get("cost") or {})
            min_amt = amt_lims.get("min")  # e.g., BTC perp: 0.001
            min_cost = cost_lims.get("min")  # not always set on Bybit, but handle if present
            step_amt = (mkt.get("precision") or {}).get("amount", None)

        # Round to precision grid first (truncates down), then bump up if below min
        size_prec = intended_size
        try:
            size_prec = float(self.exchange._exchange.amount_to_precision(sig.symbol, intended_size))
        except Exception:
            pass

        # If the precision rounding pushed us below min amount, bump to min (or ceil to the next step)
        def _ceil_to_step(x: float, step: Optional[float]) -> float:
            import math as _m
            if not step or step <= 0:
                return x
            return _m.ceil(x / step) * step

        if (min_amt is not None) and (size_prec < float(min_amt)):
            auto_pad = bool(self.cfg.get("MIN_TRADE_AUTOPAD_ENABLED", True))
            cap_usd = float(self.cfg.get("MIN_TRADE_AUTOPAD_CAP_USD", 5.0))
            bumped_size = max(float(min_amt), _ceil_to_step(size_prec, step_amt))
            required_risk = bumped_size * stop_dist  # risk = size * stopDistance
            if auto_pad and required_risk <= cap_usd:
                LOG.info("Auto-padding risk to meet min amount: size %.10f -> %.10f (risk %.4f -> %.4f)",
                         size_prec, bumped_size, risk_usd, required_risk)
                risk_usd = float(required_risk)
                size_prec = bumped_size
            else:
                LOG.warning("Size %.10f < min_amt %.10f for %s; risk needed ≈ %.4f USDT. "
                            "Auto-pad=%s cap=%.2f. Skipping entry.",
                            size_prec, float(min_amt), sig.symbol, required_risk, auto_pad, cap_usd)
                return

        # If there is a min cost (notional) requirement, ensure we meet it
        if (min_cost is not None) and mkt:
            try:
                last_px = float((await self.exchange.fetch_ticker(sig.symbol))["last"])
                notional = last_px * size_prec
                if notional < float(min_cost):
                    auto_pad = bool(self.cfg.get("MIN_TRADE_AUTOPAD_ENABLED", True))
                    cap_usd = float(self.cfg.get("MIN_TRADE_AUTOPAD_CAP_USD", 5.0))
                    need_size = float(min_cost) / last_px
                    need_size = max(need_size, size_prec)
                    need_size = _ceil_to_step(need_size, step_amt)
                    required_risk = need_size * stop_dist
                    if auto_pad and required_risk <= cap_usd:
                        LOG.info("Auto-padding for min notional: size %.10f -> %.10f (risk %.4f -> %.4f)",
                                 size_prec, need_size, risk_usd, required_risk)
                        risk_usd = float(required_risk)
                        size_prec = float(need_size)
                    else:
                        LOG.warning("Notional %.4f < min_cost %.4f on %s; needed risk ≈ %.4f. "
                                    "Auto-pad=%s cap=%.2f. Skipping.",
                                    notional, float(min_cost), sig.symbol, required_risk, auto_pad, cap_usd)
                        return
            except Exception as e:
                LOG.warning("Min-cost check failed for %s: %s (continuing)", sig.symbol, e)

        # Final safety: re-apply ccxt precision to the adjusted size
        try:
            size_prec = float(self.exchange._exchange.amount_to_precision(sig.symbol, size_prec))
        except Exception:
            pass

        if size_prec <= 0:
            LOG.warning("Final size <= 0 after min/precision checks; skipping entry.")
            return

        intended_size = size_prec

        # --- Ensure leverage/mode ---
        try:
            await self._ensure_leverage(sig.symbol)
        except Exception:
            return

        # --- Entry: market order with unique CID ---
        entry_cid = create_unique_cid(f"ENTRY_{sig.symbol}")
        try:
            await self.exchange.create_market_order(
                sig.symbol, entry_side, intended_size,
                params={"clientOrderId": entry_cid, "category": "linear"}
            )
            LOG.info("Market %s sent for %s (CID=%s)", entry_side.upper(), sig.symbol, entry_cid)
        except Exception as e:
            LOG.error("Market %s failed for %s: %s", entry_side.upper(), sig.symbol, e)
            return

        # --- Confirm position appears on exchange ---
        actual_size = 0.0
        actual_entry_price = 0.0
        live_position = None
        for _ in range(20):
            await asyncio.sleep(0.5)
            try:
                positions = await self.exchange.fetch_positions(symbols=[sig.symbol])
                pos = next((p for p in positions if p.get("info", {}).get("symbol") == sig.symbol), None)
                if pos and float(pos.get("info", {}).get("size", 0)) > 0:
                    live_position = pos
                    actual_size = float(pos["info"]["size"])
                    actual_entry_price = float(pos["info"]["avgPrice"])
                    break
            except Exception as e:
                LOG.warning("Confirm loop failed for %s: %s", sig.symbol, e)

        if not live_position:
            LOG.error("Entry failed to confirm for %s; no exchange position appeared.", sig.symbol)
            return

        slippage_usd = (actual_entry_price - px) * actual_size if is_long else (px - actual_entry_price) * actual_size
        LOG.info("Entry confirmed %s %s: size=%.6f @ %.6f (slip $%.4f)",
                 side_label, sig.symbol, actual_size, actual_entry_price, slippage_usd)

        # --- Protective levels from actual entry ---
        stop_price = (actual_entry_price - sl_mult * float(sig.atr)) if is_long else (actual_entry_price + sl_mult * float(sig.atr))

        # --- Persist to DB (then add order CIDs) ---
        now = datetime.now(timezone.utc)
        exit_deadline = None
        if self.cfg.get("TIME_EXIT_HOURS_ENABLED", False):
            exit_deadline = now + timedelta(hours=int(self.cfg.get("TIME_EXIT_HOURS", 4)))
        elif self.cfg.get("TIME_EXIT_ENABLED", False):
            exit_deadline = now + timedelta(days=int(self.cfg.get("TIME_EXIT_DAYS", 10)))

        payload = {
            "symbol": sig.symbol,
            "side": side_label,
            "size": float(actual_size),
            "entry_price": float(actual_entry_price),
            "trailing_active": False,
            "atr": float(sig.atr),
            "status": "OPEN",
            "opened_at": now,
            "exit_deadline": exit_deadline,
            "entry_cid": entry_cid,
            "market_regime_at_entry": sig.market_regime,
            "risk_usd": float(sig.risk_usd),
            "slippage_usd": float(slippage_usd),
            # feature audit
            "rsi_at_entry": float(sig.rsi),
            "adx_at_entry": float(sig.adx),
            "atr_pct_at_entry": float(sig.atr_pct),
            "price_boom_pct_at_entry": float(sig.price_boom_pct),
            "price_slowdown_pct_at_entry": float(sig.price_slowdown_pct),
            "vwap_dev_pct_at_entry": float(getattr(sig, "vwap_dev_pct", 0.0)),
            "vwap_z_at_entry": float(getattr(sig, "vwap_z_score", 0.0)),
            "ret_30d_at_entry": float(sig.ret_30d),
            "ema_fast_at_entry": float(sig.ema_fast),
            "ema_slow_at_entry": float(sig.ema_slow),
            "listing_age_days_at_entry": int(sig.listing_age_days),
            "session_tag_at_entry": sig.session_tag,
            "day_of_week_at_entry": int(sig.day_of_week),
            "hour_of_day_at_entry": int(sig.hour_of_day),
            "vwap_consolidated_at_entry": bool(sig.vwap_consolidated),
            "is_ema_crossed_down_at_entry": bool(sig.is_ema_crossed_down),
            "win_probability_at_entry": float(getattr(sig, "win_probability", 0.0)),
            # VWAP-stack extras
            "vwap_stack_frac_at_entry": float(getattr(sig, "vwap_stack_frac", 0.0)) if getattr(sig, "vwap_stack_frac", None) is not None else None,
            "vwap_stack_expansion_pct_at_entry": float(getattr(sig, "vwap_stack_expansion_pct", 0.0)) if getattr(sig, "vwap_stack_expansion_pct", None) is not None else None,
            "vwap_stack_slope_pph_at_entry": float(getattr(sig, "vwap_stack_slope_pph", 0.0)) if getattr(sig, "vwap_stack_slope_pph", None) is not None else None,
        }

        try:
            pid = await self.db.insert_position(payload)
            LOG.info("Inserted position %s (id=%s)", sig.symbol, pid)

            # --- Protective orders ---
            sl_cid = create_stable_cid(pid, "SL")
            await self.exchange.create_order(
                sig.symbol, "market", sl_close_side, actual_size, None,
                params={
                    "triggerPrice": float(stop_price),
                    "clientOrderId": sl_cid, "category": "linear",
                    "reduceOnly": True, "closeOnTrigger": True, "triggerDirection": sl_trigger_dir
                }
            )

            tp1_cid, tp_final_cid = None, None
            if self.cfg.get("PARTIAL_TP_ENABLED", False):
                tp1_cid = create_stable_cid(pid, "TP1")
                tp1_mult = float(self.cfg.get("PARTIAL_TP_ATR_MULT", self.cfg.get("TP1_ATR_MULT", 4.0)))
                tp1_price = actual_entry_price + (tp1_mult * float(sig.atr) * (+1 if is_long else -1))
                qty_tp1 = actual_size * float(self.cfg.get("PARTIAL_TP_PCT", 0.5))
                qty_tp1 = max(0.0, min(qty_tp1, actual_size))
                await self.exchange.create_order(
                    sig.symbol, "market", tp_close_side, qty_tp1, None,
                    params={
                        "triggerPrice": float(tp1_price),
                        "clientOrderId": tp1_cid, "category": "linear",
                        "reduceOnly": True, "closeOnTrigger": True, "triggerDirection": tp_trigger_dir
                    }
                )
                if self.cfg.get("FINAL_TP_ENABLED", True):
                    tp_final_cid = create_stable_cid(pid, "TP_FINAL")
                    tp_mult = float(self.cfg.get("FINAL_TP_ATR_MULT", self.cfg.get("TP_ATR_MULT", 8.0)))
                    tp_final_price = actual_entry_price + (tp_mult * float(sig.atr) * (+1 if is_long else -1))
                    remainder = max(0.0, actual_size - qty_tp1)
                    await self.exchange.create_order(
                        sig.symbol, "market", tp_close_side, remainder, None,
                        params={
                            "triggerPrice": float(tp_final_price),
                            "clientOrderId": tp_final_cid, "category": "linear",
                            "reduceOnly": True, "closeOnTrigger": True, "triggerDirection": tp_trigger_dir
                        }
                    )
            elif self.cfg.get("FINAL_TP_ENABLED", True):
                tp_final_cid = create_stable_cid(pid, "TP_FINAL")
                tp_mult = float(self.cfg.get("FINAL_TP_ATR_MULT", self.cfg.get("TP_ATR_MULT", 8.0)))
                tp_price = actual_entry_price + (tp_mult * float(sig.atr) * (+1 if is_long else -1))
                await self.exchange.create_order(
                    sig.symbol, "market", tp_close_side, actual_size, None,
                    params={
                        "triggerPrice": float(tp_price),
                        "clientOrderId": tp_final_cid, "category": "linear",
                        "reduceOnly": True, "closeOnTrigger": True, "triggerDirection": tp_trigger_dir
                    }
                )

            # --- Persist order CIDs & stop ---
            await self.db.update_position(
                pid,
                status="OPEN",
                stop_price=float(stop_price),
                sl_cid=sl_cid,
                tp1_cid=tp1_cid,
                tp_final_cid=tp_final_cid,
            )

            # --- Mirror to in-memory map ---
            row = await self.db.pool.fetchrow("SELECT * FROM positions WHERE id=$1", pid)
            self.open_positions[pid] = dict(row)

            # --- Telegram ---
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

            # thresholds from config
            rs_min       = float(self.cfg.get("RS_MIN_PERCENTILE", 70))
            vol_needed   = float(self.cfg.get("VOL_MULTIPLE", 2.0))
            regime_block = bool(self.cfg.get("REGIME_BLOCK_WHEN_DOWN", True))
            min_atr_pct  = float(self.cfg.get("ENTRY_MIN_ATR_PCT", 0.0))
            meta_thresh  = float(self.cfg.get("META_PROB_THRESHOLD", 0.0))

            # values from signal
            rs_pct       = float(getattr(sig, "rs_pct", 0.0) or 0.0)
            liq_ok       = bool(getattr(sig, "liq_ok", True))
            vol_mult     = float(getattr(sig, "vol_mult", 0.0) or 0.0)
            don_len      = getattr(sig, "don_break_len", None)
            don_level    = getattr(sig, "don_break_level", None)
            don_dist_atr = getattr(sig, "don_dist_atr", None)
            eth_hist     = float(getattr(sig, "eth_macd_hist_4h", 0.0) or 0.0)
            regime_up    = bool(getattr(sig, "regime_up_flag", False))
            p            = float(getattr(sig, "win_probability", 0.0) or 0.0)

            gates = {
                "RS": (rs_pct >= rs_min),
                "LIQ": liq_ok,
                "REGIME": (regime_up or not regime_block),
                "VOL": (vol_mult >= vol_needed),
                "ATR_MIN": ((float(sig.atr_pct) / 100.0) >= min_atr_pct),
                "META": (p >= meta_thresh) if meta_thresh > 0 else True,
            }
            ok = lambda b: "✅" if b else "❌"

            text = (
                f"📥 {side_label} {sig.symbol} opened ({ts} UTC)\n"
                f"• Size: {actual_size:.6f}  @ {actual_entry_price:.6f}\n"
                f"• Risk: ${float(sig.risk_usd):.2f}  | ATR(1h): {float(sig.atr):.6f}\n"
                f"• SL: {float(stop_price):.6f} | TP1/TPf: {'ON' if bool(self.cfg.get('FINAL_TP_ENABLED', True)) else 'OFF'}\n"
                f"• Regime: {sig.market_regime} | WinProb: {p:.3f}\n"
                f"• RS%: {rs_pct:.1f}  Vol×: {vol_mult:.2f}  ETH4h hist: {eth_hist:+.4f}\n"
                f"• Donch({don_len}d)={don_level:.6f}  distATR={don_dist_atr:+.2f}\n"
                f"• Gates: RS {ok(gates['RS'])} | LIQ {ok(gates['LIQ'])} | REG {ok(gates['REGIME'])} | "
                f"VOL {ok(gates['VOL'])} | ATR {ok(gates['ATR_MIN'])} | META {ok(gates['META'])}"
            )
            try:
                await self.tg.send(text)
            except Exception:
                pass

        except Exception as e:
            LOG.error("Open-position postflight failed for %s: %s", sig.symbol, e, exc_info=True)
            return

    # ───────────────────── Position maintenance / exits ─────────────────────

    async def _all_open_orders(self, symbol: str) -> list[dict]:
        """Fetch all open orders for a symbol (robust across ccxt versions)."""
        try:
            orders = await self.exchange.fetch_open_orders(symbol, params={"category": "linear"})
            return orders or []
        except Exception as e:
            LOG.warning("fetch_open_orders failed for %s: %s", symbol, e)
            try:
                # Fallback without params
                return (await self.exchange.fetch_open_orders(symbol)) or []
            except Exception as e2:
                LOG.warning("fetch_open_orders fallback failed for %s: %s", symbol, e2)
                return []

    async def _fetch_live_position(self, symbol: str) -> Optional[dict]:
        """Return unified position dict for symbol, or None if flat."""
        try:
            positions = await self.exchange.fetch_positions(symbols=[symbol])
            pos = next((p for p in positions if (p.get("info", {}) or {}).get("symbol") == symbol), None)
            if pos and float((pos.get("info", {}) or {}).get("size", 0.0)) > 0:
                return pos
            return None
        except Exception as e:
            LOG.warning("fetch_positions failed for %s: %s", symbol, e)
            return None

    async def _close_market(self, pid: int, pos_row: dict, reason: str = "MANUAL") -> None:
        """Market-close entire remaining size, then finalize."""
        try:
            symbol = pos_row["symbol"]
            side = pos_row["side"].upper()
            is_long = (side == "LONG")
            close_side = "sell" if is_long else "buy"

            live = await self._fetch_live_position(symbol)
            if not live:
                LOG.info("Close-market: position already flat for %s.", symbol)
                await self._finalize_zero(pid, pos_row, reason=reason)
                return
            size = float(live["info"]["size"])
            if size <= 0:
                await self._finalize_zero(pid, pos_row, reason=reason)
                return

            coid = create_unique_cid(f"CLOSE_{symbol}")
            await self.exchange.create_market_order(symbol, close_side, size, params={"clientOrderId": coid, "category": "linear"})
            LOG.info("Market-close sent for %s (%s) size=%.6f", symbol, reason, size)
        except Exception as e:
            LOG.error("Close-market failed (pid=%s): %s", pid, e)

    async def _activate_trailing(self, pid: int, pos_row: dict) -> None:
        """
        Idempotently switch from static SL to trailing SL after TP1 fill.
        - Cancels prior SL order (if exists).
        - Places a new trailing-style stop using ATR-based offset.
        """
        try:
            if bool(pos_row.get("trailing_active")):
                LOG.debug("Trailing already active for pid=%s; skip.", pid)
                return

            symbol = pos_row["symbol"]
            side = pos_row["side"].upper()
            is_long = (side == "LONG")
            sl_close_side = "sell" if is_long else "buy"
            trigger_dir = 2 if is_long else 1

            # compute new trailing trigger based on latest price & ATR multiples
            live = await self._fetch_live_position(symbol)
            if not live:
                LOG.info("No live position for %s when activating trailing; will finalize if flat.", symbol)
                await self._finalize_zero(pid, pos_row, reason="TRAIL_ACTIVATE_FLAT")
                return

            # Pull a fresh ticker for precision
            tick = await self.exchange.fetch_ticker(symbol)
            last_px = float(
                tick.get("last")
                or tick.get("mark")
                or (tick.get("info", {}) or {}).get("markPrice")
                or tick.get("close")
                or tick.get("bid")
                or tick.get("ask")
            )
            if not np.isfinite(last_px) or last_px <= 0:
                LOG.warning("Trailing activate: no usable price for %s; abort.", symbol)
                return

            atr = float(pos_row.get("atr") or 0.0)
            if not np.isfinite(atr) or atr <= 0:
                LOG.warning("Trailing activate: invalid ATR for %s; abort.", symbol)
                return

            # Use a local variable for trailing ATR multiple (prevents NameError)
            tp_mult = float(self.cfg.get("TRAIL_ATR_MULT", 2.5))
            # initial trailing trigger price
            trigger_price = last_px - tp_mult * atr if is_long else last_px + tp_mult * atr

            # cancel existing SL if we have its CID
            old_sl_cid = pos_row.get("sl_cid")
            if old_sl_cid:
                try:
                    await self.exchange.cancel_order(None, symbol, {"clientOrderId": old_sl_cid, "acknowledged": True})
                    LOG.info("Canceled static SL (cid=%s) before trailing for %s.", old_sl_cid, symbol)
                except Exception as e:
                    LOG.warning("Failed to cancel old SL (cid=%s) on %s: %s", old_sl_cid, symbol, e)

            # Create new trailing-style SL (reduce-only conditional). Keep Bybit params as-is.
            sl_trail_cid = create_stable_cid(pid, "SL_TRAIL")
            size = float((live.get("info", {}) or {}).get("size", 0.0))
            await self.exchange.create_order(
                symbol, "market", sl_close_side, size, None,
                params={
                    "triggerPrice": float(trigger_price),
                    "clientOrderId": sl_trail_cid, "category": "linear",
                    "reduceOnly": True, "closeOnTrigger": True, "triggerDirection": trigger_dir
                }
            )

            await self.db.update_position(pid, trailing_active=True, sl_trail_cid=sl_trail_cid)
            self.open_positions[pid]["trailing_active"] = True
            self.open_positions[pid]["sl_trail_cid"] = sl_trail_cid

            try:
                await self.tg.send(f"🧵 Trailing SL activated for {symbol} (tp_mult={tp_mult:.2f}, trigger={trigger_price:.6f}).")
            except Exception:
                pass

        except Exception as e:
            LOG.error("Trailing activation failed (pid=%s): %s", pid, e, exc_info=True)

    async def _finalize_zero(self, pid: int, pos_row: dict, *, reason: str = "FLAT") -> None:
        """
        Finalize if the on-exchange size is zero. Cancels any child orders,
        computes PnL using DB helper, and notifies Telegram. Uses a per-pid
        backoff to avoid thrashing.
        """
        now = datetime.now(timezone.utc)
        back = self._zero_finalize_backoff.get(pid)
        if back and now < back:
            return
        self._zero_finalize_backoff[pid] = now + timedelta(seconds=10)

        try:
            symbol = pos_row["symbol"]
            await self._cancel_reducing_orders(symbol, pos_row)

            # Compute PnL / fees via DB helper (assumes triggers & fills tracked there)
            rec = await self.db.finalize_position(pid, reason=reason)
            self.open_positions.pop(pid, None)

            msg = (
                f"✅ Closed {pos_row['side'].upper()} {symbol}\n"
                f"• Entry: {float(pos_row['entry_price']):.6f}\n"
                f"• Size:  {float(pos_row['size']):.6f}\n"
                f"• Result: PnL ${float(rec.get('pnl', 0.0)):.2f} | "
                f"ROI {float(rec.get('roi_pct', 0.0)):.2f}% | Reason: {reason}"
            )
            try:
                await self.tg.send(msg)
            except Exception:
                pass
        except Exception as e:
            LOG.error("Finalize-zero failed for pid=%s: %s", pid, e, exc_info=True)

    async def _housekeep_positions_once(self) -> None:
        """
        Periodic maintenance:
        - Detect TP1 fill (via size reduction) and activate trailing (idempotent).
        - Enforce time-based exit if deadline reached.
        - Finalize when size hits zero.
        """
        rows = list(self.open_positions.items())
        for pid, pos in rows:
            try:
                symbol = pos["symbol"]
                side = pos["side"].upper()
                is_long = (side == "LONG")

                # time exit
                deadline = pos.get("exit_deadline")
                if deadline and isinstance(deadline, datetime) and datetime.now(timezone.utc) >= deadline:
                    LOG.info("Time-exit reached for pid=%s %s → market close.", pid, symbol)
                    await self._close_market(pid, pos, reason="TIME_EXIT")
                    # finalization will be handled in next cycle when size=0
                    continue

                live = await self._fetch_live_position(symbol)
                if not live:
                    await self._finalize_zero(pid, pos, reason="FLAT")
                    continue

                live_size = float((live.get("info", {}) or {}).get("size", 0.0))
                if live_size <= 0:
                    await self._finalize_zero(pid, pos, reason="FLAT")
                    continue

                # Detect TP1 fill by size shrink
                if self.cfg.get("PARTIAL_TP_ENABLED", False) and not bool(pos.get("trailing_active")):
                    init_size = float(pos.get("size", 0.0))
                    tp_pct = float(self.cfg.get("PARTIAL_TP_PCT", 0.5))
                    threshold = max(0.0, init_size * (1.0 - tp_pct) * 0.99)  # tolerance for fees/slippage
                    if live_size <= threshold:
                        LOG.info("TP1 inferred filled for pid=%s %s → activate trailing.", pid, symbol)
                        await self._activate_trailing(pid, pos)

            except Exception as e:
                LOG.warning("Housekeep loop error (pid=%s): %s", pid, e)

    # ───────────────────── Scanning loop / orchestration ─────────────────────

    async def _refresh_universe_ctx(self) -> None:
        """
        Refresh universe context (with cache) at most once per TTL window.
        Persists rs_z and turnover_z as requested.
        """
        try:
            if (datetime.now(timezone.utc) - self._universe_ctx_ts) < timedelta(
                minutes=int(self.cfg.get("UNIVERSE_REFRESH_MINUTES", 120))
            ):
                return
            cached = self._load_universe_cache_if_fresh()
            if cached is not None:
                self._universe_ctx = cached
                self._universe_ctx_ts = datetime.now(timezone.utc)
                return
            ctx = await self._build_universe_context()
            self._universe_ctx = ctx
            self._universe_ctx_ts = datetime.now(timezone.utc)
            self._save_universe_cache(ctx)
        except Exception as e:
            LOG.warning("Universe context refresh failed: %s", e)

    def _resolve_side(self, verdict) -> str:
        """
        Resolve side from strategy verdict (default SHORT).
        Accepts 'long'/'short', True/False, or explicit strings.
        """
        raw = getattr(verdict, "side", None)
        if raw is None:
            raw = self.cfg.get("DEFAULT_SIDE", "short")
        s = str(raw).strip().lower()
        if s in ("long", "buy", "bull", "1", "true", "yes"):
            return "long"
        if s in ("short", "sell", "bear", "0", "false", "no"):
            return "short"
        return "short"

    async def _scan_once(self):
        """Single full-universe scan-and-trade pass."""
        if self.paused or not self.risk.can_trade():
            return
        try:
            await self._refresh_universe_ctx()
        except Exception:
            pass

        try:
            market_regime = await self.regime_detector.get_current_regime()
        except Exception as e:
            LOG.warning("Regime unavailable: %s", e)
            market_regime = "UNKNOWN"

        try:
            eth_macd = await self._get_eth_macd_barometer()
        except Exception:
            eth_macd = None

        # Optional: global block if regime down and config blocks
        if bool(self.cfg.get("GLOBAL_REGIME_BLOCK", False)):
            try:
                if isinstance(eth_macd, dict):
                    up = (float(eth_macd.get("hist", 0.0)) > 0.0) and (float(eth_macd.get("macd", 0.0)) > float(eth_macd.get("signal", 0.0)))
                else:
                    up = False
                if not up and bool(self.cfg.get("REGIME_BLOCK_WHEN_DOWN", True)):
                    LOG.info("Global regime gate blocking entries this pass.")
                    return
            except Exception:
                pass

        # Iterate symbols
        for sym in list(self.symbols):
            try:
                lock = self.symbol_locks[sym]
                if lock.locked():
                    continue
                async with lock:
                    sig = await self._scan_symbol_for_signal(sym, market_regime, eth_macd, gov_ctx=None)
                    if sig is None:
                        continue
                    await self._open_position(sig)
            except Exception as e:
                LOG.error("Scan error for %s: %s", sym, e)

    async def _loop_scan(self):
        """Main scanning loop."""
        interval = float(self.cfg.get("SCAN_INTERVAL_SEC", 30.0))
        while True:
            try:
                await self._scan_once()
            except Exception as e:
                LOG.error("Scan pass failed: %s", e, exc_info=True)
            await asyncio.sleep(interval)

    async def _loop_housekeep(self):
        """Maintenance loop for open positions."""
        interval = float(self.cfg.get("HOUSEKEEP_INTERVAL_SEC", 5.0))
        while True:
            try:
                await self._housekeep_positions_once()
            except Exception as e:
                LOG.error("Housekeep pass failed: %s", e, exc_info=True)
            await asyncio.sleep(interval)

    # ───────────────────── Public controls ─────────────────────

    async def pause(self):
        self.paused = True
        try:
            await self.tg.send("⏸️ Trading paused.")
        except Exception:
            pass

    async def resume(self):
        self.paused = False
        try:
            await self.tg.send("▶️ Trading resumed.")
        except Exception:
            pass

    async def start(self):
        """Start background loops."""
        self.tasks = [
            asyncio.create_task(self._loop_scan(), name="scan-loop"),
            asyncio.create_task(self._loop_housekeep(), name="housekeep-loop"),
        ]
        LOG.info("LiveTrader loops started: %d task(s).", len(self.tasks))

    async def stop(self):
        """Cancel background loops."""
        for t in list(self.tasks):
            try:
                t.cancel()
            except Exception:
                pass
        self.tasks.clear()
        LOG.info("LiveTrader loops stopped.")

# ────────────────────────────── Helpers ──────────────────────────────────────

def create_unique_cid(tag: str) -> str:
    """
    Short unique clientOrderId safe for Bybit/ccxt (≤ 36 chars).
    """
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    rnd = secrets.token_hex(3)
    return f"bot_{tag}_{ts}_{rnd}"[:36]

def create_stable_cid(pid: int, tag: str) -> str:
    """
    Stable clientOrderId derived from position id & tag (≤ 36 chars).
    """
    base = f"bot_{pid}_{tag}"
    return base[:36]

# ────────────────────────────── Entrypoint ───────────────────────────────────

async def main():
    try:
        settings = Settings()  # from .env
        cfg_yaml = load_yaml(CONFIG_PATH)
        trader = LiveTrader(settings, cfg_yaml)
        await trader.start()

        # Keep running until cancelled
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        LOG.error("Fatal error in main(): %s", e, exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
