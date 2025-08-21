"""
live_trader.py – v4.1 (Clean, YAML sizing + meta gate, robust listing-age)
=============================================================================
- Robust UTC-safe listing-age helper (uses cache, then earliest OHLCV bar).
- Removes utcnow() deprecation warnings (timezone-aware everywhere).
- Fixes NameError/UnboundLocal (no stray 'sym', no undefined listing_age_days).
- Filters get the correct listing_age_days from the built signal.
- Keeps StrategyEngine, YAML scalers, meta-prob gate, DB/Telegram/Bybit V5, etc.
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
            await telegram.send("❌ Kill-switch: max loss streak reached")

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
        """Volatility regime via Markov Switching (2 states)."""
        try:
            model = sm.tsa.MarkovRegression(daily_returns.dropna(), k_regimes=2, switching_variance=True)
            results = model.fit(disp=False)
            low_vol_regime_idx = np.argmin(results.params[-2:])
            vol_regimes = np.where(
                results.smoothed_marginal_probabilities[low_vol_regime_idx] > 0.5,
                "LOW_VOL", "HIGH_VOL"
            )
            return pd.Series(vol_regimes, index=daily_returns.dropna().index, name="vol_regime")
        except Exception as e:
            LOG.warning("Markov vol regime failed: %s. Defaulting to UNKNOWN.", e)
            return pd.Series("UNKNOWN", index=daily_returns.index, name="vol_regime")

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
            feats = getattr(self.winprob, "feature_order", None)

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
        Returns { symbol: {"rs_pct": float, "median_24h_turnover_usd": float} }
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
            }

        LOG.info("Universe context ready (%d symbols).", len(out))
        return out

    # ---- Robust listing-age helper (UTC-safe) --------------------------------
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
                # try both top-level and .info for common exchanges
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
                # ccxt unifies as fundingRate (8h); fallback to info
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
        """Latest ETHUSDT 4h MACD dict: {'macd','signal','hist'}."""
        try:
            eth_ohlcv = await self.exchange.fetch_ohlcv('ETHUSDT', '4h', limit=200)
            if not eth_ohlcv:
                return None
            df = pd.DataFrame(eth_ohlcv, columns=['ts','open','high','low','close','volume'])
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
        # 1) ETH 4h MACD barometer
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
        wp = float(getattr(sig, "win_probability", 0.0) or 0.0)
        wp_mult = self._winprob_multiplier(wp)

        # 4) YAML scalers (generic, linear maps feature→mult). Local helper.
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

        # Combine & clamp final risk
        risk_usd = base_risk_usd * eth_mult * vw_mult * wp_mult * yaml_mult
        if self.cfg.get("RISK_USD_MIN") is not None:
            risk_usd = max(float(self.cfg["RISK_USD_MIN"]), risk_usd)
        if self.cfg.get("RISK_USD_MAX") is not None:
            risk_usd = min(float(self.cfg["RISK_USD_MAX"]), risk_usd)
        sig.risk_usd = float(risk_usd)

        LOG.info("Sizing(%s) base=%.2f · eth×=%.2f · vwap×=%.2f · wp=%.2f (wp×=%.2f) · yaml×=%.2f → risk=%.2f",
                 side_label, base_risk_usd, eth_mult, vw_mult, wp, wp_mult, yaml_mult, risk_usd)

        # --- Compute size from live price / ATR distance ---
        try:
            ticker = await self.exchange.fetch_ticker(sig.symbol)
            px = float(ticker["last"])
        except Exception as e:
            LOG.error("Failed to fetch live ticker for %s: %s", sig.symbol, e)
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

            # values from signal (set during scan)
            rs_pct      = float(getattr(sig, "rs_pct", 0.0) or 0.0)
            liq_ok      = bool(getattr(sig, "liq_ok", True))
            vol_mult    = float(getattr(sig, "vol_mult", 0.0) or 0.0)
            don_len     = getattr(sig, "don_break_len", None)
            don_level   = getattr(sig, "don_break_level", None)
            don_dist    = getattr(sig, "don_dist_atr", None)
            eth_hist    = getattr(sig, "eth_macd_hist_4h", None)
            eth_above   = getattr(sig, "eth_macd_above_signal", None)
            reg_up      = bool(getattr(sig, "regime_up_flag", False))
            pullback    = getattr(sig, "pullback_type", "?")
            entry_rule  = getattr(sig, "entry_rule", "?")

            # safe strings
            don_len_s   = str(don_len) if isinstance(don_len, (int, float)) else "?"
            don_level_s = f"{don_level:.6f}" if isinstance(don_level, (int, float)) else "N/A"
            don_dist_s  = f"{don_dist:+.2f}" if isinstance(don_dist, (int, float)) else "N/A"
            eth_hist_s  = f"{eth_hist:.3f}" if isinstance(eth_hist, (int, float)) else "N/A"
            vwap_frac   = getattr(sig, "vwap_stack_frac", None)
            vwap_exp    = getattr(sig, "vwap_stack_expansion_pct", None)
            vwap_slope  = getattr(sig, "vwap_stack_slope_pph", None)
            vwap_frac_s  = f"{vwap_frac:.2f}" if isinstance(vwap_frac, (int, float)) else "N/A"
            vwap_exp_s   = f"{(vwap_exp*100):.2f}%" if isinstance(vwap_exp, (int, float)) else "N/A"
            vwap_slope_s = f"{vwap_slope:.4f}" if isinstance(vwap_slope, (int, float)) else "N/A"

            # gates (mirror the scan diagnostic)
            def _ok(b): return "✅" if b else "❌"
            g_rs    = (rs_pct >= rs_min)
            g_liq   = liq_ok
            g_reg   = (reg_up or (not regime_block))
            g_vol   = (vol_mult >= vol_needed)
            g_micro = ((float(sig.atr_pct)/100.0) >= min_atr_pct)
            g_meta  = (wp >= meta_thresh)

            # TP preview (even if partials enabled, show final)
            final_tp_mult  = float(self.cfg.get("FINAL_TP_ATR_MULT", self.cfg.get("TP_ATR_MULT", 8.0)))
            tp_final_price = actual_entry_price + (final_tp_mult * float(sig.atr) * (+1 if is_long else -1))

            msg = (
                f"🔔 OPENED {side_label} {sig.symbol} — {ts} UTC\n"
                f"Price: {actual_entry_price:.6f} | ATR1h: {sig.atr:.6f} ({sig.atr_pct:.2f}%) | RSI1h: {sig.rsi:.1f} | ADX1h: {sig.adx:.1f}\n"
                f"Donch({don_len_s}d): level={don_level_s}  dist_atr={don_dist_s}\n"
                f"Volume mult: x{vol_mult:.2f} | RS: {rs_pct:.1f}% | Regime: {sig.market_regime}\n"
                f"ETH(4h) MACD hist: {eth_hist_s}  MACD>signal: {bool(eth_above)}  → regime_up={reg_up}\n"
                f"VWAP stack: frac={vwap_frac_s}  exp={vwap_exp_s}  slope_pph={vwap_slope_s}\n"
                f"Entry Rule: {pullback} + {entry_rule}\n"
                f"GATES: RS≥{rs_min:.0f} {_ok(g_rs)} | Liquidity {_ok(g_liq)} | RegimeUp/OK {_ok(g_reg)} | "
                f"Vol x{vol_needed:.1f} {_ok(g_vol)} | MicroATR≥{min_atr_pct:.4f} {_ok(g_micro)} | META p={wp:.3f}≥{meta_thresh:.2f} {_ok(g_meta)}\n"
                f"Sizing: base ${base_risk_usd:.2f} · ETH×{eth_mult:.2f} · VWAP×{vw_mult:.2f} · WP×{wp_mult:.2f} · YAML×{yaml_mult:.2f} ⇒ risk ${risk_usd:.2f}\n"
                f"Protection: SL {sl_mult:.2f}×ATR → {stop_price:.6f} | Final TP {final_tp_mult:.2f}×ATR → {tp_final_price:.6f}"
            )
            await self.tg.send(msg)

        except Exception as e:
            # EMERGENCY CLOSE to prevent naked position
            msg = f"🚨 CRITICAL: Failed to persist/protect {sig.symbol}: {e}. Emergency closing now."
            LOG.critical(msg)
            await self.tg.send(msg)
            try:
                await self.exchange.create_market_order(
                    sig.symbol, sl_close_side, actual_size, params={"reduceOnly": True, "category": "linear"}
                )
                await self.tg.send(f"✅ Emergency close filled for {sig.symbol}.")
            except Exception as close_e:
                await self.tg.send(f"🚨 FAILED EMERGENCY CLOSE for {sig.symbol}: {close_e}")

    async def _manage_positions_loop(self):
        while True:
            if not self.open_positions:
                await asyncio.sleep(2)
                continue
            for pid, pos in list(self.open_positions.items()):
                try:
                    await self._update_single_position(pid, pos)
                except Exception as e:
                    LOG.error("manage err %s %s", pos["symbol"], e)
            await asyncio.sleep(5)

    async def _update_single_position(self, pid: int, pos: Dict[str, Any]):
        symbol = pos["symbol"]

        # Safety net: check true position size
        try:
            positions = await self.exchange.fetch_positions(symbols=[symbol])
            position_size = 0.0
            if positions and positions[0]:
                position_size = float(positions[0].get('info', {}).get('size', 0))

                
            if position_size == 0:
                # 1) Sweep & cancel stale reduce-only/children (throttled)
                if not hasattr(self, "_cancel_cleanup_backoff"):
                    self._cancel_cleanup_backoff = {}
                _now = datetime.now(timezone.utc)
                _last_cancel = self._cancel_cleanup_backoff.get(symbol)
                if (not _last_cancel) or (_now - _last_cancel).total_seconds() >= float(self.cfg.get("CANCEL_CLEANUP_BACKOFF_SEC", 180)):
                    try:
                        await self._cancel_reducing_orders(symbol, pos)
                    finally:
                        self._cancel_cleanup_backoff[symbol] = _now

                # 2) Debounce finalize to avoid log spam during transient API hiccups
                if not hasattr(self, "_zero_finalize_backoff"):
                    self._zero_finalize_backoff = {}
                _last_fin = self._zero_finalize_backoff.get(pid)
                if _last_fin and (_now - _last_fin).total_seconds() < float(self.cfg.get("FINALIZE_BACKOFF_SEC", 120)):
                    LOG.info("Position size is 0 for %s; finalize debounced.", symbol)
                    return
                self._zero_finalize_backoff[pid] = _now

                LOG.info("Position size for %s is 0. Inferring exit reason and finalizing…", symbol)

                # 3) Infer exit type conservatively; MANUAL_CLOSE by default
                inferred_reason = "MANUAL_CLOSE"
                try:
                    open_orders = await self._all_open_orders(symbol)
                    open_cids = {o.get("clientOrderId") for o in (open_orders or [])}
                    if any([
                        (pos.get("tp_final_cid") and pos["tp_final_cid"] not in open_cids),
                        (pos.get("tp2_cid")     and pos["tp2_cid"]     not in open_cids),
                        (pos.get("tp1_cid")     and pos["tp1_cid"]     not in open_cids),
                    ]):
                        inferred_reason = "TP"
                    elif any([
                        (pos.get("sl_trail_cid") and pos["sl_trail_cid"] not in open_cids),
                        (pos.get("sl_cid")       and pos["sl_cid"]       not in open_cids),
                    ]):
                        inferred_reason = "SL"
                except Exception as e:
                    LOG.warning("Open-orders probe failed while finalizing %s: %s", symbol, e)

                try:
                    await self._finalize_position(pid, pos, inferred_exit_reason=inferred_reason)
                except Exception as e:
                    LOG.exception("Finalize failed for %s pid=%s: %s", symbol, pid, e)
                return
        except Exception as e:
            LOG.error("Could not fetch position size for %s during update: %s", symbol, e)
            return



        orders = await self._all_open_orders(symbol)
        open_cids = {o.get("clientOrderId") for o in orders}

        if self.cfg.get("TIME_EXIT_ENABLED", cfg.TIME_EXIT_ENABLED):
            ddl = pos.get("exit_deadline")
            if ddl and datetime.now(timezone.utc) >= ddl:
                LOG.info("Time-exit firing on %s (pid %d)", symbol, pid)
                await self._force_close_position(pid, pos, tag="TIME_EXIT")
                return

        if self.cfg.get("PARTIAL_TP_ENABLED", False) and not pos["trailing_active"] and pos.get("tp1_cid") not in open_cids:
            fill_price = None
            filled_qty = 0.0
            try:
                o = await self._fetch_by_cid(pos["tp1_cid"], symbol)
                if o and str(o.get('status','')).lower() == 'closed':
                    fill_price = o.get('average') or o.get('price')
                    filled_qty = float(o.get('filled') or o.get('amount') or 0.0)
            except Exception as e:
                LOG.warning("Failed to fetch TP1 order %s: %s", pos["tp1_cid"], e)

            # **Only** record a TP1 fill if the order actually closed with >0 qty.
            if (filled_qty or 0.0) > 0.0 and (fill_price is not None):
                await self.db.add_fill(
                    pid, "TP1", float(fill_price), float(filled_qty), datetime.now(timezone.utc)
                )
                await self.db.update_position(pid, trailing_active=True)
                pos["trailing_active"] = True
                await self._activate_trailing(pid, pos)
                await self.tg.send(f"📈 TP1 hit on {symbol}, trailing activated")
            else:
                LOG.info("TP1 for %s not filled (status!=closed or filled=0). Skipping phantom fill.", symbol)


            await self.db.add_fill(
                pid, "TP1", fill_price, float(pos["size"]) * self.cfg["PARTIAL_TP_PCT"], datetime.now(timezone.utc)
            )
            await self.db.update_position(pid, trailing_active=True)
            pos["trailing_active"] = True
            await self._activate_trailing(pid, pos)
            await self.tg.send(f"📈 TP1 hit on {symbol}, trailing activated")

        if pos["trailing_active"]:
            await self._trail_stop(pid, pos)

        active_stop_cid = pos.get("sl_trail_cid") if pos["trailing_active"] else pos.get("sl_cid")
        is_closed = active_stop_cid not in open_cids
        if not is_closed and pos["trailing_active"] and self.cfg.get("FINAL_TP_ENABLED", False):
            if pos.get("tp2_cid") not in open_cids:
                is_closed = True

        if is_closed:
            await self._finalize_position(pid, pos)

    async def _activate_trailing(self, pid: int, pos: Dict[str, Any]):
        symbol = pos["symbol"]
        try:
            if pos.get("sl_cid"):
                await self._cancel_by_cid(pos["sl_cid"], symbol)
        except ccxt.OrderNotFound:
            pass
        except Exception as e:
            LOG.warning("Could not cancel original SL for %d (%s): %s", pid, pos.get('sl_cid'), e)

        await self._trail_stop(pid, pos, first=True)

        if self.cfg.get("FINAL_TP_ENABLED", False):
            try:
                is_long = (pos.get("side","SHORT").upper() == "LONG")
                tp_dir = (+1 if is_long else -1)
                final_tp_price = float(pos["entry_price"]) + tp_dir * tp_mult * float(pos["atr"])

                qty_left = float(pos["size"]) * (1 - self.cfg["PARTIAL_TP_PCT"])
                tp2_cid = create_stable_cid(pid, "TP2")

                tp2_side = "sell" if is_long else "buy"
                tp2_trigger_dir = 1 if is_long else 2  # LONG: rise to target; SHORT: fall to target

                await self.exchange.create_order(
                    symbol, "market", tp2_side, qty_left, None,
                    params={
                        "triggerPrice": float(final_tp_price),
                        "clientOrderId": tp2_cid,
                        'reduceOnly': True, 'closeOnTrigger': True,
                        'triggerDirection': tp2_trigger_dir, 'category': 'linear'
                    }
                )
                await self.db.update_position(pid, tp2_cid=tp2_cid)
                pos["tp2_cid"] = tp2_cid
                LOG.info("Final TP2 placed for %s with CID %s", symbol, tp2_cid)
            except Exception as e:
                LOG.error("Failed to place TP2 for %d: %s", pid, e)

    async def _trail_stop(self, pid: int, pos: Dict[str, Any], first: bool = False):
        """
        Trailing stop for BOTH sides:
        LONG : new_stop = price - k*ATR  (below price), move up only
        SHORT: new_stop = price + k*ATR  (above price), move down only
        """
        symbol = pos["symbol"]
        is_long = (pos.get("side", "SHORT").upper() == "LONG")
        price = float((await self.exchange.fetch_ticker(symbol))["last"])
        atr = float(pos["atr"])
        prev_stop = float(pos.get("stop_price", 0) or 0.0)

        k = float(self.cfg.get("TRAIL_DISTANCE_ATR_MULT", 1.0))
        min_move = price * float(self.cfg.get("TRAIL_MIN_MOVE_PCT", 0.001))

        if is_long:
            new_stop = price - k * atr
            favorable = (prev_stop == 0.0) or (new_stop > prev_stop)   # raise stop only
            close_side = "sell"
            trigger_dir = 2  # falling touches stop for LONG
            qty_left = float(pos["size"]) * (1 - float(self.cfg.get("PARTIAL_TP_PCT", 0.7)))
        else:
            new_stop = price + k * atr
            favorable = (prev_stop == 0.0) or (new_stop < prev_stop)   # lower stop only
            close_side = "buy"
            trigger_dir = 1  # rising touches stop for SHORT
            qty_left = float(pos["size"]) * (1 - float(self.cfg.get("PARTIAL_TP_PCT", 0.7)))

        significant = abs(prev_stop - new_stop) > min_move
        if not first and not (favorable and significant):
            return

        sl_trail_cid = create_stable_cid(pid, "SL_TRAIL")

        # Cancel previous trailing SL if any
        try:
            if (not first) and pos.get("sl_trail_cid"):
                await self._cancel_by_cid(pos["sl_trail_cid"], symbol)
        except ccxt.OrderNotFound:
            pass
        except Exception as e:
            LOG.warning("Trail cancel failed for %s: %s", symbol, e)
            return

        # Place new conditional stop
        await self.exchange.create_order(
            symbol, 'market', close_side, qty_left, None,
            params={
                "triggerPrice": float(new_stop),
                "clientOrderId": sl_trail_cid, "category": "linear",
                "reduceOnly": True, "closeOnTrigger": True, "triggerDirection": trigger_dir
            }
        )
        await self.db.update_position(pid, stop_price=float(new_stop), sl_trail_cid=sl_trail_cid)
        pos["stop_price"] = float(new_stop)
        pos["sl_trail_cid"] = sl_trail_cid
        LOG.info("Trail updated %s to %.6f", symbol, new_stop)


    async def _finalize_position(self, pid: int, pos: Dict[str, Any], inferred_exit_reason: str = None):
        symbol = pos["symbol"]
        opened_at = pos["opened_at"]
        entry_price = float(pos["entry_price"])
        size = float(pos["size"])
        side = (pos.get("side") or "SHORT").lower()  # "long" or "short"

        closed_at = datetime.now(timezone.utc)
        exit_price, exit_qty = None, 0.0
        closing_order_type = inferred_exit_reason or "UNKNOWN"

        # 1) Try to tie to a specific closing order (TP/SL/Trail)
        closing_order_cid = None
        if inferred_exit_reason == "TP":
            closing_order_cid = pos.get("tp_final_cid") or pos.get("tp2_cid") or pos.get("tp1_cid")
        elif inferred_exit_reason == "SL":
            closing_order_cid = pos.get("sl_trail_cid") or pos.get("sl_cid")

        if closing_order_cid:
            try:
                order = await self._fetch_by_cid(closing_order_cid, symbol)
                if order and (order.get("average") or order.get("price")):
                    exit_price = float(order.get("average") or order.get("price"))
                    exit_qty = float(order.get("filled") or 0) or 0.0
                    closing_order_type = inferred_exit_reason
            except Exception as e:
                LOG.warning("Fetch by CID %s for %s failed (fallback to trades): %s", closing_order_cid, symbol, e)

        # 2) Fallback to recent trades if we couldn't bind to an order
        if not exit_price:
            try:
                await asyncio.sleep(1.5)
                my_trades = await self.exchange.fetch_my_trades(symbol, limit=10)
                close_side = "sell" if side == "long" else "buy"
                closing_trade = next((t for t in reversed(my_trades) if str(t.get("side", "")).lower() == close_side), None)
                if closing_trade:
                    exit_price = float(closing_trade["price"])
                    exit_qty = float(closing_trade.get("amount") or 0)
                    if closing_order_type == "UNKNOWN":
                        closing_order_type = "FALLBACK_FILL"
                else:
                    LOG.error("No closing trade found for %s; fallback to entry price.", symbol)
                    exit_price, exit_qty = entry_price, size
            except Exception as e:
                LOG.error("fetch_my_trades fallback failed for %s: %s. Using entry price.", symbol, e)
                exit_price, exit_qty = entry_price, size

        # 3) Persist this last observed fill (won’t double-count; we aggregate below)
        try:
            await self.db.add_fill(pid, closing_order_type, float(exit_price), float(exit_qty), closed_at)
        except Exception as e:
            LOG.warning("Could not persist closing fill for %s: %s", symbol, e)

        # 4) Aggregate PnL from ALL fills (handles TP1/TP2/trailing)
        rows = await self.db.pool.fetch(
            "SELECT fill_type AS kind, price, qty FROM fills WHERE position_id=$1 ORDER BY ts ASC",
            pid
        )

        entry_notional = exit_notional = entry_qty = exit_qty_sum = 0.0
        for r in rows:
            px = float(r["price"] or 0.0)
            q  = float(r["qty"] or 0.0)
            k  = (r["kind"] or "").upper()
            is_exit = (k.startswith("TP") or k.startswith("SL") or "TIME_EXIT" in k or "FALLBACK" in k)
            if is_exit:
                if px > 0 and q > 0:
                    exit_notional += px * q
                    exit_qty_sum  += q
            else:
                if px > 0 and q > 0:
                    entry_notional += px * q
                    entry_qty      += q

        # If we never stored an entry fill, synthesize it from entry snapshot
        if entry_qty <= 0.0:
            entry_notional = entry_price * size
            entry_qty      = size

        # ---- Sanity: if fills are clearly inconsistent, fall back to safe formula
        def _pnl_linear(e_px, x_px, q, sd):
            return (e_px - x_px) * q if sd == "short" else (x_px - e_px) * q

        bad_fills = (exit_qty_sum <= 0.0) or (abs(exit_qty_sum - entry_qty) > 0.05 * entry_qty)
        if bad_fills:
            LOG.warning("PnL fallback on %s: exit_qty %.6f vs entry_qty %.6f. Using direct formula.",
                        symbol, exit_qty_sum, entry_qty)
            # Use the best exit price we have (from a bound order, a trade, or ticker)
            safe_exit_px = float(exit_price or entry_price)
            total_pnl = _pnl_linear(entry_price, safe_exit_px, float(size), side)
            avg_exit = safe_exit_px
            pnl_pct = ((entry_price / avg_exit - 1.0) * 100.0) if side == "short" else ((avg_exit / entry_price - 1.0) * 100.0)
        else:
            if side == "short":
                total_pnl = entry_notional - exit_notional
                avg_exit = (exit_notional / exit_qty_sum) if exit_qty_sum > 0 else entry_price
                pnl_pct = (entry_price / avg_exit - 1.0) * 100.0 if avg_exit > 0 else 0.0
            else:
                total_pnl = exit_notional - entry_notional
                avg_exit = (exit_notional / exit_qty_sum) if exit_qty_sum > 0 else entry_price
                pnl_pct = (avg_exit / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0

        holding_minutes = (closed_at - opened_at).total_seconds() / 60 if opened_at else 0.0

        # 5) Post-trade analytics + fees (sum fees from recent trades in window if available)
        mae_usd = mfe_usd = mae_over_atr = mfe_over_atr = 0.0
        realized_vol_during_trade = btc_beta_during_trade = 0.0
        fees_paid = 0.0
        try:
            since_ts = int((opened_at - timedelta(minutes=5)).timestamp() * 1000)
            until_ts = int((closed_at + timedelta(minutes=5)).timestamp() * 1000)
            my_trades = await self.exchange.fetch_my_trades(symbol, limit=200, since=since_ts)
            for t in my_trades:
                ts = int(t.get("timestamp") or 0)
                if ts and since_ts <= ts <= until_ts:
                    f = t.get("fee") or {}
                    fees_paid += float(f.get("cost") or 0.0)
        except Exception as e:
            LOG.warning("Fee aggregation skipped for %s: %s", symbol, e)

        await self.db.update_position(
            pid,
            status="CLOSED",
            closed_at=closed_at,
            exit_reason=closing_order_type,
            pnl=total_pnl,
            pnl_pct=pnl_pct,
            holding_minutes=holding_minutes,
            avg_exit_price=avg_exit if 'avg_exit_price' in getattr(self.db, 'columns_positions', set()) else None,
            fees_paid=fees_paid
        )

        await self.risk.on_trade_close(total_pnl, self.tg)
        self.open_positions.pop(pid, None)
        self.last_exit[symbol] = closed_at
        await self.tg.send(f"✅ {symbol} position closed. Total PnL ≈ {total_pnl:.2f} USDT")



    async def _force_open_position(self, symbol: str):
        """
        Manually triggers a trade for testing. Still respects veto filters.
        """
        await self.tg.send(f"Force-open requested for {symbol}. Building signal…")
        LOG.info("Manual trade requested for %s", symbol)

        if any(p['symbol'] == symbol for p in self.open_positions.values()):
            msg = f"⚠️ Cannot force open {symbol}: position already open."
            LOG.warning(msg); await self.tg.send(msg); return

        try:
            current_market_regime = await self.regime_detector.get_current_regime()
            eth_macd_data = None
            try:
                eth_ohlcv = await self.exchange.fetch_ohlcv('ETHUSDT', '4h', limit=100)
                if eth_ohlcv:
                    df_eth = pd.DataFrame(eth_ohlcv, columns=['timestamp','open','high','low','close','volume'])
                    macd_df = ta.macd(df_eth['close'])
                    latest = macd_df.iloc[-1]
                    eth_macd_data = {"macd": latest['macd'], "signal": latest['signal'], "hist": latest['hist']}
            except Exception as e:
                LOG.warning("ETH MACD barometer failed: %s", e)

            signal = await self._scan_symbol_for_signal(symbol, current_market_regime, eth_macd_data)
            if signal is None:
                msg = f"❌ Force open {symbol}: no signal."
                LOG.error(msg); await self.tg.send(msg); return

            # Optional meta gate for forced opens too (set FORCE_BYPASS_META=true to ignore)
            if not bool(self.cfg.get("FORCE_BYPASS_META", False)):
                pstar = self.cfg.get("META_PROB_THRESHOLD", None)
                if pstar is not None and float(signal.win_probability) < float(pstar):
                    await self.tg.send(f"⚠️ Gated by meta: p={signal.win_probability:.3f} < p*={float(pstar):.3f}")
                    return

            equity = await self.db.latest_equity() or 0.0
            open_positions_count = len(self.open_positions)
            ok, vetoes = filters.evaluate(
                signal, listing_age_days=signal.listing_age_days,
                open_positions=open_positions_count, equity=equity
            )
            if not ok:
                await self.tg.send(f"⚠️ VETOED {symbol}: {' | '.join(vetoes)}")
                return

            await self.tg.send(f"✅ Proceeding with forced entry for {symbol}.")
            await self._open_position(signal)

        except Exception as e:
            msg = f"❌ Force open error for {symbol}: {e}"
            LOG.error(msg, exc_info=True); await self.tg.send(msg)

    async def _force_close_position(self, pid: int, pos: Dict[str, Any], tag: str):
        symbol = pos["symbol"]
        size = float(pos["size"])
        entry_price = float(pos["entry_price"])
        side = (pos.get("side") or "SHORT").upper()

        try:
            await self.exchange.cancel_all_orders(symbol, params={"category": "linear"})
            close_side = "sell" if side == "LONG" else "buy"
            await self.exchange.create_market_order(
                symbol, close_side, size, params={"reduceOnly": True, "category": "linear"}
            )
            LOG.info("Force-closed %s (pid %d) due to: %s", symbol, pid, tag)
        except Exception as e:
            LOG.warning("Force-close issue on %s: %s", symbol, e)

        await asyncio.sleep(2)

        exit_price = None
        try:
            my_trades = await self.exchange.fetch_my_trades(symbol, limit=10)
            closing_trade = next(
                (t for t in reversed(my_trades) if str(t.get("side","")).lower() == ("sell" if side == "LONG" else "buy")),
                None
            )
            if closing_trade:
                exit_price = float(closing_trade["price"])
                LOG.info("Confirmed force-close fill for %s at %.8f", symbol, exit_price)
        except Exception as e:
            LOG.error("Error fetching force-close fill for %s: %s", symbol, e)

        if not exit_price:
            exit_price = float((await self.exchange.fetch_ticker(symbol))["last"])

        pnl = (entry_price - exit_price) * size if side == "SHORT" else (exit_price - entry_price) * size
        closed_at = datetime.now(timezone.utc)
        holding_minutes = (closed_at - pos.get("opened_at", closed_at)).total_seconds()/60.0

        await self.db.update_position(
            pid, status="CLOSED", closed_at=closed_at, pnl=pnl,
            exit_reason=tag, holding_minutes=holding_minutes
        )
        await self.db.add_fill(pid, tag, exit_price, size, closed_at)
        await self.risk.on_trade_close(pnl, self.tg)
        self.last_exit[symbol] = closed_at
        self.open_positions.pop(pid, None)
        await self.tg.send(f"⏰ {symbol} closed by {tag}. PnL ≈ {pnl:.2f} USDT")







    # ───────────────────── Loops & commands ─────────────────────

    async def _main_signal_loop(self):
        LOG.info("Starting main signal scan loop.")
        while True:
            try:
                if self.paused or not self.risk.can_trade():
                    await asyncio.sleep(5)
                    continue

                current_market_regime = await self.regime_detector.get_current_regime()
                LOG.info("New scan cycle for %d symbols | regime: %s", len(self.symbols), current_market_regime)

                uc = self._load_universe_cache_if_fresh()
                if uc is not None:
                    self._universe_ctx = uc
                    LOG.info("Universe context loaded from cache (%d symbols).", len(self._universe_ctx))
                else:
                    try:
                        self._universe_ctx = await self._build_universe_context()
                        LOG.info("Universe context ready (%d symbols).", len(self._universe_ctx))
                        self._save_universe_cache(self._universe_ctx)
                    except Exception as e:
                        LOG.warning("Universe context failed: %s (continuing with empty).", e)
                        self._universe_ctx = {}

                # ETH MACD Barometer (4h)
                eth_macd_data = None
                try:
                    eth_ohlcv = await self.exchange.fetch_ohlcv('ETHUSDT', '4h', limit=100)
                    if eth_ohlcv:
                        df_eth = pd.DataFrame(eth_ohlcv, columns=['timestamp','open','high','low','close','volume'])
                        macd_df = ta.macd(df_eth['close'])
                        latest_macd = macd_df.iloc[-1]
                        eth_macd_data = {"macd": latest_macd['macd'], "signal": latest_macd['signal'], "hist": latest_macd['hist']}
                        LOG.info("ETH MACD(4h): macd=%.2f, hist=%.2f", latest_macd['macd'], latest_macd['hist'])
                except Exception as e:
                    LOG.warning("ETH MACD barometer failed: %s", e)

                equity = await self.db.latest_equity() or 0.0
                open_positions_count = len(self.open_positions)
                open_symbols = {p['symbol'] for p in self.open_positions.values()}

                for sym in self.symbols:
                    if self.paused or not self.risk.can_trade():
                        break
                    if sym in open_symbols:
                        continue

                    # Cooldown fast-skip
                    cd_h = self.cfg.get("SYMBOL_COOLDOWN_HOURS", cfg.SYMBOL_COOLDOWN_HOURS)
                    last_x = self.last_exit.get(sym)
                    if last_x and datetime.now(timezone.utc) - last_x < timedelta(hours=cd_h):
                        continue

                    async with self.symbol_locks[sym]:
                        # Pre-flight: skip if exchange already has a position
                        try:
                            positions = await self.exchange.fetch_positions(symbols=[sym])
                            if positions and positions[0] and float(positions[0].get('info', {}).get('size', 0)) > 0:
                                LOG.info("Skipping scan for %s, pre-flight position exists.", sym)
                                if sym not in open_symbols:
                                    LOG.warning("ORPHAN POSITION DETECTED for %s! Reconcile later.", sym)
                                continue
                        except Exception as e:
                            LOG.error("Pre-flight position check failed for %s: %s", sym, e)
                            continue

                        # Build signal
                        signal = await self._scan_symbol_for_signal(sym, current_market_regime, eth_macd_data, gov_ctx=None)
                        if signal:
                            # Optional meta probability gate
                            pstar = self.cfg.get("META_PROB_THRESHOLD", None)
                            if pstar is not None and float(signal.win_probability) < float(pstar):
                                LOG.info("Signal %s gated by meta p=%.3f < p*=%.3f",
                                         sym, float(signal.win_probability), float(pstar))
                                await asyncio.sleep(0.2)
                                continue

                            # Old-school filters (belt & suspenders)
                            ok, vetoes = filters.evaluate(
                                signal,
                                listing_age_days=signal.listing_age_days,
                                open_positions=open_positions_count,
                                equity=equity,
                            )

                            if ok:
                                await self._open_position(signal)
                                open_positions_count += 1
                                open_symbols.add(sym)
                            else:
                                LOG.info("Signal for %s vetoed: %s", sym, " | ".join(vetoes))

                    await asyncio.sleep(0.5)

            except Exception as e:
                LOG.error("Critical error in main signal loop: %s", e)
                traceback.print_exc()

            LOG.info("Scan cycle complete. Sleeping…")
            await asyncio.sleep(self.cfg.get("SCAN_INTERVAL_SEC", 60))

    async def _equity_loop(self):
        while True:
            try:
                bal = await self._fetch_platform_balance()
                current_equity = bal["total"].get("USDT", 0.0)
                await self.db.snapshot_equity(current_equity, datetime.now(timezone.utc))

                if current_equity > self.peak_equity:
                    self.peak_equity = current_equity

                if self.cfg.get("DD_PAUSE_ENABLED", True) and not self.risk.kill_switch:
                    if self.peak_equity and current_equity < self.peak_equity:
                        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity * 100
                        max_dd_pct = self.cfg.get("DD_MAX_PCT", 10.0)
                        if drawdown_pct >= max_dd_pct:
                            self.risk.kill_switch = True
                            msg = f"❌ KILL-SWITCH: Equity drawdown {drawdown_pct:.2f}% ≥ {max_dd_pct}%."
                            LOG.warning(msg)
                            await self.tg.send(msg)
            except Exception as e:
                LOG.error("Error in equity loop: %s", e)
            await asyncio.sleep(3600)

    # ───────────────────── NEW: pid lookup + repair helpers ───────────────────
    async def _finalize_zero_position_safe(self, pid: int, pos: dict, symbol: str):
        """
        Finalize a position whose exchange size is zero.
        Try to infer exit from our SL/TP cids; else mark MANUAL_CLOSE and compute PnL via fallback.
        Debounced via self._zero_finalize_backoff.
        """
        now = datetime.now(timezone.utc)
        last_try = self._zero_finalize_backoff.get(pid)
        if last_try and (now - last_try).total_seconds() < float(self.cfg.get("FINALIZE_BACKOFF_SEC", 120)):
            return  # too soon; skip noisy retries
        self._zero_finalize_backoff[pid] = now

        # 1) try to detect which order closed it
        exit_price = None
        exit_kind  = "MANUAL_CLOSE"
        for label, cid in (("TP", pos.get("tp_final_cid")),
                           ("TP1", pos.get("tp1_cid")),
                           ("SL", pos.get("sl_cid"))):
            o = await self._fetch_by_cid(cid, symbol, silent=True)
            if o and str(o.get("status","")).lower() == "closed":
                exit_kind  = label
                exit_price = o.get("average") or o.get("price")
                try:
                    exit_price = float(exit_price) if exit_price is not None else None
                except Exception:
                    exit_price = None
                break

        # 2) if still unknown, try trades around now; else use ticker last
        if exit_price is None:
            try:
                since_ts = int((now - timedelta(minutes=15)).timestamp() * 1000)
                trades = await self.exchange.fetch_my_trades(symbol, limit=100, since=since_ts)
                # pick the most recent trade with opposite side to the position
                side = str(pos.get("side","")).lower()
                close_side = "sell" if side == "long" else "buy"
                for t in reversed(trades or []):
                    if str(t.get("side","")).lower() == close_side:
                        exit_price = float(t.get("price"))
                        break
            except Exception:
                pass
        if exit_price is None:
            try:
                tk = await self.exchange.fetch_ticker(symbol)
                exit_price = float(tk.get("last") or tk.get("mark") or tk.get("index") or pos.get("entry_price"))
            except Exception:
                exit_price = float(pos.get("entry_price"))

        # 3) hand off to your finalize routine (uses the safe PnL fallback you added)
        await self._finalize_position(
            pid,
            closing_order_type=exit_kind,
            exit_price=exit_price,
            closed_at=now,
        )


    async def _find_pid_by_symbol(self, symbol: str, status: str = "CLOSED") -> Optional[int]:
        """Return most recent position id for a symbol with given status."""
        row = await self.db.pool.fetchrow(
            "SELECT id FROM positions WHERE symbol=$1 AND status=$2 ORDER BY COALESCE(closed_at, opened_at) DESC LIMIT 1",
            symbol.upper(), status.upper()
        )
        return int(row["id"]) if row else None

    async def _recent_positions_text(self, symbol: Optional[str] = None, limit: int = 10) -> str:
        """Human-readable recent positions list for TG."""
        if symbol:
            q = """
            SELECT id, symbol, status, side, size, entry_price, closed_at, pnl
            FROM positions WHERE symbol=$1 ORDER BY id DESC LIMIT $2
            """
            rows = await self.db.pool.fetch(q, symbol.upper(), limit)
        else:
            q = """
            SELECT id, symbol, status, side, size, entry_price, closed_at, pnl
            FROM positions ORDER BY id DESC LIMIT $1
            """
            rows = await self.db.pool.fetch(q, limit)
        if not rows:
            return "No positions."
        lines = []
        for r in rows:
            lines.append(f"pid={r['id']}  {r['symbol']}  {r['status']}  "
                         f"{r['side']}  sz={float(r['size']):.4g}  "
                         f"EP={float(r['entry_price']):.6f}  "
                         f"closed={str(r['closed_at'])[:19]}  "
                         f"PnL={float(r['pnl'] or 0):.4f}")
        return "Recent positions:\n" + "\n".join(lines)

    async def _repair_closed_position(self, pid: int):
        """Recompute PnL for a closed position using fallback logic and update DB."""
        pos = await self.db.pool.fetchrow("SELECT * FROM positions WHERE id=$1", pid)
        if not pos or str(pos["status"]).upper() != "CLOSED":
            await self.tg.send(f"Repair: position {pid} not found or not closed."); return

        side = (pos["side"] or "SHORT").lower()
        size = float(pos["size"]); entry_price = float(pos["entry_price"])
        exit_px = None
        try:
            symbol = pos["symbol"]; closed_at = pos["closed_at"]
            since_ts = int((closed_at - timedelta(minutes=10)).timestamp() * 1000)
            my_trades = await self.exchange.fetch_my_trades(symbol, limit=50, since=since_ts)
            close_side = "sell" if side == "long" else "buy"
            t = next((t for t in reversed(my_trades) if str(t.get("side","")).lower()==close_side), None)
            if t: exit_px = float(t["price"])
        except Exception:
            pass
        if exit_px is None:
            ticker = await self.exchange.fetch_ticker(pos["symbol"])
            exit_px = float(ticker.get("last") or entry_price)

        new_pnl = (entry_price - exit_px)*size if side=="short" else (exit_px-entry_price)*size
        new_pct = (entry_price/exit_px-1.0)*100.0 if side=="short" else (exit_px/entry_price-1.0)*100.0
        await self.db.update_position(pid, pnl=new_pnl, pnl_pct=new_pct)
        await self.tg.send(f"🔧 Repaired PnL for {pos['symbol']} pid={pid}: {new_pnl:.6f} USDT")



    async def _fetch_platform_balance(self) -> dict:
        account_type = self.cfg.get("BYBIT_ACCOUNT_TYPE", "STANDARD").upper()
        params = {}
        if account_type == "UNIFIED":
            params['accountType'] = 'UNIFIED'
        try:
            return await self.exchange.fetch_balance(params=params)
        except Exception as e:
            LOG.error("Failed to fetch %s account balance: %s", account_type, e)
            return {"total": {"USDT": 0.0}, "free": {"USDT": 0.0}}

    async def _telegram_loop(self):
        while True:
            async for cmd in self.tg.poll_cmds():
                await self._handle_cmd(cmd)
            await asyncio.sleep(1)

    async def _handle_cmd(self, cmd: str):
        parts = cmd.split()
        root = parts[0].lower()
        if root == "/pause":
            self.paused = True
            await self.tg.send("⏸ Paused")

        elif root == "/report" and len(parts) == 2:
            period = parts[1].lower()
            if period in ['6h', 'daily', 'weekly', 'monthly']:
                await self.tg.send(f"Generating on-demand '{period}' report…")
                summary_text = await self._generate_summary_report(period)
                await self.tg.send(summary_text)
            else:
                await self.tg.send("Unknown period. Use: 6h, daily, weekly, monthly.")

        elif root == "/resume":
            if self.risk.can_trade():
                self.paused = False
                await self.tg.send("▶️ Resumed")
            else:
                await self.tg.send("⚠️ Kill switch active")

        elif root == "/set" and len(parts) == 3:
            raw_key, val = parts[1], parts[2]
            key = self._normalize_cfg_key(raw_key)
            try:
                cast = json.loads(val)
            except json.JSONDecodeError:
                cast = val

            self.cfg[key] = cast
            setattr(cfg, key, cast)
            await self.tg.send(f"✅ {raw_key} → {key} set to {cast}")
            return

        elif root == "/open" and len(parts) == 2:
            symbol = parts[1].upper()
            asyncio.create_task(self._force_open_position(symbol))
            return

        elif root == "/status":
            await self.tg.send(json.dumps({
                "paused": self.paused,
                "open": len(self.open_positions),
                "loss_streak": self.risk.loss_streak,
            }, indent=2))

        elif root == "/analyze":
            await self.tg.send("🤖 Starting analysis… needs `bybit.csv`. Will send the report here.")
            try:
                subprocess.Popen(["/opt/livefader/src/run_weekly_report.sh"])
            except Exception as e:
                await self.tg.send(f"❌ Failed to start analysis: {e}")
            return
        elif root == "/recent":
            # /recent                → last 10 of all symbols
            # /recent XCNUSDT 5      → last 5 for XCNUSDT
            sym = parts[1].upper() if len(parts) >= 2 and parts[1][0].isalpha() else None
            lim = int(parts[-1]) if parts and parts[-1].isdigit() else 10
            txt = await self._recent_positions_text(sym, lim)
            await self.tg.send(txt)

        elif root == "/pid" and len(parts) >= 2:
            # /pid XCNUSDT            → most recent pid for symbol (any status)
            sym = parts[1].upper()
            row = await self.db.pool.fetchrow(
                "SELECT id,status FROM positions WHERE symbol=$1 ORDER BY id DESC LIMIT 1", sym
            )
            if not row:
                await self.tg.send(f"No positions for {sym}.")
            else:
                await self.tg.send(f"{sym}: last pid={int(row['id'])} status={row['status']}")

        elif root == "/repair" and len(parts) >= 2:
            # /repair 12345          → repair by pid
            # /repair XCNUSDT        → repair most recent CLOSED pid for symbol
            arg = parts[1]
            if arg.isdigit():
                await self._repair_closed_position(int(arg))
            else:
                pid = await self._find_pid_by_symbol(arg, status="CLOSED")
                if pid is None:
                    await self.tg.send(f"No CLOSED position found for {arg.upper()}.")
                else:
                    await self._repair_closed_position(pid)



    async def _resume(self):
        """
        Startup reconciliation: find valid positions, close orphans, clean orphan orders,
        resume valid ones, set peak equity and cooldowns.
        """
        LOG.info("--> Resuming state with intelligent reconciliation…")

        # 1) Exchange & DB state
        LOG.info("Fetching open positions from EXCHANGE & DATABASE…")
        try:
            exchange_positions = await self.exchange.fetch_positions()
            open_exchange_positions = {
                p['info']['symbol']: p for p in exchange_positions if float(p['info'].get('size', 0)) > 0
            }
            LOG.info("…exchange has %d open positions.", len(open_exchange_positions))
        except Exception as e:
            LOG.error("CRITICAL: Could not fetch exchange positions: %s. Exiting.", e)
            sys.exit(1)

        db_positions_rows = await self.db.fetch_open_positions()
        db_positions = {r["symbol"]: dict(r) for r in db_positions_rows}
        LOG.info("…database has %d OPEN positions.", len(db_positions))

        # 2) Reconcile & identify valid order CIDs
        LOG.info("Reconciling and identifying valid protective orders…")
        valid_cids = set()
        now_utc = datetime.now(timezone.utc)

        for symbol, pos_data in open_exchange_positions.items():
            if symbol not in db_positions:
                msg = f"🚨 ORPHAN DETECTED: Exchange position {symbol} not in DB. Closing."
                LOG.warning(msg)
                await self.tg.send(msg)
                try:
                    side = 'buy' if pos_data['side'] == 'short' else 'sell'
                    size = float(pos_data['info']['size'])
                    await self.exchange.create_market_order(symbol, side, size, params={'reduceOnly': True})
                except Exception as e:
                    LOG.error("Failed to force-close orphan position %s: %s", symbol, e)

        for symbol, pos_row in list(db_positions.items()):
            pid = pos_row["id"]
            if symbol not in open_exchange_positions:
                LOG.warning("DB/EX mismatch: %s OPEN in DB but not on exchange. Closing in DB.", symbol)
                await self.db.update_position(pid, status="CLOSED", closed_at=now_utc, pnl=0, exit_reason="RECONCILE_CLOSE")
                del db_positions[symbol]
                continue

            # Stale on restart?
            opened_at = pos_row.get("opened_at")
            max_holding_duration = None
            if self.cfg.get("TIME_EXIT_HOURS_ENABLED", False):
                max_holding_duration = timedelta(hours=self.cfg.get("TIME_EXIT_HOURS", 4))
            elif self.cfg.get("TIME_EXIT_ENABLED", False):
                max_holding_duration = timedelta(days=self.cfg.get("TIME_EXIT_DAYS", 10))

            if opened_at and max_holding_duration and (now_utc - opened_at) > max_holding_duration:
                msg = f"⏰ STALE ON RESTART: {symbol} (pid {pid}) older than time limit. Forcing close."
                LOG.warning(msg); await self.tg.send(msg)
                asyncio.create_task(self._force_close_position(pid, pos_row, tag="STALE_ON_RESTART"))
                del db_positions[symbol]
                continue

            ex_pos = open_exchange_positions[symbol]
            db_size = float(pos_row['size'])
            ex_size = float(ex_pos['info']['size'])
            if abs(db_size - ex_size) > 1e-9:
                msg = (f"🚨 SIZE MISMATCH for {symbol} (pid {pid}): DB={db_size}, EX={ex_size}. Marking for review.")
                LOG.critical(msg); await self.tg.send(msg)
                await self.db.update_position(pid, status="SIZE_MISMATCH")
                del db_positions[symbol]
                continue

            if pos_row.get("sl_cid"): valid_cids.add(pos_row["sl_cid"])
            if pos_row.get("tp1_cid"): valid_cids.add(pos_row["tp1_cid"])
            if pos_row.get("tp_final_cid"): valid_cids.add(pos_row["tp_final_cid"])
            if pos_row.get("sl_trail_cid"): valid_cids.add(pos_row["sl_trail_cid"])

        LOG.info("…identified %d valid CIDs to preserve.", len(valid_cids))

        # 3) Clean orphan orders
        LOG.info("Fetching all open orders to clean orphans…")
        try:
            all_open_orders = await self._all_open_orders_for_all_symbols()
            cancel_tasks = []
            for order in all_open_orders:
                cid = order.get("clientOrderId")
                if cid and cid.startswith("bot_") and cid not in valid_cids:
                    LOG.warning("Orphaned order %s (%s). Cancelling.", cid, order['symbol'])
                    cancel_tasks.append(self.exchange.cancel_order(order['id'], order['symbol'], params={'category': 'linear'}))
            if cancel_tasks:
                await asyncio.gather(*cancel_tasks, return_exceptions=True)
                LOG.info("…orphaned order cleanup complete.")
            else:
                LOG.info("…no orphaned orders found.")
        except Exception as e:
            LOG.error("Failed clean slate protocol on startup: %s", e)
            traceback.print_exc()

        # 4) Load reconciled positions into memory
        LOG.info("Loading reconciled positions into memory…")
        for symbol, pos_row in db_positions.items():
            self.open_positions[pos_row["id"]] = pos_row
            LOG.info("Resumed open position for %s (ID %d)", symbol, pos_row["id"])

        LOG.info("Fetching peak equity from DB…")
        peak = await self.db.pool.fetchval("SELECT MAX(equity) FROM equity_snapshots")
        self.peak_equity = float(peak) if peak is not None else 0.0
        LOG.info("Peak equity loaded: $%.2f", self.peak_equity)

        LOG.info("Loading recent exit timestamps for cooldowns…")
        cd_h = int(self.cfg.get("SYMBOL_COOLDOWN_HOURS", cfg.SYMBOL_COOLDOWN_HOURS))
        rows = await self.db.pool.fetch(
            "SELECT symbol, closed_at FROM positions "
            "WHERE status='CLOSED' AND closed_at > (NOW() AT TIME ZONE 'utc') - $1::interval",
            timedelta(hours=cd_h),
        )
        for r in rows:
            self.last_exit[r["symbol"]] = r["closed_at"]

        LOG.info("<-- Resume complete.")

    async def _generate_summary_report(self, period: str) -> str:
        now = datetime.now(timezone.utc)
        period_map = {
            '6h': timedelta(hours=6),
            'daily': timedelta(days=1),
            'weekly': timedelta(weeks=1),
            'monthly': timedelta(days=30)
        }
        if period not in period_map:
            return f"Error: Unknown report period '{period}'."

        start_time = now - period_map[period]
        LOG.info("Generating %s summary report since %s", period, start_time.isoformat())

        try:
            query = """
                SELECT pnl FROM positions
                WHERE status = 'CLOSED' AND closed_at >= $1
            """
            records = await self.db.pool.fetch(query, start_time)
            if not records:
                return f"📊 *{period.capitalize()} Report*\n\nNo trades were closed in the last {period}."

            total_trades = len(records)
            pnl_values = [float(r['pnl']) for r in records if r['pnl'] is not None]
            wins = [p for p in pnl_values if p > 0]
            losses = [p for p in pnl_values if p < 0]

            win_count = len(wins)
            loss_count = len(losses)
            win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
            total_pnl = sum(pnl_values)
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            avg_win = sum(wins) / win_count if win_count > 0 else 0
            avg_loss = sum(losses) / loss_count if loss_count > 0 else 0
            expectancy = (avg_win * (win_rate / 100)) - (abs(avg_loss) * (1 - (win_rate / 100)))

            report_lines = [
                f"📊 *{period.capitalize()} Performance Summary*",
                f"```{'-'*25}",
                f" Period: Last {period}",
                f" Total Closed Trades: {total_trades}",
                f" Total PnL: {total_pnl:+.2f} USDT",
                f"",
                f" Win Rate: {win_rate:.2f}% ({win_count} W / {loss_count} L)",
                f" Profit Factor: {profit_factor:.2f}",
                f" Expectancy/Trade: {expectancy:+.2f} USDT",
                f"",
                f" Avg Win:  {avg_win:+.2f} USDT",
                f" Avg Loss: {avg_loss:+.2f} USDT",
                f"```{'-'*25}",
            ]
            return "\n".join(report_lines)
        except Exception as e:
            LOG.error("Failed to generate summary report: %s", e)
            return f"Error: Could not generate {period} report. Check logs."

    def _strategy_declared_side(self) -> str | None:
        """
        Read 'strategy.side' from the loaded YAML spec file, robustly.
        Returns 'long' | 'short' | None
        """
        # 1) Try reading from the actual YAML file
        try:
            path = getattr(self, "_strategy_spec_path", None)
            if path:
                from pathlib import Path as _P
                import yaml as _yaml
                data = _yaml.safe_load(_P(path).read_text()) or {}
                st = data.get("strategy") or {}
                s = st.get("side")
                if s:
                    s = str(s).strip().lower()
                    if s in ("long", "short"):
                        return s
        except Exception:
            pass

        # 2) Fallback: try whatever the StrategyEngine exposes
        try:
            spec = getattr(self.strategy_engine, "spec", None)
            if isinstance(spec, dict):
                st = spec.get("strategy") or {}
                s = st.get("side")
                if s:
                    s = str(s).strip().lower()
                    if s in ("long", "short"):
                        return s
        except Exception:
            pass
        return None

    def _resolve_side(self, verdict) -> str:
        """
        Precedence for deciding side:
        1) STRATEGY_SIDE_OVERRIDE in config.yaml (if set to 'long'/'short')
        2) YAML 'strategy.side' (from the strategy file)
        3) verdict.side (if provided by StrategyEngine)
        4) default: 'short'
        """
        # 1) Hard override via config
        try:
            forced = str(self.cfg.get("STRATEGY_SIDE_OVERRIDE", "") or "").strip().lower()
            if forced in ("long", "short"):
                return forced
        except Exception:
            pass

        # 2) YAML
        y = self._strategy_declared_side()
        if y:
            return y

        # 3) Verdict
        try:
            v = getattr(verdict, "side", None)
            if v:
                vs = str(v).strip().lower()
                if vs in ("long", "short"):
                    return vs
        except Exception:
            pass

        # 4) Default
        return "short"

    async def _reporting_loop(self):
        LOG.info("Reporting loop started.")
        last_report_sent = {}
        while True:
            await asyncio.sleep(60 * 5)
            now = datetime.now(timezone.utc)
            periods_to_check = {
                '6h': now.hour % 6 == 0,
                'daily': now.hour == 0,
                'weekly': now.weekday() == 0 and now.hour == 0,
            }
            for period, should_send in periods_to_check.items():
                last_sent_date = last_report_sent.get(period)
                is_already_sent = False
                if period in ['daily', 'weekly'] and last_sent_date == now.date():
                    is_already_sent = True
                elif period == '6h' and last_sent_date == (now.date(), now.hour // 6):
                    is_already_sent = True

                if should_send and not is_already_sent:
                    LOG.info("Sending scheduled '%s' report.", period)
                    summary_text = await self._generate_summary_report(period)
                    await self.tg.send(summary_text)
                    if period in ['daily', 'weekly']:
                        last_report_sent[period] = now.date()
                    elif period == '6h':
                        last_report_sent[period] = (now.date(), now.hour // 6)

    # ───────────────────── Run ─────────────────────

    async def run(self):
        await self.db.init()
        await self.db.migrate_schema()

        if self.settings.bybit_testnet:
            LOG.warning("="*60)
            LOG.warning("RUNNING ON TESTNET")
            LOG.warning("Testnet data is unreliable for most altcoins.")
            LOG.warning("="*60)

        LOG.info("Loading exchange markets…")
        try:
            await self.exchange._exchange.load_markets()
            LOG.info("Markets loaded.")
        except Exception as e:
            LOG.error("Could not load markets: %s. Exiting.", e)
            return

        LOG.info("Loading symbol listing dates…")
        self._listing_dates_cache = await self._load_listing_dates()

        await self._resume()
        await self.tg.send("🤖 DONCH v1.0")

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._main_signal_loop())
                tg.create_task(self._manage_positions_loop())
                tg.create_task(self._telegram_loop())
                tg.create_task(self._equity_loop())
                tg.create_task(self._reporting_loop())
        except* (asyncio.CancelledError, KeyboardInterrupt):
            LOG.info("Shutdown signal received.")
        finally:
            await self.exchange.close()
            if self.db.pool:
                await self.db.pool.close()
            await self.tg.close()
            LOG.info("Bot shut down cleanly.")

    # ───────────────────── Listing-date I/O (unchanged logic) ─────────────────

    async def _load_listing_dates(self) -> Dict[str, datetime.date]:
        if LISTING_PATH.exists():
            raw = json.loads(LISTING_PATH.read_text())
            return {s: datetime.fromisoformat(ts).date() for s, ts in raw.items()}

        LOG.info("listing_dates.json not found. Fetching from exchange.")
        async def fetch_date(sym):
            try:
                candles = await self.exchange.fetch_ohlcv(sym, timeframe="1d", limit=1000)
                if candles:
                    ts = min(row[0] for row in candles)
                    return sym, datetime.fromtimestamp(ts/1000, tz=timezone.utc).date()
            except Exception as e:
                LOG.warning("Could not fetch listing date for %s: %s", sym, e)
            return sym, None

        results = await asyncio.gather(*(fetch_date(s) for s in self.symbols))
        out = {sym: d for sym, d in results if d}
        if out:
            LISTING_PATH.write_text(json.dumps({k: v.isoformat() for k, v in out.items()}, indent=2))
            LOG.info("Saved %d listing dates to %s", len(out), LISTING_PATH)
        else:
            LOG.warning("No listing dates could be determined; leaving file absent.")
        return out

    # ───────────────────── Exchange order helpers ─────────────────────



    async def _fetch_by_cid(self, cid: Optional[str], symbol: str, silent: bool = False):
        if not cid:
            return None
        try:

            base = {"clientOrderId": cid, "acknowledged": True, "category": "linear"}
            # 1) Try conditional pools (StopOrder / TP/SL)
            for params in (
                {**base, "trigger": True, "orderFilter": "StopOrder"},
                {**base, "trigger": True, "orderFilter": "tpslOrder"},
            ):
                try:
                    o = await self.exchange.fetch_order(None, symbol, params)
                    if o:
                        return o
                except Exception:
                    pass
            # 2) Fall back to active orders
            return await self.exchange.fetch_order(None, symbol, base)

        except Exception as e:
            if not silent:
                LOG.warning("Fetch by CID %s for %s failed (fallback to trades): %s", cid, symbol, e)
            return None


    async def _cancel_by_cid(self, cid: str, symbol: str):
        try:
            return await self.exchange.cancel_order(
                None, symbol, params={"clientOrderId": cid, "category": "linear"}
            )
        except ccxt.OrderNotFound:
            LOG.warning("Order %s for %s already filled/cancelled.", cid, symbol)
        except Exception as e:
            LOG.error("Failed to cancel order %s for %s: %s", cid, symbol, e)

    async def _all_open_orders(self, symbol: str) -> list:
        params_linear = {'category': 'linear'}
        try:
            active = await self.exchange.fetch_open_orders(symbol, params=params_linear)
            stop = await self.exchange.fetch_open_orders(symbol, params={**params_linear, 'orderFilter': 'StopOrder'})
            tpsl = await self.exchange.fetch_open_orders(symbol, params={**params_linear, 'orderFilter': 'tpslOrder'})
            return active + stop + tpsl
        except Exception as e:
            LOG.warning("Could not fetch open orders for %s: %s", symbol, e)
            return []

    async def _all_open_orders_for_all_symbols(self) -> list:
        params_linear = {'category': 'linear'}
        try:
            active = await self.exchange.fetch_open_orders(params=params_linear)
            stop = await self.exchange.fetch_open_orders(params={**params_linear, 'orderFilter': 'StopOrder'})
            tpsl = await self.exchange.fetch_open_orders(params={**params_linear, 'orderFilter': 'tpslOrder'})
            return active + stop + tpsl
        except Exception as e:
            LOG.warning("Could not fetch all open orders: %s", e)
            return []


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (module-level)
# ──────────────────────────────────────────────────────────────────────────────

def create_unique_cid(tag: str) -> str:
    """Unique client order ID for entries."""
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    random_suffix = secrets.token_hex(2)
    return f"bot_{tag}_{timestamp_ms}_{random_suffix}"[:36]

def create_stable_cid(pid: int, tag: str) -> str:
    """Stable client order ID for SL/TP."""
    return f"bot_{pid}_{tag}"[:36]


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

async def async_main():
    try:
        settings = Settings()
    except ValidationError as e:
        LOG.error("Bad env: %s", e)
        sys.exit(1)

    cfg_dict = load_yaml(CONFIG_PATH)
    trader = LiveTrader(settings, cfg_dict)
    await trader.run()

if __name__ == "__main__":
    asyncio.run(async_main())
