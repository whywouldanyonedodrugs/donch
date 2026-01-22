# live/strategy_engine.py
from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

@dataclass
class Verdict:
    should_enter: bool
    side: str                  # "long" | "short"
    reason_tags: List[str]

class StrategyEngine:
    """
    YAML-driven rule engine for live_trader.
    Schema (example):
      name: DonchPullbackLong
      timeframes:
        base: 5m
        donch: 1h
        volume: 5m
      params:
        DONCH_PERIOD: 55
        PULLBACK_PCT_MAX: 0.015
        VOL_SMA_WIN: 50
      entry:
        side: long
        all:
          - donch_breakout: { tf: donch, period: "@params.DONCH_PERIOD", direction: up }
          - pullback_under_ma: { tf: base, ema_period: 200, max_pullback_pct: "@params.PULLBACK_PCT_MAX" }
          - volume_surge: { tf: base, win: "@params.VOL_SMA_WIN", min_mult: 1.5 }
      veto:
        any:
          - blacklist_symbol: {}
          - cooldown_hours: { hours: "@cfg.SYMBOL_COOLDOWN_HOURS" }
          - min_listing_age_days: { days: 7 }
    """
    def __init__(self, spec_path: str, cfg: Optional[Dict[str, Any]] = None):
        self.spec_path = Path(spec_path).resolve()
        self.cfg = cfg or {}
        if not self.spec_path.exists():
            raise FileNotFoundError(self.spec_path)
        self._spec = self._load()
        self._ops = _build_ops_registry()

    # --- public API ---
    def reload(self):
        self._spec = self._load()

    def required_timeframes(self) -> List[str]:
        tfs = self._spec.get("timeframes", {}) or {}
        return list({v for v in tfs.values() if isinstance(v, str)})

    def evaluate(self, dfs: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> Verdict:
        spec = self._spec
        tfs = spec.get("timeframes", {}) or {}
        params = spec.get("params", {}) or {}

        # resolve template refs like "@cfg.FOO" / "@params.BAR"
        def _resolve(x):
            if isinstance(x, str) and x.startswith("@"):
                if x.startswith("@cfg."):
                    return _deep_get(self.cfg, x[5:])
                if x.startswith("@params."):
                    return _deep_get(params, x[8:])
            return x

        # entry block
        entry = spec.get("entry", {}) or {}
        side = str(entry.get("side", "short")).lower()
        all_ops = entry.get("all", []) or []
        any_ops = entry.get("any", []) or []

        tags: List[str] = []

        # VETOS first (fail-fast)
        veto = spec.get("veto", {}) or {}
        veto_any = veto.get("any", []) or []
        for op in veto_any:
            ok, t = _eval_one(op, dfs, tfs, context, _resolve, self._ops)
            if not ok:
                tags.append(f"veto:{t or list(op.keys())[0]}")
                return Verdict(False, side, tags)

        # "all" ops must all pass
        for op in all_ops:
            ok, t = _eval_one(op, dfs, tfs, context, _resolve, self._ops)
            if not ok:
                return Verdict(False, side, tags + [t or list(op.keys())[0]])
            tags.append(t or list(op.keys())[0])

        # optional "any" block (if present, at least one must pass)
        if any_ops:
            passed_any = False
            temp_tags = []
            for op in any_ops:
                ok, t = _eval_one(op, dfs, tfs, context, _resolve, self._ops)
                if ok:
                    passed_any = True
                    temp_tags.append(t or list(op.keys())[0])
            if not passed_any:
                return Verdict(False, side, tags + temp_tags)
            tags.extend(temp_tags)

        return Verdict(True, side, tags)

    # --- internals ---
    def _load(self):
        with open(self.spec_path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return raw


def _deep_get(d: Dict[str, Any], dotted: str):
    cur = d
    for k in dotted.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur

# ---------------- ops registry ----------------

def _build_ops_registry() -> Dict[str, Callable]:
    return {
        "donch_breakout": _op_donch_breakout,
        "pullback_under_ma": _op_pullback_under_ma,
        "volume_surge": _op_volume_surge,
        "trend_filter_ema": _op_trend_filter_ema,
        "blacklist_symbol": _op_blacklist_symbol,
        "cooldown_hours": _op_cooldown_hours,
        "min_listing_age_days": _op_min_listing_age_days,
        "eth_macd_hist_above": _op_eth_macd_hist_above,
        "universe_rs_pct_gte": _op_universe_rs_pct_gte,
        "liquidity_median_24h_usd_gte": _op_liquidity_median_24h_usd_gte,
        "eth_macd_bull": _op_eth_macd_bull,
        "donch_breakout_daily_confirm": _op_donch_breakout_daily_confirm,
        "volume_median_multiple": _op_volume_median_multiple,
        "pullback_retest_close_above_break": _op_pullback_retest_close_above_break,
        "micro_vol_filter": _op_micro_vol_filter,
    }

def _get_tf_df(dfs: Dict[str, pd.DataFrame], tfs: Dict[str, str], tf_key: str) -> Optional[pd.DataFrame]:
    tf = tfs.get(tf_key, tf_key) if isinstance(tf_key, str) else tf_key
    return dfs.get(tf)

def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=int(period), adjust=False).mean()

def _op_donch_breakout(args, dfs, tfs, ctx, resolve):
    # args: { tf: <key or timeframe>, period: int, direction: up|down }
    tf_df = _get_tf_df(dfs, tfs, resolve(args.get("tf", "base")))
    period = int(resolve(args.get("period", 55)) or 55)
    direction = str(resolve(args.get("direction", "up"))).lower()
    if tf_df is None or len(tf_df) < period + 2:
        return False, "donch:insufficient_data"
    df = tf_df
    hh = df["high"].rolling(period).max().shift(1)  # yesterday's channel
    ll = df["low"].rolling(period).min().shift(1)
    close = df["close"]
    if close.isna().any() or hh.isna().any() or ll.isna().any():
        return False, "donch:nan"
    c = float(close.iloc[-1])
    up = bool(c > float(hh.iloc[-1]))
    dn = bool(c < float(ll.iloc[-1]))
    if direction == "up" and up:
        return True, "donch_up"
    if direction == "down" and dn:
        return True, "donch_down"
    return False, "donch:no_break"

def _op_pullback_under_ma(args, dfs, tfs, ctx, resolve):
    # args: { tf: base, ema_period: 200, max_pullback_pct: 0.015 }
    tf_df = _get_tf_df(dfs, tfs, resolve(args.get("tf", "base")))
    ema_period = int(resolve(args.get("ema_period", 200)) or 200)
    max_pb = float(resolve(args.get("max_pullback_pct", 0.02)) or 0.02)
    if tf_df is None or len(tf_df) < ema_period + 5:
        return False, "pb:insufficient"
    df = tf_df
    ema = _ema(df["close"], ema_period)
    c = float(df["close"].iloc[-1])
    e = float(ema.iloc[-1])
    if e <= 0:
        return False, "pb:ema0"
    # For longs, "under" means small dip below EMA; for shorts, use negative if needed later
    pb = (e - c) / e
    ok = 0 <= pb <= max_pb
    return (ok, f"pullback_{pb:.3f}")

def _op_volume_surge(args, dfs, tfs, ctx, resolve):
    # args: { tf: base, win: 50, min_mult: 1.5 }
    tf_df = _get_tf_df(dfs, tfs, resolve(args.get("tf", "base")))
    win = int(resolve(args.get("win", 50)) or 50)
    mult = float(resolve(args.get("min_mult", 1.5)) or 1.5)
    if tf_df is None or len(tf_df) < win + 5:
        return False, "vol:insufficient"
    df = tf_df
    v = df["volume"]
    vbar = v.rolling(win).mean().shift(1)
    if float(vbar.iloc[-1] or 0) <= 0:
        return False, "vol:avg0"
    ok = float(v.iloc[-1]) >= mult * float(vbar.iloc[-1])
    return (ok, f"vol_x{float(v.iloc[-1])/float(vbar.iloc[-1]):.2f}")

def _op_trend_filter_ema(args, dfs, tfs, ctx, resolve):
    # args: { tf: base, fast: 50, slow: 200, direction: up|down }
    tf_df = _get_tf_df(dfs, tfs, resolve(args.get("tf", "base")))
    fast = int(resolve(args.get("fast", 50)) or 50)
    slow = int(resolve(args.get("slow", 200)) or 200)
    direction = str(resolve(args.get("direction", "up"))).lower()
    if tf_df is None or len(tf_df) < slow + 5:
        return False, "trend:insufficient"
    df = tf_df
    ema_f = _ema(df["close"], fast)
    ema_s = _ema(df["close"], slow)
    f = float(ema_f.iloc[-1]); s = float(ema_s.iloc[-1])
    if direction == "up" and (f > s):
        return True, "trend_up"
    if direction == "down" and (f < s):
        return True, "trend_down"
    return False, "trend:no"

def _op_blacklist_symbol(args, dfs, tfs, ctx, resolve):
    return (not bool(ctx.get("is_symbol_blacklisted", False)), "not_blacklisted")

def _op_cooldown_hours(args, dfs, tfs, ctx, resolve):
    hours = float(resolve(args.get("hours", 0)) or 0.0)
    if hours <= 0:
        return True, "cooldown:off"
    last_exit_dt = ctx.get("last_exit_dt")
    if not last_exit_dt:
        return True, "cooldown:none"
    delta = datetime.now(timezone.utc) - last_exit_dt
    ok = delta >= timedelta(hours=hours)
    return ok, f"cooldown_ok_{ok}"

def _op_min_listing_age_days(args, dfs, tfs, ctx, resolve):
    days = int(resolve(args.get("days", 0)) or 0)
    age = ctx.get("listing_age_days")
    if age is None:
        return False, "list_age:unknown"
    return (age >= days, f"list_age_{age}d")

def _op_eth_macd_hist_above(args, dfs, tfs, ctx, resolve):
    """
    Gate by ETH 4h MACD histogram (provided via context).
    args: { min: 0.0 }  # trade only if hist >= min
    Context key expected: ctx["eth_macdhist"] (float)
    """
    thr = float(resolve(args.get("min", 0.0)) or 0.0)
    val = float(ctx.get("eth_macdhist", 0.0) or 0.0)
    ok = (val >= thr)
    return ok, f"eth_hist_{val:.3f}_{'>=' if ok else '<'}_{thr:.3f}"

def _op_universe_rs_pct_gte(args, dfs, tfs, ctx, resolve):
    """
    Require a minimum weekly relative-strength percentile (0..100).
    Expects ctx["rs_pct"] precomputed by the bot's universe pass.
    """
    thr = float(resolve(args.get("min", 0)) or 0)
    rs = float(ctx.get("rs_pct", 0.0) or 0.0)
    return (rs >= thr), f"rs_pct={rs:.1f}>= {thr:.1f}"

def _op_liquidity_median_24h_usd_gte(args, dfs, tfs, ctx, resolve):
    """
    Liquidity cut: median 24h USD turnover >= threshold.
    Expects ctx["median_24h_turnover_usd"] from universe pass.
    """
    thr = float(resolve(args.get("min_usd", 5e5)) or 0.0)
    med = float(ctx.get("median_24h_turnover_usd", 0.0) or 0.0)
    return (med >= thr), f"liq_med24h=${med:,.0f}>={thr:,.0f}"

def _op_eth_macd_bull(args, dfs, tfs, ctx, resolve):
    """
    Gate by ETH 4h MACD regime.
    Requires BOTH (macd > signal) AND (hist > 0).
    Expects ctx['eth_macd'] dict: {'macd':..., 'signal':..., 'hist':...}
    """
    ed = ctx.get("eth_macd") or {}
    macd = float(ed.get("macd", 0.0) or 0.0)
    sig  = float(ed.get("signal", 0.0) or 0.0)
    hist = float(ed.get("hist", 0.0) or 0.0)
    ok = (macd > sig) and (hist > 0.0)
    return ok, f"eth_bull(macd={macd:.3f},sig={sig:.3f},hist={hist:.3f})"

def _op_donch_breakout_daily_confirm(args, dfs, tfs, ctx, resolve):
    """
    Daily Donchian breakout using prior N *full* days (shifted by 1 day).
    Require CLOSE above the upper band on the breakout bar.
    Args: { donch_tf: '1d', period: 20 }
    Returns (ok, meta) and stores ctx['donch_break_level'] for later ops.
    """
    # BEFORE:
    # donch_tf = str(resolve(args.get("donch_tf", "1d")))
    # n = int(resolve(args.get("period", 20)))
    # ddf = dfs.get(donch_tf)
    # if ddf is None or len(ddf) < n + 2:
    #     return False, "donch_daily:insufficient"

    # AFTER (alias-safe):
    donch_tf = str(resolve(args.get("donch_tf", "1d")))
    n = int(resolve(args.get("period", 20)))
    ddf = _get_tf_df(dfs, tfs, donch_tf)
    if ddf is None or len(ddf) < n + 2:
        return False, "donch_daily:insufficient"

    # Use prior N full days (exclude the current unfinished/most recent day)
    dd = ddf.copy().sort_index()
    highs = dd["high"].rolling(n).max().shift(1)  # prior full day
    last_close = float(dd["close"].iloc[-1])
    upper = float(highs.iloc[-1])
    ok = last_close > upper and np.isfinite(upper)

    if ok:
        ctx["donch_break_level"] = upper
        ctx["donch_dist_pct"] = (last_close - upper) / upper if upper > 0 else 0.0
    return ok, f"donch{n}_close>{upper:.6f}"

def _op_micro_vol_filter(args, dfs, tfs, ctx, resolve):
    """
    Avoid dust moves: ATR(1h) / last_price >= min_ratio.
    Args: { atr_tf: '1h', min_ratio: 0.0001, atr_len: 14 }
    """
    atr_tf = str(resolve(args.get("atr_tf", "1h")))
    atr_len = int(resolve(args.get("atr_len", 14)))
    min_ratio = float(resolve(args.get("min_ratio", 0.0001)))
    basetf = tfs.get("base", "5m")
    df_base = _get_tf_df(dfs, tfs, basetf)
    df_atr  = _get_tf_df(dfs, tfs, str(resolve(args.get("atr_tf","1h"))))
    if df_base is None or df_atr is None or len(df_atr) < atr_len + 5:
        return False, "microvol:short"

    # compute ATR on atr_tf then ffill to base
    from . import indicators as ta
    atr_s = ta.atr(df_atr, atr_len)
    atr_last = float(atr_s.iloc[-1])
    price_last = float(df_base["close"].iloc[-1])
    ratio = (atr_last / price_last) if price_last > 0 else 0.0
    ok = ratio >= min_ratio
    return ok, f"atr1h/px={ratio:.5f}>={min_ratio:.5f}"

def _op_pullback_retest_close_above_break(args, dfs, tfs, ctx, resolve):
    """
    After a Donchian breakout (ctx['donch_break_level']), require:
      - a retest near the break level within a lookback window, then
      - current close above the break.
    Args: { tf: '5m', eps_pct: 0.003, lookback_bars: 288 }
    """
    level = ctx.get("donch_break_level", None)
    if level is None or not np.isfinite(level):
        return False, "retest:no_break_ctx"

    tf = str(resolve(args.get("tf", "5m")))
    eps = float(resolve(args.get("eps_pct", 0.003)))
    lb  = int(resolve(args.get("lookback_bars", 288)))

    # ALIAS-SAFE (fix): use _get_tf_df instead of dfs.get(tf)
    df = _get_tf_df(dfs, tfs, tf)
    if df is None or len(df) < lb + 5:
        return False, "retest:short"

    sub = df.tail(lb)
    band_hi = level * (1.0 + eps)
    band_lo = level * (1.0 - eps)
    touched = ((sub["low"] <= band_hi) & (sub["high"] >= band_lo)).any()
    cur_close = float(sub["close"].iloc[-1])
    trig_ok = cur_close > level
    ok = bool(touched and trig_ok)
    return ok, f"retest@{level:.6f}&close>{level:.6f}"


def _op_volume_median_multiple(args, dfs, tfs, ctx, resolve):
    """
    Volume confirmation on base tf (usually 5m).
    Args: { tf: '5m', days: 30, min_mult: 2.0, cap_bars: 9000 }
    Uses rolling median of volume over last 'days' (bounded by 'cap_bars').
    """
    tf = str(resolve(args.get("tf", "5m")))
    days = int(resolve(args.get("days", 30)))
    mult = float(resolve(args.get("min_mult", 2.0)))
    cap = int(resolve(args.get("cap_bars", 9000)))
    df = _get_tf_df(dfs, tfs, tf)

    if df is None or df.empty:
        return False, "vol:missing"

    # Bars/day estimate (regular spacing fallback)
    try:
        bpd = int(round(24*60 / max(1, int((df.index[1]-df.index[0]).total_seconds()/60))))
    except Exception:
        bpd = 288  # 5m fallback

    lookback = min(cap, days * bpd)
    sub = df.tail(lookback)
    if len(sub) < max(100, bpd):
        return False, "vol:short"

    cur_vol = float(sub["volume"].iloc[-1])
    med_vol = float(sub["volume"].median())
    ratio = (cur_vol / med_vol) if med_vol > 0 else 0.0
    ok = ratio >= mult
    ctx["vol_mult"] = float(ratio)
    return ok, f"vol×={ratio:.2f}"


def _eval_one(op_dict, dfs, tfs, ctx, resolve, registry):
    if not isinstance(op_dict, dict) or len(op_dict) != 1:
        return False, "bad_op"
    name, args = next(iter(op_dict.items()))
    args = args or {}
    fn = registry.get(name)
    if not fn:
        return False, f"unknown:{name}"
    # resolve any template refs inside args dict (shallow is fine for our simple schema)
    args = {k: resolve(v) for k, v in args.items()}
    ok, tag = fn(args, dfs, tfs, ctx, resolve)
    return ok, tag
