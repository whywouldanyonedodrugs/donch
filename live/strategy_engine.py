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
