"""
strategy_hooks.py — LiveFader glue
==================================

This file gives tiny helpers so you can wire StrategyEngine into the existing
live_trader.py with minimal edits, without changing DB or Telegram flows.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from dataclasses import asdict
from datetime import datetime, timezone, timedelta

def build_context_for_engine(
    *, base_tf: str, ema_tf: str, rsi_tf: str, atr_tf: str,
    market_regime: str, eth_macd_hist: float | None
) -> Dict[str, Any]:
    """Create a minimal context dict for sizing/veto scalers (technical + regime)."""
    return {
        "base_tf": base_tf,
        "ema_tf": ema_tf,
        "rsi_tf": rsi_tf,
        "atr_tf": atr_tf,
        "market_regime": market_regime,
        "eth_macdhist": float(eth_macd_hist or 0.0),
    }

def build_governance_context(
    symbol: str,
    *,
    equity: float,
    open_positions: int,
    listing_age_days: Optional[int],
    last_exit_dt: Optional[datetime],
    cooldown_hours: float,
    is_symbol_blacklisted: bool
) -> Dict[str, Any]:
    """
    Governance context that the DSL can use in veto logic.
    You can reference these keys in YAML with source: "context".
    """
    now = datetime.now(timezone.utc)
    cd_ok = True
    if last_exit_dt is not None:
        cd_ok = (now - last_exit_dt) >= timedelta(hours=float(cooldown_hours))

    return {
        "symbol": symbol,
        "equity": float(equity or 0.0),
        "open_positions": int(open_positions or 0),
        "listing_age_days": int(listing_age_days) if listing_age_days is not None else -1,
        "cooldown_ok": bool(cd_ok),
        "is_blacklisted": bool(is_symbol_blacklisted),
    }

def build_signal_from_verdict(SignalClass, verdict, *, symbol: str, listing_age_days: int | None,
                              session_tag: str, day_of_week: int, hour_of_day: int,
                              vwap_diag: dict[str, float]):
    """Convert EngineVerdict to your bot's Signal dataclass instance."""
    sig = SignalClass(
        symbol=symbol,
        entry=float(verdict.entry_price),
        atr=float(verdict.atr_value or 0.0),
        rsi=float(verdict.rsi_value or 0.0),
        adx=float(verdict.adx_value or 0.0),
        atr_pct=float((verdict.atr_value or 0.0) / max(1e-12, verdict.entry_price) * 100.0),
        market_regime="UNKNOWN",
        price_boom_pct=0.0,
        price_slowdown_pct=0.0,
        vwap_dev_pct=float(vwap_diag.get("vwap_dev_pct", 0.0)),
        vwap_z_score=float(vwap_diag.get("vwap_z_score", 0.0)),
        ret_30d=0.0,
        ema_fast=0.0,
        ema_slow=0.0,
        listing_age_days=listing_age_days or -1,
        session_tag=session_tag,
        day_of_week=day_of_week,
        hour_of_day=hour_of_day,
        vwap_consolidated=bool(vwap_diag.get("vwap_consolidated", False)),
        is_ema_crossed_down=False,
    )
    # Attach optional diagnostics if your DB expects them
    setattr(sig, "vwap_stack_frac", float(vwap_diag.get("vwap_stack_frac", 0.0)))
    setattr(sig, "vwap_stack_expansion_pct", float(vwap_diag.get("vwap_stack_expansion_pct", 0.0)))
    setattr(sig, "vwap_stack_slope_pph", float(vwap_diag.get("vwap_stack_slope_pph", 0.0)))
    return sig
