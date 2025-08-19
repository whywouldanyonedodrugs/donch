"""
filters.py – v3.0 (Simplified)
Removes the Runtime class and relies on data passed directly to the evaluate function.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import config as cfg

# The Runtime class and its helper functions have been removed.

# ════════════════════════════════════════════════════════════════════════════
# Veto gate
# ════════════════════════════════════════════════════════════════════════════
def evaluate(
    sig, *, listing_age_days: Optional[int], open_positions: int, equity: float
) -> Tuple[bool, List[str]]:
    """
    Return (ok?, [tags]).
    This function now receives all necessary data directly.
    'sig' is a dataclass that now contains the vwap_consolidated status.
    """
    vetoes: List[str] = []
    ok = True

    # ── 1. VWAP gap check (MODIFIED) ────────────────────────────────────────
    # The signal object itself now carries the result of the VWAP check.
    if getattr(cfg, "GAP_FILTER_ENABLED", True):
        if not sig.vwap_consolidated:
            vetoes.append("GAP")
            ok = False

    # ── 2. Coin‑age veto ────────────────────────────────────────────────
    if listing_age_days is not None:
        if listing_age_days < cfg.MIN_COIN_AGE_DAYS:
            vetoes.append("AGE_TOO_NEW")
            ok = False
        elif listing_age_days > cfg.MAX_COIN_AGE_DAYS:
            vetoes.append("AGE_TOO_OLD")
            ok = False

    # ── 3. Primary RSI window ───────────────────────────────────────────
    # Allow toggle + small epsilon to avoid borderline 0.1–0.5pt misses.
    if getattr(cfg, "RSI_FILTER_ENABLED", True):
        _rsi = float(getattr(sig, "rsi", 0.0) or 0.0)
        _lo  = float(getattr(cfg, "RSI_ENTRY_MIN", 30))
        _hi  = float(getattr(cfg, "RSI_ENTRY_MAX", 70))
        _eps = float(getattr(cfg, "RSI_EPS", 0.0))
        if not ((_lo - _eps) <= _rsi <= (_hi + _eps)):
            vetoes.append(f"RSI_RANGE(rsi={_rsi:.2f}∉[{_lo:.0f},{_hi:.0f}])")
            ok = False
        
    # ── 4. 30‑day structural‑trend veto ─────────────────────────────────
    # Use getattr for a safe way to check for the new switch
    if getattr(cfg, "STRUCTURAL_TREND_FILTER_ENABLED", True):
        if sig.ret_30d is not None and sig.ret_30d > cfg.STRUCTURAL_TREND_RET_PCT:
            vetoes.append("STRUCTURAL_TREND")
            ok = False

    # ── 5. Optional ADX trend‑strength veto ─────────────────────────────
    if cfg.ADX_FILTER_ENABLED:
        if not (cfg.ADX_MIN <= sig.adx <= cfg.ADX_MAX):
            vetoes.append("ADX")
            ok = False

    # ── 6. Portfolio / equity caps ──────────────────────────────────────
    if open_positions >= cfg.MAX_OPEN:
        vetoes.append("MAX_OPEN")
        ok = False
    if equity < getattr(cfg, "MIN_EQUITY_USDT", 0):
        vetoes.append("LOW_EQUITY")
        ok = False

    return ok, vetoes