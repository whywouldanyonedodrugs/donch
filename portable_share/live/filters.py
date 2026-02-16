from __future__ import annotations

from typing import List, Tuple, Optional
import config as cfg


def evaluate(
    sig, *, listing_age_days: Optional[int], open_positions: int, equity: float
) -> Tuple[bool, List[str]]:


    vetoes: List[str] = []
    ok = True


    if getattr(cfg, "GAP_FILTER_ENABLED", True):
        if not sig.vwap_consolidated:
            vetoes.append("GAP")
            ok = False


    if listing_age_days is not None:
        if listing_age_days < cfg.MIN_COIN_AGE_DAYS:
            vetoes.append("AGE_TOO_NEW")
            ok = False
        elif listing_age_days > cfg.MAX_COIN_AGE_DAYS:
            vetoes.append("AGE_TOO_OLD")
            ok = False


    if getattr(cfg, "RSI_FILTER_ENABLED", True):
        _rsi = float(getattr(sig, "rsi", 0.0) or 0.0)
        _lo  = float(getattr(cfg, "RSI_ENTRY_MIN", 30))
        _hi  = float(getattr(cfg, "RSI_ENTRY_MAX", 70))
        _eps = float(getattr(cfg, "RSI_EPS", 0.0))
        if not ((_lo - _eps) <= _rsi <= (_hi + _eps)):
            vetoes.append(f"RSI_RANGE(rsi={_rsi:.2f}∉[{_lo:.0f},{_hi:.0f}])")
            ok = False


    if getattr(cfg, "STRUCTURAL_TREND_FILTER_ENABLED", True):
        if sig.ret_30d is not None and sig.ret_30d > cfg.STRUCTURAL_TREND_RET_PCT:
            vetoes.append("STRUCTURAL_TREND")
            ok = False


    if cfg.ADX_FILTER_ENABLED:
        if not (cfg.ADX_MIN <= sig.adx <= cfg.ADX_MAX):
            vetoes.append("ADX")
            ok = False


    if open_positions >= cfg.MAX_OPEN:
        vetoes.append("MAX_OPEN")
        ok = False
    if equity < getattr(cfg, "MIN_EQUITY_USDT", 0):
        vetoes.append("LOW_EQUITY")
        ok = False

    return ok, vetoes
