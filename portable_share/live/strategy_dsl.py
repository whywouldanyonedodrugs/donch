from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import re


def _to_float(x: Any) -> float:
    try:
        if isinstance(x, bool):
            return 1.0 if x else 0.0
        return float(x)
    except Exception:
        raise ValueError(f"Cannot cast to float: {x!r}")

def _cmp(value: Any, op: str, threshold: Any = None, *, min_val=None, max_val=None) -> bool:
    if op in ("is_true", "isfalse", "is_false"):
        v = bool(value)
        return v if op == "is_true" else (not v)

    if op == "outside":
        assert min_val is not None and max_val is not None, "outside op requires min and max"
        v = _to_float(value); lo = _to_float(min_val); hi = _to_float(max_val)
        return not (lo <= v <= hi)

    v = _to_float(value)
    t = _to_float(threshold) if threshold is not None else None

    if op == "==":
        return v == t
    if op == "!=":
        return v != t
    if op == ">":
        return v > t
    if op == ">=":
        return v >= t
    if op == "<":
        return v < t
    if op == "<=":
        return v <= t

    raise ValueError(f"Unsupported op: {op}")

_cfg_re = re.compile(r"@cfg\.([A-Za-z_][A-Za-z0-9_]*)")

def resolve_cfg_refs(x: Any) -> Any:


    if not isinstance(x, str):
        return x
    try:
        import config as cfg
    except Exception:

        cfg = None

    def repl(m):
        key = m.group(1)
        if cfg and hasattr(cfg, key):
            v = getattr(cfg, key)
            return str(v)
        return m.group(0)

    s = _cfg_re.sub(repl, x)

    try:
        return float(s)
    except Exception:
        return s

@dataclass
class ConditionResult:
    ok: bool
    tag: str
    value: Any = None

@dataclass
class EngineVerdict:
    should_enter: bool
    reason_tags: List[str] = field(default_factory=list)
    entry_price: float = float("nan")
    atr_value: float = float("nan")
    rsi_value: float = float("nan")
    adx_value: float = float("nan")
    extras: Dict[str, Any] = field(default_factory=dict)
