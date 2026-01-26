# live/parity_utils.py
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
from . import indicators as ta


def _norm_tf(tf: str) -> str:
    """
    Normalize timeframe strings to pandas-safe aliases.
    Prevents '15m' being interpreted as 15 months.
    """
    tf = str(tf).strip()
    mapping = {
        "1m": "1min", "3m": "3min", "5m": "5min",
        "15m": "15min", "30m": "30min",
        "1h": "1h", "2h": "2h", "4h": "4h",
        "1d": "1D", "1D": "1D",
    }
    if tf in mapping:
        return mapping[tf]

    if tf.endswith("m") and not tf.endswith("min"):
        return tf[:-1] + "min"

    if tf.lower().endswith("d"):
        return tf.upper()

    return tf


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Standard resampling (Label='left', Closed='left').
    Used for Entry Quality features (ATR, RSI, ADX, etc.) and days_since_prev_break.
    """
    if df is None or df.empty:
        return df

    freq = _norm_tf(timeframe)

    # Explicit left/left to match offline/scout behavior
    res = df.resample(freq, label="left", closed="left")
    agg = res.agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return agg.dropna()


def resample_ohlcv_robust(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Robust resampling (Label='right', Closed='right').
    Matches offline 'scout' usage for Regime detection and Donchian 'completed days' logic.
    """
    if df.empty:
        return df

    freq = _norm_tf(timeframe)
    res = df.resample(freq, label="right", closed="right")
    agg = res.agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return agg.dropna()


def map_to_left_index(target_index: pd.DatetimeIndex, source_series: pd.Series) -> pd.Series:
    """
    Forward-fill source_series onto target_index.
    Matches offline indicators.map_to_left_index.
    """
    return source_series.reindex(target_index, method="ffill")


def donchian_upper_days_no_lookahead(high_5m: pd.Series, n_days: int) -> pd.Series:
    """
    Daily Donchian upper on *completed* days only.
    Matches offline scout.py logic (Right/Right + dropna + shift).

    Returns a pd.Series indexed by high_5m.index (wrapping the numpy array from scout logic).
    """
    daily_high = high_5m.resample("1D", label="right", closed="right").max().dropna()
    don_daily = daily_high.rolling(n_days, min_periods=n_days).max().shift(1)

    # Map to 5m bars by day start, then ffill through the day.
    keyed = don_daily.reindex(high_5m.index.floor("D"))
    arr = keyed.ffill().to_numpy(dtype=float)
    return pd.Series(arr, index=high_5m.index)


# =============================================================================
# META scope evaluation (pure helper; unit-testable)
# =============================================================================

def _to_float_or_none(v: Any) -> Optional[float]:
    """
    Best-effort parse to float. Returns None if missing/unparseable/non-finite.
    This is used as the internal primitive for "to_numeric(errors='coerce')".
    """
    if v is None:
        return None

    # pandas/np missing
    try:
        if isinstance(v, (float, np.floating)) and (not np.isfinite(float(v))):
            return None
    except Exception:
        pass

    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        try:
            x = float(s)
        except Exception:
            return None
        return float(x) if np.isfinite(float(x)) else None

    try:
        x = float(v)
    except Exception:
        return None

    return float(x) if np.isfinite(float(x)) else None


def eval_meta_scope(pstar_scope: Optional[str], meta_row: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluate meta scope with offline-parity semantics and fail-closed behavior.

    Offline semantics for decision.scope == "RISK_ON_1":
      - Use risk_on_1 if the key exists in the row dict.
      - Otherwise (only if the key is absent), fallback to risk_on.
      - Coerce with to_numeric(errors='coerce').fillna(0), then compare == 1.
      - If neither key exists at all, offline raises; live returns False and flags missing_cols.

    Supported scopes:
      - None / ""     : scope passes (True)
      - "RISK_ON_1"   : described above
      - any other     : False (fail-closed)

    Returns (scope_ok, info) where info includes raw inputs and the chosen source.
    """
    sc = (pstar_scope or "").strip()
    sc_u = sc.upper() if sc else ""

    has_risk_on_1 = ("risk_on_1" in meta_row)
    has_risk_on = ("risk_on" in meta_row)

    info: Dict[str, Any] = {
        "scope": sc_u or None,
        "risk_on_1_raw": meta_row.get("risk_on_1", None),
        "risk_on_raw": meta_row.get("risk_on", None),
        "risk_on_1_present": bool(has_risk_on_1),
        "risk_on_present": bool(has_risk_on),
        "scope_val": None,     # numeric value used after coercion/fillna(0)
        "scope_src": None,     # "risk_on_1" | "risk_on" | None
        "missing_cols": False,
    }

    if not sc_u:
        return True, info

    if sc_u != "RISK_ON_1":
        return False, info

    # Choose column by presence only (no row-wise fallback when present-but-NaN)
    if has_risk_on_1:
        raw = meta_row.get("risk_on_1", None)
        x = _to_float_or_none(raw)
        x = 0.0 if x is None else float(x)
        info["scope_val"] = x
        info["scope_src"] = "risk_on_1"
        return (x == 1.0), info

    if has_risk_on:
        raw = meta_row.get("risk_on", None)
        x = _to_float_or_none(raw)
        x = 0.0 if x is None else float(x)
        info["scope_val"] = x
        info["scope_src"] = "risk_on"
        return (x == 1.0), info

    info["missing_cols"] = True
    return False, info
