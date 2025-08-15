# research/data/tca.py
import pandas as pd
import numpy as np

def _to_float_series(s: pd.Series) -> pd.Series:
    """Coerce a Series (Decimal/str/None/float) to float, NaNs -> 0.0."""
    if s is None:
        return pd.Series([], dtype=float)
    out = pd.to_numeric(s, errors="coerce")
    # If it's not numeric yet (object), try casting elementwise
    if not np.issubdtype(out.dtype, np.number):
        out = pd.to_numeric(s.astype(str), errors="coerce")
    return out.astype(float).fillna(0.0)

def add_cost_audit(
    df: pd.DataFrame,
    funding_col: str = "funding_last_at_entry",
    taker_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Add rough fee + funding audit columns so reports can decompose P&L.
    Vectorized; safe for Decimal/None/object types.

    Columns used (if missing, treated as 0):
      - size, entry_price, opened_at, closed_at
      - funding_last_at_entry (or whatever you pass via funding_col)
    """
    out = df.copy()

    # Coerce numerics
    size = _to_float_series(out.get("size"))
    entry_price = _to_float_series(out.get("entry_price"))
    funding_rate = _to_float_series(out.get(funding_col))

    # Notional and fees
    notional = size.abs() * entry_price
    fee_rate = float(taker_bps) / 1e4  # bps -> decimal
    out["fee_usd_est"] = notional * fee_rate

    # Holding hours
    opened = pd.to_datetime(out.get("opened_at"), utc=True, errors="coerce")
    closed = pd.to_datetime(out.get("closed_at"), utc=True, errors="coerce")
    hours = (closed - opened).dt.total_seconds().fillna(0.0) / 3600.0

    # Funding (approx; assumes 8h funding interval)
    interval_h = 8.0
    out["funding_usd_est"] = notional * funding_rate * (hours / interval_h)

    # If any of the required columns were missing, the result will be 0.0 gracefully.
    return out
