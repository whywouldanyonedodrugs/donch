# research/data/labels.py
import pandas as pd
import numpy as np

def compute_binary_label_from_realized(df: pd.DataFrame) -> pd.Series:
    """
    For the first research pass, label a trade win if realized P&L > 0.
    This mirrors what your current win-prob model learns from (on-taken trades).
    Later we can switch to counterfactual labels (TP/SL/time-exit from ex-ante rules).
    """
    if "pnl" in df.columns:
        return (df["pnl"].astype(float) > 0.0).astype(int)
    if "pnl_pct" in df.columns:
        return (df["pnl_pct"].astype(float) > 0.0).astype(int)
    raise ValueError("Neither pnl nor pnl_pct found to derive labels.")

def add_time_order_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'order_idx' for PurgedKFold/CPCV splits."""
    out = df.copy()
    out = out.sort_values(["opened_at", "symbol"]).reset_index(drop=True)
    out["order_idx"] = np.arange(len(out))
    return out
