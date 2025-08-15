# research/data/loader.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_dataset(path: str | Path = "research/data/dataset.parquet") -> pd.DataFrame:
    """
    Load the training dataset written by research/cli/build_dataset.py.
    Adds a stable 'order_idx' if missing and normalizes label naming.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p.resolve()}")
    df = pd.read_parquet(p)

    # Stable row id if not present
    if "order_idx" not in df.columns:
        sort_keys = [c for c in ("opened_at", "symbol") if c in df.columns]
        if sort_keys:
            df = df.sort_values(sort_keys).reset_index(drop=True)
        df["order_idx"] = range(len(df))

    # Normalize label name: many CLIs expect 'win'
    if "win" not in df.columns and "y" in df.columns:
        df = df.rename(columns={"y": "win"})

    return df
