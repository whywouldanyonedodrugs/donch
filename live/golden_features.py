from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class GoldenFeatureStore:
    """
    Loads golden_features.parquet and serves exact-manifest raw feature rows by (symbol, timestamp).

    Expected parquet columns:
      - symbol (string)
      - timestamp (UTC; tz-aware or parseable to UTC)
      - plus at least the 73 manifest feature columns
    Extra parquet columns are allowed and ignored.

    Note: this store is intended for parity testing / validation and should be disabled in production.
    """

    path: Path
    feature_names: Sequence[str]
    symbols_allowlist: Optional[Sequence[str]] = None

    _df: Optional[pd.DataFrame] = None  # MultiIndex (symbol, timestamp) -> feature columns only

    def load(self) -> None:
        cols_needed = ["symbol", "timestamp"] + list(self.feature_names)
        df = pd.read_parquet(self.path, columns=cols_needed)

        if "symbol" not in df.columns or "timestamp" not in df.columns:
            raise ValueError("golden parquet must contain columns: symbol, timestamp")

        df["symbol"] = df["symbol"].astype(str).str.upper()

        ts = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
        df["timestamp"] = ts

        if self.symbols_allowlist:
            allow = {str(s).upper() for s in self.symbols_allowlist}
            df = df[df["symbol"].isin(sorted(allow))]

        missing_cols = sorted(set(self.feature_names) - set(df.columns))
        if missing_cols:
            raise ValueError(f"golden parquet missing required feature columns: {missing_cols[:50]}")

        # Keep only exact feature columns; index by (symbol, timestamp).
        keep = ["symbol", "timestamp"] + list(self.feature_names)
        df = df[keep].set_index(["symbol", "timestamp"]).sort_index()

        self._df = df

    def get(self, symbol: str, ts_utc: pd.Timestamp) -> Optional[Dict[str, object]]:
        """
        Exact-match lookup only. Returns dict(feature -> value) or None if not found.
        """
        if self._df is None:
            return None
        sym = str(symbol).upper()
        ts = pd.to_datetime(ts_utc, utc=True)
        key = (sym, ts)

        try:
            row = self._df.loc[key]
        except KeyError:
            return None

        # row is a Series of feature columns
        return {k: row[k] for k in self.feature_names}

    def available_keys(self) -> int:
        return 0 if self._df is None else int(len(self._df))

    def minmax_ts(self) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        if self._df is None or self._df.empty:
            return None
        idx = self._df.index.get_level_values("timestamp")
        return (pd.to_datetime(idx.min(), utc=True), pd.to_datetime(idx.max(), utc=True))
