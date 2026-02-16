from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .artifact_bundle import ArtifactBundle


def _ensure_utc_ts(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:

        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def _load_truth_df(path: Path, required_cols: Tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop(columns=["timestamp"])
        df.index = idx
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    df = df.sort_index()
    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(f"truth parquet missing column '{c}': {path}")
    if df.index.has_duplicates:
        raise RuntimeError(f"truth parquet has duplicate timestamps: {path}")
    return df


@dataclass(frozen=True)
class RegimeTruthStore:
    daily: pd.DataFrame
    markov4h: pd.DataFrame

    def daily_asof(self, decision_ts: Any) -> Optional[Dict[str, Any]]:
        ts = _ensure_utc_ts(decision_ts)


        if len(self.daily.index) == 0:
            return None
        if ts > self.daily.index[-1]:
            return None

        idx = self.daily.index.values
        pos = idx.searchsorted(ts.to_datetime64(), side="right") - 1
        if pos < 0:
            return None
        row = self.daily.iloc[int(pos)]
        code = row["regime_code_1d"]
        prob = row["vol_prob_low_1d"]
        if pd.isna(code) or pd.isna(prob):
            return None
        return {
            "regime_code_1d": int(code),
            "vol_prob_low_1d": float(prob),
        }

    def markov4h_asof(self, decision_ts: Any) -> Optional[Dict[str, Any]]:
        ts = _ensure_utc_ts(decision_ts)


        if len(self.markov4h.index) == 0:
            return None
        if ts > self.markov4h.index[-1]:
            return None

        idx = self.markov4h.index.values
        pos = idx.searchsorted(ts.to_datetime64(), side="right") - 1
        if pos < 0:
            return None
        row = self.markov4h.iloc[int(pos)]

        s = row["markov_state_4h"]
        p = row["markov_prob_up_4h"]
        if pd.isna(s) or pd.isna(p):
            return None
        return {
            "markov_state_4h": int(s),
            "markov_prob_up_4h": float(p),
        }


@lru_cache(maxsize=8)
def _load_store_cached(daily_path: str, markov_path: str) -> RegimeTruthStore:
    daily = _load_truth_df(Path(daily_path), ("regime_code_1d", "vol_prob_low_1d"))
    markov = _load_truth_df(Path(markov_path), ("markov_state_4h", "markov_prob_up_4h"))
    return RegimeTruthStore(daily=daily, markov4h=markov)


def load_regime_truth_store(bundle: ArtifactBundle) -> RegimeTruthStore:
    if bundle.regime_daily_truth_path is None or bundle.regime_markov4h_truth_path is None:
        raise RuntimeError("Regime truth artifacts are required but missing in bundle")
    return _load_store_cached(str(bundle.regime_daily_truth_path), str(bundle.regime_markov4h_truth_path))


def macro_regimes_asof(bundle: ArtifactBundle, decision_ts: Any) -> Dict[str, Any]:


    store = load_regime_truth_store(bundle)
    d = store.daily_asof(decision_ts)
    m = store.markov4h_asof(decision_ts)
    if d is None or m is None:
        raise RuntimeError(f"Regime truth missing as-of decision_ts={decision_ts}")
    out = {}
    out.update(d)
    out.update(m)
    return out
