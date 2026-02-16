from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import math

@dataclass(frozen=True)
class TradeOutcome:
    ts: pd.Timestamp
    win: int
    pnl: float


class OnlinePerformanceState:


    def __init__(self, path: str, max_records: int = 2000, cold_start_winrate: float = 0.25):
        self.path = Path(path)
        self.max_records = int(max_records)

        self.cold_start_winrate = float(cold_start_winrate)
        if not math.isfinite(self.cold_start_winrate):
            self.cold_start_winrate = 0.25

        self._records: List[TradeOutcome] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()


    def _load(self) -> None:
        self._records = []
        if not self.path.exists():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    ts = pd.to_datetime(obj.get("ts"), utc=True, errors="coerce")
                    if pd.isna(ts):
                        continue
                    win = int(obj.get("win"))
                    pnl = float(obj.get("pnl"))
                    self._records.append(TradeOutcome(ts=ts, win=1 if win == 1 else 0, pnl=pnl))
                except Exception:
                    continue
            self._records.sort(key=lambda r: r.ts)
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records :]
        except Exception:
            self._records = []

    def _append_persist(self, rec: TradeOutcome) -> None:
        line = json.dumps(
            {"ts": rec.ts.isoformat().replace("+00:00", "Z"), "win": int(rec.win), "pnl": float(rec.pnl)},
            separators=(",", ":"),
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def record_trade_close(self, ts: pd.Timestamp, pnl: float) -> None:


        ts = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(ts):
            return
        pnl_f = float(pnl)
        win = 1 if pnl_f > 0.0 else 0

        rec = TradeOutcome(ts=ts, win=win, pnl=pnl_f)
        self._records.append(rec)
        self._records.sort(key=lambda r: r.ts)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]

        self._append_persist(rec)

    def features_asof(self, decision_ts: pd.Timestamp) -> Dict[str, float]:


        decision_ts = pd.to_datetime(decision_ts, utc=True, errors="coerce")
        if pd.isna(decision_ts):
            return {
                "recent_winrate_20": float(np.nan),
                "recent_winrate_50": float(np.nan),
                "recent_winrate_ewm_20": float(np.nan),
            }

        cold = float(self.cold_start_winrate)
        cold_feats = {
            "recent_winrate_20": cold,
            "recent_winrate_50": cold,
            "recent_winrate_ewm_20": cold,
        }

        prior = [r for r in self._records if r.ts < decision_ts]
        if len(prior) == 0:
            return cold_feats


        if len(prior) >= 1:
            prior = prior[:-1]
        if len(prior) == 0:
            return cold_feats

        wins = np.array([r.win for r in prior], dtype=float)

        def _mean_last(n: int) -> float:
            x = wins[-n:] if wins.size >= n else wins
            return float(np.mean(x)) if x.size > 0 else float(np.nan)

        wr20 = _mean_last(20)
        wr50 = _mean_last(50)

        try:
            s = pd.Series(wins)
            ewm = float(s.ewm(span=20, adjust=False).mean().iloc[-1])
        except Exception:
            ewm = float(np.nan)

        return {
            "recent_winrate_20": float(wr20),
            "recent_winrate_50": float(wr50),
            "recent_winrate_ewm_20": float(ewm),
        }
