from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeOutcome:
    ts: pd.Timestamp          # event timestamp (must be data-derived, UTC)
    win: int                  # 1 if pnl > 0 else 0
    pnl: float                # raw pnl (for debugging / optional future features)


class OnlinePerformanceState:
    """
    Persistent store of closed-trade outcomes for Sprint 2.4.

    Hard requirements:
    - Features must be computed strictly as-of decision_ts (no wall-clock).
    - Persistence across restarts.
    - Explicit shift(1): current/most-recent outcome is never included at decision time.

    Storage format: JSONL (one record per line)
      {"ts":"2026-01-30T12:35:00Z","win":1,"pnl":12.34}
    """

    def __init__(self, path: str, max_records: int = 2000, cold_start_winrate: float = float(np.nan)):
        self.path = Path(path)
        self.max_records = int(max_records)
        self.cold_start_winrate = float(cold_start_winrate)
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
        """
        Record a closed trade outcome at deterministic timestamp ts (UTC, data-derived).
        """
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
        """
        Compute recent performance features strictly as-of decision_ts with explicit shift(1).

        Steps:
        1) Filter outcomes with ts < decision_ts
        2) Apply shift(1): drop the most recent remaining outcome
        3) Compute:
           - recent_winrate_20: mean(last 20)
           - recent_winrate_50: mean(last 50)
           - recent_winrate_ewm_20: EWM mean(span=20)
        """
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

        # explicit shift(1)
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
