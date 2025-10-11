"""Comprehensive live-trade performance audit (no CLI args needed).

Run:
    python -m research.reports.performance_audit

What it does:
- Auto-loads Postgres DSN and Bybit credentials from .env or environment.
- Pulls positions/fills/equity from Postgres.
- Computes core & advanced performance metrics:
    * Trades, Win%, Profit Factor, Expectancy, Payoff ratio
    * MAE/MFE distribution
    * Sharpe (daily), Max Drawdown, MAR, Ulcer Index
    * Longest win/loss streaks & session/day/hour expectancies
    * R-multiples (if risk_usd present)
- Meta-model analysis (if win-prob is stored on positions):
    * Brier score, reliability bins (calibration)
    * Threshold sweep (gate tuning): coverage, Win%, expectancy, net PnL
- Optional exchange reconciliation (if Bybit keys present).
- Saves a Markdown report (+ optional charts) in results/reports/.

Dependencies: pandas, numpy, asyncpg, ccxt[async], matplotlib (optional for charts).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from research.reports.perf_extras import build_extras_markdown

import asyncpg
import pandas as pd
import numpy as np

try:
    import ccxt.async_support as ccxt  # type: ignore
except Exception:  # pragma: no cover - if ccxt not installed, we degrade gracefully for reconciliation
    ccxt = None  # type: ignore

LOG = logging.getLogger("performance_audit")

# ────────────────────────── Paths & environment ──────────────────────────

REPO_ROOT_SENTINELS = {".env", "config.yaml", "apps", "strategies"}
DEFAULT_RESULTS_DIR = Path("results") / "reports"

def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for up in [p, *p.parents]:
        has = {f for f in os.listdir(up) if f in REPO_ROOT_SENTINELS}
        if has:
            return up
    return start

def _parse_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        env[k] = v
    return env

def _load_env_defaults() -> None:
    """Load .env if present and export into os.environ when missing."""
    here = Path(__file__).resolve()
    root = _find_repo_root(here.parent.parent.parent)  # research/reports/ → repo
    dotenv = root / ".env"
    values = _parse_dotenv(dotenv)
    for k, v in values.items():
        if k not in os.environ:
            os.environ[k] = v

def _autodetect_dsn() -> Optional[str]:
    return os.getenv("PG_DSN") or os.getenv("DATABASE_URL")

def _autodetect_bybit() -> tuple[Optional[str], Optional[str], bool]:
    key = os.getenv("BYBIT_API_KEY")
    secret = os.getenv("BYBIT_API_SECRET")
    testnet = str(os.getenv("BYBIT_TESTNET", "false")).lower() == "true"
    return key, secret, testnet

def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)

async def _load_symbol_maps(exchange):
    await exchange.load_markets()
    id_to_symbol = {m['id']: m['symbol'] for m in exchange.markets.values()}
    # also allow DB symbols like 'ETHUSDT' or '1000PEPEUSDT' to map by ID directly
    # and a few heuristics for spot vs linear vs coin-margined:
    def heuristics(db):
        db = db.upper().replace('/', '').replace(':', '')
        if db in id_to_symbol:            # exact market id
            return id_to_symbol[db]
        if db.endswith('USDT'):
            base = db[:-4]
            for cand in (f'{base}/USDT:USDT', f'{base}/USDT', f'{base}/USDT:USD'):
                if cand in exchange.markets:
                    return cand
        return None
    return heuristics


# ─────────────────────────────── Data classes ───────────────────────────────

@dataclasses.dataclass
class TradeMismatch:
    position_id: int
    symbol: str
    reason: str
    db_qty: float
    exchange_qty: float
    db_avg_price: Optional[float] = None
    exchange_avg_price: Optional[float] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

@dataclasses.dataclass
class PerformanceSummary:
    total_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    payoff_ratio: Optional[float]
    avg_mae: Optional[float]
    avg_mfe: Optional[float]
    sharpe_daily: Optional[float]
    max_drawdown_pct: Optional[float]
    ulcer_index: Optional[float]
    mar_ratio: Optional[float]
    cagr: Optional[float]
    avg_holding_minutes: Optional[float]
    fees_paid: float
    net_pnl: float
    brier: Optional[float] = None

# ─────────────────────────────── DB helpers ───────────────────────────────

async def _fetch_table(
    conn: asyncpg.Connection,
    table: str,
    *,
    where: str | None = None,
    params: Sequence[Any] = (),
) -> list[asyncpg.Record]:
    query = f"SELECT * FROM {table}"
    if where:
        query += f" WHERE {where}"
    if table == "positions":
        query += " ORDER BY opened_at ASC"
    elif table == "fills":
        query += " ORDER BY ts ASC"
    elif table == "equity_snapshots":
        query += " ORDER BY ts ASC"
    return await conn.fetch(query, *params)

async def load_datasets(
    dsn: str,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load positions, fills and equity snapshots from Postgres."""
    conn = await asyncpg.connect(dsn)
    try:
        params: list[Any] = []
        clauses: list[str] = ["status = 'CLOSED'"]
        if start:
            clauses.append("opened_at >= $%d" % (len(params) + 1))
            params.append(start)
        if end:
            clauses.append("opened_at <= $%d" % (len(params) + 1))
            params.append(end)
        where = " AND ".join(clauses)
        positions = await _fetch_table(conn, "positions", where=where, params=params)

        fills_where = None
        fill_params: list[Any] = []
        if positions:
            pos_ids = [r["id"] for r in positions]
            fills_where = "position_id = ANY($1::int[])"
            fill_params = [pos_ids]
        fills = await _fetch_table(conn, "fills", where=fills_where, params=fill_params)

        eq_params: list[Any] = []
        eq_where = None
        if start and end:
            eq_where = "ts BETWEEN $1 AND $2"
            eq_params = [start - timedelta(days=1), end + timedelta(days=1)]
        equity = await _fetch_table(conn, "equity_snapshots", where=eq_where, params=eq_params)
    finally:
        await conn.close()

    def _to_df(rows: Iterable[asyncpg.Record]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        for col in ("opened_at", "closed_at", "exit_deadline", "ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        return df

    return _to_df(positions), _to_df(fills), _to_df(equity)

# ─────────────────────────────── Metrics ───────────────────────────────

def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1.0
    return dd.min()

def _ulcer_index(equity_curve: pd.Series) -> float:
    """Ulcer Index: sqrt(mean(drawdown_pct^2))."""
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    dd_pct = (equity_curve / running_max - 1.0) * 100.0
    return float(np.sqrt(np.mean(np.square(dd_pct.fillna(0.0)))))

def _cagr(equity_curve: pd.Series, index: pd.DatetimeIndex) -> float:
    if equity_curve.empty:
        return float("nan")
    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    if start <= 0 or end <= 0:
        return float("nan")
    days = max(1.0, (index[-1] - index[0]).total_seconds() / 86400.0)
    years = days / 365.0
    return (end / start) ** (1.0 / years) - 1.0

def _payoff_ratio(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    wins = df[df["net_pnl"] > 0]["net_pnl"]
    losses = -df[df["net_pnl"] < 0]["net_pnl"]
    if wins.empty or losses.empty:
        return None
    return float(wins.mean() / losses.mean()) if losses.mean() > 0 else None

def _streaks(pnl: pd.Series) -> tuple[int, int]:
    longest_win = longest_loss = cur = 0
    last_sign = 0
    for x in pnl:
        s = 1 if x > 0 else (-1 if x < 0 else 0)
        if s == 0:
            cur = 0
            last_sign = 0
            continue
        if s == last_sign:
            cur += 1
        else:
            cur = 1
            last_sign = s
        if s > 0:
            longest_win = max(longest_win, cur)
        else:
            longest_loss = max(longest_loss, cur)
    return longest_win, longest_loss

def compute_performance_summary(
    positions: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    starting_equity: float = 1000.0,
) -> PerformanceSummary:
    if positions.empty:
        raise ValueError("No closed positions found – nothing to analyse.")

    df = positions.copy()
    for col in ("pnl", "pnl_pct", "mae_usd", "mfe_usd", "fees_paid", "holding_minutes", "risk_usd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["fees_paid"] = pd.to_numeric(df.get("fees_paid", 0.0), errors="coerce").fillna(0.0)
    df["net_pnl"] = pd.to_numeric(df.get("pnl", 0.0), errors="coerce") - df["fees_paid"]

    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] < 0]

    gross_profit = wins["net_pnl"].sum()
    gross_loss = -losses["net_pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf
    win_rate = (len(wins) / len(df)) * 100 if len(df) else float("nan")
    expectancy = df["net_pnl"].mean()
    payoff_ratio = _payoff_ratio(df)

    avg_mae = df["mae_usd"].dropna().mean() if "mae_usd" in df else None
    avg_mfe = df["mfe_usd"].dropna().mean() if "mfe_usd" in df else None
    avg_hold = df["holding_minutes"].dropna().mean() if "holding_minutes" in df else None

    # Equity curve
    if not equity.empty and "equity" in equity.columns:
        eq = equity.copy()
        eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
        eq.dropna(subset=["equity"], inplace=True)
        eq.sort_values("ts", inplace=True)
        eq.set_index("ts", inplace=True)
        eq_curve = eq["equity"]
    else:
        df.sort_values("closed_at", inplace=True)
        eq_curve = starting_equity + df["net_pnl"].cumsum()
        eq_curve.index = df["closed_at"].fillna(pd.Timestamp.utcnow())

    # Sharpe
    sharpe = None
    if len(eq_curve) >= 3:
        daily = eq_curve.resample("1D").last().dropna()
        if len(daily) >= 3:
            rets = daily.pct_change().dropna()
            if not rets.empty:
                sd = rets.std(ddof=1)
                sharpe = (rets.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0

    max_dd = _max_drawdown(eq_curve)
    ulcer = _ulcer_index(eq_curve)
    cagr = _cagr(eq_curve, eq_curve.index)
    mar = None
    if cagr == cagr and max_dd == max_dd and max_dd < 0:
        mar = (cagr) / abs(max_dd)

    return PerformanceSummary(
        total_trades=len(df),
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=float(expectancy),
        payoff_ratio=payoff_ratio,
        avg_mae=float(avg_mae) if avg_mae is not None else None,
        avg_mfe=float(avg_mfe) if avg_mfe is not None else None,
        sharpe_daily=float(sharpe) if sharpe is not None else None,
        max_drawdown_pct=max_dd * 100 if max_dd == max_dd else None,
        ulcer_index=ulcer if ulcer == ulcer else None,
        mar_ratio=mar if mar is not None else None,
        cagr=cagr * 100 if cagr == cagr else None,
        avg_holding_minutes=float(avg_hold) if avg_hold is not None else None,
        fees_paid=float(df["fees_paid"].sum()),
        net_pnl=float(df["net_pnl"].sum()),
    )

def mae_mfe_distribution(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "mae_usd" in df:
        mae = pd.to_numeric(df["mae_usd"], errors="coerce").dropna()
        if not mae.empty:
            out["mae"] = {"mean": float(mae.mean()), "median": float(mae.median()), "p95": float(mae.quantile(0.95))}
    if "mfe_usd" in df:
        mfe = pd.to_numeric(df["mfe_usd"], errors="coerce").dropna()
        if not mfe.empty:
            out["mfe"] = {"mean": float(mfe.mean()), "median": float(mfe.median()), "p95": float(mfe.quantile(0.95))}
    return out

# ───────────────── Stop/TP consistency (robust, MAE/MFE/R-based) ─────────────────

def _num(s: pd.Series, default=np.nan) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)

def _approx_eq(a: pd.Series, b: pd.Series, rel=0.12) -> pd.Series:
    """Vectorized ~relative equality: |a-b| <= rel * max(1, |b|)."""
    return (a - b).abs() <= rel * (b.abs().clip(lower=1.0))

def exit_consistency_audit(
    df: pd.DataFrame,
    *,
    tp_r_guess: float = 8.0,       # your current TP ≈ +8R
    stop_r_guess: float = -1.0,    # stop at -1R
    rel_tol: float = 0.20          # tolerance for noisy MAE/MFE, fees, slippage
) -> dict[str, Any]:
    """
    Cross-check recorded exit reasons against what MAE/MFE/R imply.
    Requires: net_pnl, mae_usd, mfe_usd, risk_usd (or r_multiple), optionally exit_reason.
    Never assumes intra-trade resizing.
    """

    if df.empty:
        return {"summary": {}, "mismatches": pd.DataFrame()}

    x = df.copy()
    # Coerce numerics
    for c in ("net_pnl", "mae_usd", "mfe_usd", "risk_usd", "r_multiple"):
        if c in x.columns:
            x[c] = _num(x[c])

    # Prefer r_multiple if present; else compute from risk_usd
    if "r_multiple" in x and x["r_multiple"].notna().any():
        r = x["r_multiple"]
    else:
        if "risk_usd" in x and x["risk_usd"].abs().gt(0).any():
            r = x["net_pnl"] / x["risk_usd"].abs().replace(0, np.nan)
        else:
            r = pd.Series(np.nan, index=x.index)

    # If we have usable risk_usd, treat it as the *initial stop distance in USD*
    has_risk = "risk_usd" in x and x["risk_usd"].abs().gt(0).any()
    stop_usd = x["risk_usd"].abs() if has_risk else pd.Series(np.nan, index=x.index)

    # Recorded reason (normalize)
    reason_raw = x.get("exit_reason", pd.Series("", index=x.index)).astype(str).str.upper().str.strip()
    reason_rec = reason_raw.replace({
        "STOP": "STOP", "SL": "STOP", "STOP_LOSS": "STOP",
        "TP": "TP", "TAKE_PROFIT": "TP",
        "TIME": "TIME", "TL": "TIME", "TIME_LIMIT": "TIME"
    })

    # Inferred reason by R
    infer_by_r = pd.Series("OTHER", index=x.index)
    infer_by_r = np.where(r <= stop_r_guess * (1 - rel_tol), "STOP", infer_by_r)
    infer_by_r = np.where(r >= tp_r_guess   * (1 - rel_tol), "TP",   infer_by_r)
    infer_by_r = pd.Series(infer_by_r, index=x.index)

    # Inferred “stop proximity” by MAE vs initial stop distance
    # If a loser and MAE ~ stop_usd → likely STOP. If loser and MAE << stop_usd → suspect TIME/other.
    loser = x["net_pnl"] < 0
    near_stop = pd.Series(False, index=x.index)
    far_from_stop = pd.Series(False, index=x.index)
    if has_risk and "mae_usd" in x:
        near_stop = loser & _approx_eq(x["mae_usd"].abs(), stop_usd, rel=max(rel_tol, 0.15))
        far_from_stop = loser & (x["mae_usd"].abs() < 0.5 * stop_usd)

    # Final inferred reason combining signals
    reason_inf = pd.Series("OTHER", index=x.index)
    reason_inf = np.where(infer_by_r == "STOP", "STOP", reason_inf)
    reason_inf = np.where(infer_by_r == "TP", "TP", reason_inf)
    reason_inf = np.where((infer_by_r == "OTHER") & near_stop, "STOP", reason_inf)
    reason_inf = np.where((infer_by_r == "OTHER") & (loser & far_from_stop), "TIME", reason_inf)
    reason_inf = pd.Series(reason_inf, index=x.index)

    # Metrics akin to your diagnostics (guarded)
    winners = x["net_pnl"] > 0
    losers  = loser

    stop_adequacy = np.nan
    stop_waste    = np.nan
    tp_harvest    = np.nan
    mae_p80_win   = np.nan
    mfe_p50_win   = np.nan
    mfe_p70_win   = np.nan

    if has_risk and "mae_usd" in x and winners.any():
        # Share of winners whose MAE would have *exceeded* the initial stop distance (bad, should be ~0)
        stop_adequacy = float((winners & (x["mae_usd"].abs() >= 1.0 * stop_usd * (1 - rel_tol))).sum() / winners.sum() * 100)

    if has_risk and "mae_usd" in x and losers.any():
        # Share of losers with MAE < 0.5×stop (likely not a true stop-out if reason says STOP)
        stop_waste = float((losers & (x["mae_usd"].abs() < 0.5 * stop_usd * (1 + rel_tol))).sum() / losers.sum() * 100)

    if "mfe_usd" in x and winners.any():
        with np.errstate(divide='ignore', invalid='ignore'):
            capture = x.loc[winners, "net_pnl"] / x.loc[winners, "mfe_usd"].replace(0, np.nan).abs()
        tp_harvest  = float(np.nanmedian(capture.dropna())) * 100 if capture.notna().any() else np.nan

    if "mae_usd" in x and winners.any():
        mae_p80_win = float(x.loc[winners, "mae_usd"].abs().quantile(0.80))
    if "mfe_usd" in x and winners.any():
        mfe_p50_win = float(x.loc[winners, "mfe_usd"].abs().quantile(0.50))
        mfe_p70_win = float(x.loc[winners, "mfe_usd"].abs().quantile(0.70))

    # Mismatch table
    mismatch = pd.DataFrame({
        "id": x.get("id"),
        "symbol": x.get("symbol"),
        "r_multiple": r,
        "risk_usd": stop_usd if has_risk else np.nan,
        "mae_usd": x.get("mae_usd"),
        "mfe_usd": x.get("mfe_usd"),
        "net_pnl": x.get("net_pnl"),
        "reason_recorded": reason_rec,
        "reason_inferred": reason_inf,
    })

    # Flag “improbable STOP” (recorded STOP but far from stop), and “improbable TIME” (recorded TIME but near stop)
    improbable_stop = (mismatch["reason_recorded"] == "STOP") & far_from_stop
    improbable_time = (mismatch["reason_recorded"] == "TIME") & near_stop
    probable_tp_but_not = (mismatch["reason_recorded"] != "TP") & (r >= tp_r_guess * (1 - rel_tol))

    mismatch["flag"] = ""
    mismatch.loc[improbable_stop, "flag"] = "REC=STOP but MAE << stop (suspect)"
    mismatch.loc[improbable_time, "flag"] = "REC=TIME but MAE ~ stop (likely STOP)"
    mismatch.loc[probable_tp_but_not, "flag"] = "R >> TP threshold; REC not TP (check)"

    mismatches = mismatch[mismatch["flag"] != ""].copy().sort_values("id").reset_index(drop=True)

    return {
        "summary": {
            "stop_adequacy_pct": stop_adequacy,     # winners that would have tripped initial stop (want ~0%)
            "stop_waste_pct": stop_waste,           # losers that never got close to stop (should be low if most are SL)
            "tp_harvest_pct": tp_harvest,           # median share of MFE captured on winners
            "mae_p80_winners": mae_p80_win,
            "mfe_p50_winners": mfe_p50_win,
            "mfe_p70_winners": mfe_p70_win,
        },
        "mismatches": mismatches,
    }

def segment_performance(df: pd.DataFrame, column: str, min_trades: int = 10) -> list[tuple[str, float, float, int]]:
    if column not in df.columns:
        return []
    grouped = df.dropna(subset=[column]).groupby(column)
    stats: list[tuple[str, float, float, int]] = []
    for key, sub in grouped:
        pnl = pd.to_numeric(sub.get("net_pnl", sub.get("pnl", 0.0)), errors="coerce").dropna()
        if len(pnl) < min_trades or pnl.empty:
            continue
        stats.append((str(key), float(pnl.mean()), float(pnl.sum()), int(len(pnl))))
    stats.sort(key=lambda x: x[1])  # ascending by expectancy
    return stats

# ───────────── Exit reason inference & audit (robust) ─────────────

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _infer_stop_usd(row: pd.Series) -> float:
    """
    Compute initial stop distance in USD. Prefer risk_usd; else derive from prices*size.
    Returns NaN if not computable.
    """
    r = _safe_float(row.get("risk_usd"))
    if r == r and r > 0:
        return r

    entry = _safe_float(row.get("entry_price"))
    stop  = _safe_float(row.get("stop_price"))
    size  = _safe_float(row.get("size"))
    if all(x == x for x in (entry, stop, size)) and abs(size) > 0 and entry > 0 and stop > 0:
        return abs(entry - stop) * abs(size)

    return float("nan")

def _infer_tp_usd(row: pd.Series) -> float:
    """
    Try to compute TP distance in USD if a static take-profit price was used.
    Returns NaN if not computable.
    """
    tp    = _safe_float(row.get("tp_price"))
    entry = _safe_float(row.get("entry_price"))
    size  = _safe_float(row.get("size"))
    if all(x == x for x in (tp, entry, size)) and abs(size) > 0 and tp > 0 and entry > 0:
        return abs(tp - entry) * abs(size)
    return float("nan")

def _near(a: float, b: float, rel_tol: float = 0.15, abs_tol: float = 5e-3) -> bool:
    """is a ≈ b with relative/absolute tolerance (handles small-dollar alts)."""
    if not (a == a and b == b):
        return False
    if abs(a - b) <= abs_tol:
        return True
    denom = max(abs(a), abs(b), abs_tol)
    return abs(a - b) / denom <= rel_tol

def _infer_reason(row: pd.Series, *, rel_tol=0.15, abs_tol=5e-3, deadline_tol_min=3.0) -> str:
    """
    Infer exit reason using MAE/MFE, stop/tp distances, and exit_deadline proximity.

    Returns: "SL" | "TP" | "TL" | "MANUAL" | "UNKNOWN"
    """
    pnl      = _safe_float(row.get("net_pnl", row.get("pnl")))
    mae      = _safe_float(row.get("mae_usd"))
    mfe      = _safe_float(row.get("mfe_usd"))
    stop_usd = _infer_stop_usd(row)
    tp_usd   = _infer_tp_usd(row)

    closed_at     = row.get("closed_at")
    exit_deadline = row.get("exit_deadline")

    # 1) Time-limit (deadline) proximity
    if isinstance(closed_at, pd.Timestamp) and isinstance(exit_deadline, pd.Timestamp):
        dt_min = abs((closed_at - exit_deadline).total_seconds()) / 60.0
        if dt_min <= deadline_tol_min:
            return "TL"

    # 2) Stop-loss (for losers) — loser MAE ≈ stop distance, or |pnl| ≈ stop_usd
    if pnl == pnl and pnl <= 0:
        if stop_usd == stop_usd:
            if (mae == mae and _near(mae, stop_usd, rel_tol=rel_tol, abs_tol=abs_tol)) or \
               _near(abs(pnl), stop_usd, rel_tol=rel_tol, abs_tol=abs_tol):
                return "SL"

    # 3) Take-profit (for winners) — MFE or PnL near TP distance if known, else strong harvest
    if pnl == pnl and pnl > 0:
        if tp_usd == tp_usd:
            if (mfe == mfe and _near(mfe, tp_usd, rel_tol=rel_tol, abs_tol=abs_tol)) or \
               _near(pnl, tp_usd * 0.9, rel_tol=rel_tol, abs_tol=abs_tol):
                return "TP"
        # If we don't know TP distance but harvest is very high (e.g., captured > 70% of MFE)
        if mfe == mfe and mfe > 0:
            harvest = pnl / mfe
            if harvest >= 0.7:
                return "TP"

    # 4) Manual / Unknown fallback
    # If neither SL/TP/TL criteria match, but pnl ≈ 0 (flat near break-even), treat as MANUAL flatten
    if pnl == pnl and abs(pnl) <= abs_tol:
        return "MANUAL"

    return "UNKNOWN"

def audit_exit_reasons(
    df: pd.DataFrame,
    *,
    rel_tol: float = 0.15,
    abs_tol: float = 5e-3,
    deadline_tol_min: float = 3.0,
    max_rows: int = 25,
) -> dict[str, Any]:
    """
    Build recorded distribution, inferred distribution, confusion (recorded vs inferred),
    suspicious rows where they disagree or inference is UNKNOWN, and coverage counters.
    """
    if df.empty:
        return {
            "recorded_dist": pd.Series(dtype=int),
            "inferred_dist": pd.Series(dtype=int),
            "confusion": pd.DataFrame(),
            "suspicious": pd.DataFrame(),
            "coverage": {},
        }

    work = df.copy()
    # Prepare required numeric fields
    for col in ("pnl", "net_pnl", "mae_usd", "mfe_usd", "risk_usd", "entry_price", "stop_price", "tp_price", "size"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    # Inference
    work["exit_reason_inferred"] = work.apply(
        lambda r: _infer_reason(r, rel_tol=rel_tol, abs_tol=abs_tol, deadline_tol_min=deadline_tol_min), axis=1
    )

    # Recorded reason (if present)
    recorded_col = None
    for c in work.columns:
        if c.lower() in {"exit_reason", "reason", "close_reason"}:
            recorded_col = c
            break
    if recorded_col is None:
        work["exit_reason_recorded"] = pd.Series(["(missing)"] * len(work), index=work.index)
    else:
        work["exit_reason_recorded"] = work[recorded_col].astype(str).fillna("(missing)").str.upper()

    # Distributions
    recorded_dist = work["exit_reason_recorded"].value_counts(dropna=False).sort_index()
    inferred_dist = work["exit_reason_inferred"].value_counts(dropna=False).sort_index()

    # Confusion matrix
    confusion = (
        work.pivot_table(
            index="exit_reason_recorded",
            columns="exit_reason_inferred",
            values="id" if "id" in work.columns else "symbol",
            aggfunc="count",
            fill_value=0,
        )
        .sort_index()
        .sort_index(axis=1)
    )

    # Suspicious rows
    sus = work[
        (work["exit_reason_recorded"] != "(missing)") &
        (work["exit_reason_recorded"] != work["exit_reason_inferred"])
        | (work["exit_reason_inferred"].isin(["UNKNOWN"]))
    ].copy()

    keep_cols = [
        c for c in [
            "id", "symbol", "opened_at", "closed_at",
            "net_pnl", "mae_usd", "mfe_usd", "risk_usd",
            "entry_price", "stop_price", "tp_price", "size",
            "exit_deadline",
            "exit_reason_recorded", "exit_reason_inferred",
        ] if c in work.columns
    ]
    suspicious = sus[keep_cols].head(max_rows)

    # Coverage counters for diagnostics sections
    cov = {
        "n_total": int(len(work)),
        "n_have_mae": int(work["mae_usd"].notna().sum()) if "mae_usd" in work else 0,
        "n_have_mfe": int(work["mfe_usd"].notna().sum()) if "mfe_usd" in work else 0,
        "n_have_risk_or_stop": int(
            ((work["risk_usd"].notna() & (work["risk_usd"] > 0)) |
             (work[["entry_price","stop_price","size"]].notna().all(axis=1) if set(["entry_price","stop_price","size"]).issubset(work.columns) else False)
            ).sum()
        ) if "risk_usd" in work.columns else 0,
    }

    return {
        "recorded_dist": recorded_dist,
        "inferred_dist": inferred_dist,
        "confusion": confusion,
        "suspicious": suspicious,
        "coverage": cov,
    }


# ───────────────────────── Meta-model diagnostics ─────────────────────────

def _find_winprob_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = c.lower()
        if "win" in cl and "prob" in cl:
            return c
        if cl in {"win_probability", "winprob", "meta_p", "p_win"}:
            return c
    return None

def calibration_bins(
    probs: pd.Series, y: pd.Series, n_bins: int = 10
) -> tuple[pd.DataFrame, float]:
    """Return reliability table and Brier score."""
    probs = pd.to_numeric(probs, errors="coerce").clip(0, 1)
    y = (y > 0).astype(int)  # win if net_pnl > 0
    brier = float(((probs - y) ** 2).mean())

    df = pd.DataFrame({"p": probs, "y": y}).dropna()
    if df.empty:
        return pd.DataFrame(), brier

    # Quantile bins; duplicates='drop' keeps the bin count stable on ties
    df["bin"] = pd.qcut(df["p"], q=min(n_bins, max(2, df["p"].nunique())), duplicates="drop")

    # Future-proof: pandas default of observed will change → set it explicitly
    tab = df.groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"),
        frac_positive=("y", "mean"),
        n=("y", "size"),
    ).reset_index(drop=True)
    return tab, brier


def threshold_sweep(df: pd.DataFrame, prob_col: str, thresholds: Sequence[float]) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        sub = df[df[prob_col] >= t]
        n = len(sub)
        if n == 0:
            rows.append({"thr": t, "n": 0, "win_rate": np.nan, "exp": np.nan, "net_pnl": 0.0})
            continue
        pnl = pd.to_numeric(sub["net_pnl"], errors="coerce")
        wins = (pnl > 0).sum()
        rows.append(
            {
                "thr": float(t),
                "n": int(n),
                "win_rate": float(wins / n * 100.0),
                "exp": float(pnl.mean()),
                "net_pnl": float(pnl.sum()),
            }
        )
    return pd.DataFrame(rows)

# ───────────────────── Exchange reconciliation (optional) ───────────────────

async def _init_exchange(key: str, secret: str, testnet: bool):
    if ccxt is None:
        raise RuntimeError("ccxt not installed; cannot reconcile.")
    exchange = ccxt.bybit(
        {
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
    )
    if testnet:
        exchange.set_sandbox_mode(True)
    return exchange

async def fetch_trades_for_symbols(
    exchange,
    symbols: Sequence[str],
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    since_ms = int(since.timestamp() * 1000) if since else None
    until_ms = int(until.timestamp() * 1000) if until else None

    for symbol in symbols:
        ms = since_ms
        trades: list[dict[str, Any]] = []
        while True:
            try:
                batch = await exchange.fetch_my_trades(symbol, since=ms, limit=200)
            except Exception as exc:  # pragma: no cover - network/ccxt errors
                LOG.warning("fetch_my_trades failed for %s: %s", symbol, exc)
                break
            if not batch:
                break
            batch = [t for t in batch if t.get("timestamp")]
            batch.sort(key=lambda t: int(t["timestamp"]))
            trades.extend(batch)
            last_ts = int(batch[-1]["timestamp"])
            if until_ms and last_ts > until_ms:
                trades = [t for t in trades if t["timestamp"] <= until_ms]
                break
            if len(batch) < 200:
                break
            ms = last_ts + 1
            await asyncio.sleep(0.8)
        results[symbol] = trades
    return results

def _aggregate_trades(trades: list[dict[str, Any]], side: str, start: datetime, end: datetime) -> tuple[float, Optional[float]]:
    if not trades:
        return 0.0, None
    total_qty = 0.0
    total_notional = 0.0
    side = side.lower()
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    for t in trades:
        ts = int(t.get("timestamp") or 0)
        if ts < start_ms or ts > end_ms:
            continue
        if str(t.get("side", "")).lower() != side:
            continue
        qty = float(t.get("amount") or 0.0)
        price = float(t.get("price") or 0.0)
        if qty <= 0 or price <= 0:
            continue
        total_qty += qty
        total_notional += qty * price
    if total_qty <= 0:
        return 0.0, None
    return total_qty, total_notional / total_qty

def reconcile_positions(positions, fills, trades_by_symbol, *, tolerance_qty=1e-6):
    mismatches = []
    fills = fills.copy()
    if not fills.empty:
        fills['ts'] = pd.to_datetime(fills['ts'], utc=True, errors='coerce')
        fills['qty'] = pd.to_numeric(fills['qty'], errors='coerce')
        fills['price'] = pd.to_numeric(fills['price'], errors='coerce')

    for _, row in positions.iterrows():
        pid = int(row['id']); symbol = str(row['symbol'])
        # skip symbols we failed or skipped
        sym_trades = trades_by_symbol.get(symbol, None)
        if sym_trades is None:
            # skipped/unsupported/delisted – report elsewhere, do not count as mismatch
            continue

        opened_at = row.get('opened_at') or row.get('created_at')
        closed_at  = row.get('closed_at') or opened_at
        if not isinstance(opened_at, pd.Timestamp):
            continue
        if not isinstance(closed_at, pd.Timestamp):
            closed_at = opened_at

        # widen entry/exit windows using DB fills if present
        ff = fills[fills['position_id'] == pid]
        if not ff.empty:
            entry_start = ff['ts'].min() - pd.Timedelta(minutes=5)
            entry_end   = ff['ts'].max() + pd.Timedelta(minutes=5)
        else:
            entry_start = opened_at - pd.Timedelta(minutes=5)
            entry_end   = opened_at + pd.Timedelta(minutes=30)

        exit_start = opened_at
        exit_end   = closed_at + pd.Timedelta(minutes=30)

        entry_side = 'buy' if str(row.get('side','short')).lower() == 'long' else 'sell'
        exit_side  = 'sell' if entry_side == 'buy' else 'buy'

        eqty, eavg = _aggregate_trades(sym_trades or [], entry_side, entry_start, entry_end)
        xqty,  xavg = _aggregate_trades(sym_trades or [], exit_side,  exit_start,  exit_end)

        db_size = float(row.get("size") or 0.0)
        if abs(abs(eqty) - abs(db_size)) > tolerance_qty:
            mismatches.append(
                TradeMismatch(
                    position_id=pid,
                    symbol=symbol,
                    reason="ENTRY_SIZE_MISMATCH",
                    db_qty=db_size,
                    exchange_qty=eqty,
                    db_avg_price=float(row.get("entry_price") or 0.0),
                    exchange_avg_price=eavg,
                    opened_at=opened_at.to_pydatetime(),
                    closed_at=closed_at.to_pydatetime() if isinstance(closed_at, pd.Timestamp) else None,
                )
            )


        exit_fills = fills[fills["position_id"] == pid]
        db_exit_qty = exit_fills["qty"].abs().sum() if not exit_fills.empty else abs(db_size)
        db_exit_avg = (
            (exit_fills["qty"].abs() * exit_fills["price"]).sum() / db_exit_qty
            if (not exit_fills.empty and db_exit_qty > 0)
            else None
        )

        if abs(xqty - db_exit_qty) > tolerance_qty:
            mismatches.append(
                TradeMismatch(
                    position_id=pid,
                    symbol=symbol,
                    reason="EXIT_SIZE_MISMATCH",
                    db_qty=db_exit_qty,
                    exchange_qty=xqty,
                    db_avg_price=db_exit_avg,
                    exchange_avg_price=xavg,
                    opened_at=opened_at.to_pydatetime(),
                    closed_at=closed_at.to_pydatetime() if isinstance(closed_at, pd.Timestamp) else None,
                )
            )

    return mismatches

# ─────────────────────────── Reporting utilities ───────────────────────────

def _fmt_money(x: Optional[float]) -> str:
    if x is None or not (x == x):
        return "n/a"
    return f"{x:+.2f}"

def _fmt_pct(x: Optional[float]) -> str:
    if x is None or not (x == x):
        return "n/a"
    return f"{x:.2f}%"

def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def _maybe_make_plots(out_dir: Path, equity_curve: pd.Series, dd_curve: pd.Series,
                      calib: Optional[pd.DataFrame], sweep: Optional[pd.DataFrame]) -> dict[str, str]:
    saved: dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt  # optional
    except Exception:
        return saved

    if not equity_curve.empty:
        fig = plt.figure()
        equity_curve.plot()
        plt.title("Equity Curve")
        plt.xlabel("Time"); plt.ylabel("Equity")
        f = out_dir / "equity_curve.png"; fig.savefig(f, bbox_inches="tight"); plt.close(fig)
        saved["equity_curve"] = str(f)

        fig = plt.figure()
        dd_curve.plot()
        plt.title("Underwater (Drawdown)")
        plt.xlabel("Time"); plt.ylabel("Drawdown (%)")
        f = out_dir / "drawdown.png"; fig.savefig(f, bbox_inches="tight"); plt.close(fig)
        saved["drawdown"] = str(f)

    if calib is not None and not calib.empty:
        fig = plt.figure()
        x = calib["mean_pred"]; y = calib["frac_positive"]
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.scatter(x, y)
        plt.title("Reliability (Calibration)")
        plt.xlabel("Predicted prob"); plt.ylabel("Observed win rate")
        f = out_dir / "calibration.png"; fig.savefig(f, bbox_inches="tight"); plt.close(fig)
        saved["calibration"] = str(f)

    if sweep is not None and not sweep.empty:
        fig = plt.figure()
        plt.plot(sweep["thr"], sweep["win_rate"])
        plt.title("Threshold Sweep — Win%")
        plt.xlabel("Threshold"); plt.ylabel("Win%")
        f = out_dir / "threshold_winrate.png"; fig.savefig(f, bbox_inches="tight"); plt.close(fig)
        saved["threshold_winrate"] = str(f)

    return saved

# ─────────────────────────────── CLI (optional) ───────────────────────────────

def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Comprehensive live-trade performance audit", add_help=False)
    ap.add_argument("--dsn")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--exchange", default="bybit")
    ap.add_argument("--bybit-key")
    ap.add_argument("--bybit-secret")
    ap.add_argument("--bybit-testnet", action="store_true")
    ap.add_argument("--starting-equity", type=float)
    ap.add_argument("--skip-exchange", action="store_true")
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--verbose", action="store_true")
    # no error if unknown args → zero-argument friendly
    known, _ = ap.parse_known_args()
    return known

# ─────────────────────────────── Main logic ───────────────────────────────

async def main_async():
    # Load .env → os.environ first so defaults are populated.
    _load_env_defaults()

    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Autodetect configuration when args missing
    dsn = args.dsn or _autodetect_dsn()
    if not dsn:
        raise RuntimeError("Postgres DSN missing – set PG_DSN / DATABASE_URL in .env or environment.")

    start = _parse_dt(args.start)
    end = _parse_dt(args.end)
    # Load initial datasets (we may widen the window automatically)
    LOG.info("Loading datasets from Postgres…")
    positions_all, fills_all, equity_all = await load_datasets(dsn, start=start, end=end)

    if positions_all.empty:
        print("No closed trades found in DB (for the selected window).")
        return

    # If user provided no start/end → auto window from actual trades
    if not start:
        start = pd.to_datetime(positions_all["opened_at"].min()).to_pydatetime()
    if not end:
        end = pd.to_datetime(positions_all["closed_at"].max()).to_pydatetime() + timedelta(hours=1)

    # Re-load equity to cover full range with margin
    positions, fills, equity = positions_all, fills_all, equity_all
    if equity.empty or equity["ts"].min() > start or equity["ts"].max() < end:
        equity = (await load_datasets(dsn, start=start, end=end))[2]

    # Prepare positions
    positions_aug = positions.copy()
    positions_aug["fees_paid"] = pd.to_numeric(positions_aug.get("fees_paid", 0.0), errors="coerce").fillna(0.0)
    positions_aug["pnl"] = pd.to_numeric(positions_aug.get("pnl", 0.0), errors="coerce").fillna(0.0)
    positions_aug["net_pnl"] = positions_aug["pnl"] - positions_aug["fees_paid"]

    # Derived columns if helpful
    if "opened_at" in positions_aug and "closed_at" in positions_aug and "holding_minutes" not in positions_aug:
        dt = (positions_aug["closed_at"] - positions_aug["opened_at"]).dt.total_seconds() / 60.0
        positions_aug["holding_minutes"] = dt

    if "risk_usd" in positions_aug:
        positions_aug["r_multiple"] = np.where(
            positions_aug["risk_usd"].abs() > 0, positions_aug["net_pnl"] / positions_aug["risk_usd"].abs(), np.nan
        )

    # Equity curve (for plots)
    if not equity.empty and "equity" in equity.columns:
        eq = equity.copy()
        eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
        eq.dropna(subset=["equity"], inplace=True)
        eq.sort_values("ts", inplace=True)
        eq.set_index("ts", inplace=True)
        eq_curve = eq["equity"]
    else:
        s0 = float(os.getenv("STARTING_EQUITY", "1000")) if args.starting_equity is None else args.starting_equity
        positions_sorted = positions_aug.sort_values("closed_at")
        eq_curve = s0 + positions_sorted["net_pnl"].cumsum()
        eq_curve.index = positions_sorted["closed_at"].fillna(pd.Timestamp.utcnow())
    dd_curve = (eq_curve / eq_curve.cummax() - 1.0) * 100.0

    LOG.info("Computing performance analytics…")
    starting_equity = float(os.getenv("STARTING_EQUITY", "1000")) if args.starting_equity is None else args.starting_equity
    summary = compute_performance_summary(positions_aug, equity, starting_equity=starting_equity)
    dist = mae_mfe_distribution(positions_aug)
    exit_audit = exit_consistency_audit(positions_aug, tp_r_guess=8.0, stop_r_guess=-1.0, rel_tol=0.20)
    
    session_segs = segment_performance(positions_aug, "session_tag_at_entry")
    dow_segs = segment_performance(positions_aug, "day_of_week_at_entry")
    # Hour-of-day from opened_at if not present
    if "hour_at_entry" in positions_aug.columns:
        hour_segs = segment_performance(positions_aug, "hour_at_entry")
    elif "opened_at" in positions_aug.columns:
        positions_aug["hour_at_entry"] = positions_aug["opened_at"].dt.hour
        hour_segs = segment_performance(positions_aug, "hour_at_entry")
    else:
        hour_segs = []

    # Streaks
    longest_win, longest_loss = _streaks(positions_aug["net_pnl"].values)

    # Meta-model calibration & threshold sweep (if we find a prob column)
    prob_col = _find_winprob_column(positions_aug)
    calib_tab = None
    brier = None
    sweep_tab = None
    suggested_thr = None
    if prob_col:
        calib_tab, brier = calibration_bins(positions_aug[prob_col], positions_aug["net_pnl"])
        summary.brier = brier
        # Sweep thresholds 0.50→0.90
        thresholds = np.round(np.linspace(0.50, 0.90, 9), 2)
        sweep_tab = threshold_sweep(positions_aug, prob_col, thresholds)
        # Suggest threshold that maximizes net_pnl with n≥20 (tunable)
        cand = sweep_tab[sweep_tab["n"] >= 20]
        if not cand.empty:
            suggested_thr = float(cand.sort_values("net_pnl", ascending=False).iloc[0]["thr"])

    # Optional exchange reconciliation
    key, secret, testnet = _autodetect_bybit()
    mismatches: list[TradeMismatch] = []
    if key and secret and not args.skip_exchange:
        try:
            LOG.info("Fetching exchange trades for reconciliation…")
            ex = await _init_exchange(key, secret, testnet)
            try:
                # Load markets and keep only symbols the exchange currently knows about.
                await ex.load_markets()
                symbols_all = sorted(set(positions_aug["symbol"].dropna().unique()))
                ex_markets = set(getattr(ex, "markets", {}).keys())
                supported = [s for s in symbols_all if s in ex_markets]
                unsupported = sorted(set(symbols_all) - set(supported))
                if unsupported:
                    LOG.warning(
                        "Skipping %d symbols not present on exchange (likely delisted): %s",
                        len(unsupported),
                        ", ".join(unsupported[:10]) + ("…" if len(unsupported) > 10 else "")
                    )

                since = positions_aug["opened_at"].min() - timedelta(days=1) if "opened_at" in positions_aug else None
                until = positions_aug["closed_at"].max() + timedelta(days=1) if "closed_at" in positions_aug else None

                trade_map = await fetch_trades_for_symbols(ex, supported, since=since, until=until)
            finally:
                await ex.close()

            LOG.info("Reconciling %d positions across %d supported symbols…", len(positions_aug), len(supported))
            mismatches = reconcile_positions(positions_aug, fills, trade_map, tolerance_qty=args.tolerance)
        except Exception as e:
            LOG.warning("Exchange reconciliation skipped due to error: %s", e)


    # Build report
    ts = _now_utc().strftime("%Y%m%d_%H%M%S")
    out_dir = DEFAULT_RESULTS_DIR / f"audit_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    md: list[str] = []
    md.append(f"# DONCH — Performance Audit ({ts} UTC)\n")
    md.append(f"**Window:** {start} → {end}\n")
    md.append("## Summary\n")

    # Pre-format numbers safely
    sharpe_txt = f"{summary.sharpe_daily:.3f}" if summary.sharpe_daily is not None else "nan"
    ulcer_txt  = f"{summary.ulcer_index:.2f}" if summary.ulcer_index is not None else "nan"
    mar_txt    = f"{summary.mar_ratio:.3f}" if summary.mar_ratio is not None else "nan"
    payoff_txt = f"{summary.payoff_ratio:.2f}" if summary.payoff_ratio is not None else "nan"

    md.append(
        f"- Trades: **{summary.total_trades}**  |  Win%: **{summary.win_rate:.2f}%**  |  PF: **{summary.profit_factor:.3f}**\n"
        f"- Expectancy: **{_fmt_money(summary.expectancy)}**  |  Payoff: **{payoff_txt}**\n"
        f"- Net PnL: **{_fmt_money(summary.net_pnl)}**  |  Fees: **{_fmt_money(summary.fees_paid)}**\n"
        f"- Sharpe(d): **{sharpe_txt}**  |  Max DD: **{_fmt_pct(summary.max_drawdown_pct)}**  |  Ulcer: **{ulcer_txt}**\n"
        f"- CAGR: **{_fmt_pct(summary.cagr)}**  |  MAR: **{mar_txt}**\n"
    )


    if dist:
        md.append("\n## Stop / Take-Profit Diagnostics (audited)\n")
        ea = exit_audit["summary"]
        def _p(x): return "n/a" if x != x else f"{x:.1f}%"
        md.append(f"- Stop adequacy (winners with MAE > stop): **{_p(ea.get('stop_adequacy_pct', float('nan')))}**")
        md.append(f"- Stop waste (losers with MAE < 0.5×stop): **{_p(ea.get('stop_waste_pct', float('nan')))}**")
        md.append(f"- TP harvest (median pnl/MFE on winners): **{_p(ea.get('tp_harvest_pct', float('nan')))}**")
        if np.isfinite(ea.get('mae_p80_winners', float('nan'))):
            md.append(f"- Quantiles (winners): MAE p80 ≈ **{ea['mae_p80_winners']:+.2f}**, "
                    f"MFE p50 ≈ **{ea['mfe_p50_winners']:+.2f}**, p70 ≈ **{ea['mfe_p70_winners']:+.2f}**")

        # Optional: list suspicious trades (first 25) so you can eyeball logs
        mm = exit_audit["mismatches"]
        if not mm.empty:
            md.append("\n### Exit Consistency Audit — Anomalies (first 25)")
            md.append("| id | symbol | r | risk_usd | mae | mfe | net | recorded | inferred | note |")
            md.append("|---:|:------|---:|---:|---:|---:|---:|:---|:---|:---|")
            for _, row in mm.head(25).iterrows():
                md.append(f"| {int(row['id']) if pd.notna(row['id']) else ''} "
                        f"| {str(row['symbol'])} "
                        f"| {row['r_multiple']:+.2f} "
                        f"| {row['risk_usd']:+.2f} "
                        f"| {row['mae_usd']:+.2f} "
                        f"| {row['mfe_usd']:+.2f} "
                        f"| {row['net_pnl']:+.2f} "
                        f"| {row['reason_recorded']} "
                        f"| {row['reason_inferred']} "
                        f"| {row['flag']} |")


    # Exit-reason audit block
    audit = audit_exit_reasons(positions_aug, rel_tol=0.15, abs_tol=5e-3, deadline_tol_min=3.0, max_rows=25)

    md.append("\n## Exit Reasons\n")
    cov = audit["coverage"]
    md.append(f"_Coverage: n={cov.get('n_total',0)} | has MAE={cov.get('n_have_mae',0)} | "
              f"has MFE={cov.get('n_have_mfe',0)} | has risk/stop={cov.get('n_have_risk_or_stop',0)}_")

    # Recorded distribution
    rd = audit["recorded_dist"]
    if rd is not None and len(rd) > 0:
        md.append("\n**Recorded (from DB):**")
        md.append("\n| reason | n |")
        md.append("|---|---:|")
        for k, v in rd.sort_index().items():
            md.append(f"| {k} | {int(v)} |")
    else:
        md.append("\n_No recorded exit_reason column found._")

    # Inferred distribution
    infd = audit["inferred_dist"]
    if infd is not None and len(infd) > 0:
        md.append("\n\n**Inferred (from MAE/MFE/stop/TP/deadline):**")
        md.append("\n| reason | n |")
        md.append("|---|---:|")
        for k, v in infd.sort_index().items():
            md.append(f"| {k} | {int(v)} |")

    # Confusion table
    conf = audit["confusion"]
    if conf is not None and not conf.empty:
        md.append("\n\n**Recorded vs Inferred (counts):**")
        cols = list(conf.columns)
        md.append("| recorded \\ inferred | " + " | ".join(cols) + " |")
        md.append("|---" + "|---" * len(cols) + "|")
        for idx, row in conf.iterrows():
            md.append("| " + str(idx) + " | " + " | ".join(str(int(row[c])) for c in cols) + " |")

    # Suspicious examples
    sus = audit["suspicious"]
    if sus is not None and not sus.empty:
        md.append("\n\n**Suspicious rows (first 25):**")
        # keep it compact
        md.append("| id | symbol | closed_at | pnl | mae | mfe | risk | rec | inf |")
        md.append("|---:|:------|:----------|----:|----:|----:|----:|:----:|:---:|")
        for _, r in sus.iterrows():
            md.append(
                f"| {int(r['id']) if 'id' in r and pd.notna(r['id']) else ''} "
                f"| {r.get('symbol','')} "
                f"| {str(r.get('closed_at',''))[:19]} "
                f"| {(_safe_float(r.get('net_pnl', r.get('pnl')))):+.2f} "
                f"| {(_safe_float(r.get('mae_usd'))):+.2f} "
                f"| {(_safe_float(r.get('mfe_usd'))):+.2f} "
                f"| {(_safe_float(r.get('risk_usd'))):+.2f} "
                f"| {r.get('exit_reason_recorded','')} "
                f"| {r.get('exit_reason_inferred','')} |"
            )
    else:
        md.append("\n\n_No suspicious exit reason rows detected (good!)._")


    if "r_multiple" in positions_aug.columns:
        r = positions_aug["r_multiple"].dropna()
        if not r.empty:
            md.append("\n## R-Multiples\n")
            md.append(f"- Mean **{r.mean():+.2f}R**, Median **{r.median():+.2f}R**, p95 **{r.quantile(0.95):+.2f}R**")

    md.append("\n## Streaks\n")
    md.append(f"- Longest winning streak: **{longest_win}** trades")
    md.append(f"- Longest losing streak: **{longest_loss}** trades")

    if session_segs:
        md.append("\n## Session Expectancy\n")
        for name, exp, total, n in session_segs:
            md.append(f"- {name:<15} | n={n:>3} | exp={exp:+.2f} | total={total:+.2f}")
    if dow_segs:
        md.append("\n## Day-of-Week Expectancy\n")
        for name, exp, total, n in dow_segs:
            md.append(f"- {name:<15} | n={n:>3} | exp={exp:+.2f} | total={total:+.2f}")
    if hour_segs:
        md.append("\n## Hour-of-Day Expectancy (UTC)\n")
        for name, exp, total, n in hour_segs:
            md.append(f"- {str(name):<2}h | n={n:>3} | exp={exp:+.2f} | total={total:+.2f}")

    if prob_col:
        md.append("\n## Meta-Model Calibration\n")
        if calib_tab is not None and not calib_tab.empty:
            md.append(f"- **Brier score:** {summary.brier:.4f}")
            md.append("\n| bin | mean_pred | win_rate | n |\n|---:|---:|---:|---:|")
            for _, r in calib_tab.iterrows():
                md.append(f"|  | {r['mean_pred']:.3f} | {r['frac_positive']*100:5.2f}% | {int(r['n'])} |")
        if sweep_tab is not None and not sweep_tab.empty:
            md.append("\n### Threshold Sweep (p≥thr)\n")
            md.append("| thr | n | win% | exp | net_pnl |")
            md.append("|---:|---:|---:|---:|---:|")
            for _, r in sweep_tab.iterrows():
                md.append(f"| {r['thr']:.2f} | {int(r['n'])} | {r['win_rate']:.2f}% | {r['exp']:+.2f} | {r['net_pnl']:+.2f} |")
            if suggested_thr is not None:
                md.append(f"\n**Suggested META_PROB_THRESHOLD:** **{suggested_thr:.2f}** (max net PnL with n≥20)")

    if mismatches:
        md.append("\n## Reconciliation Warnings\n")
        for mm in mismatches:
            md.append(
                f"- PID {mm.position_id} {mm.symbol}: {mm.reason} | "
                f"db_qty={mm.db_qty:.6f} vs exch_qty={mm.exchange_qty:.6f}"
            )
    elif key and secret:
        md.append("\n_Exchange reconciliation passed – no quantity mismatches detected._")

    # ── Advanced analytics (vol-targeted equity, SQN/E-ratio, risk-of-ruin, stops/TP, rolling calibration, TWR)
    extras_md = build_extras_markdown(
        positions=positions_aug,
        equity=equity,
        prob_col=prob_col,
        rolling_window=500,
        starting_equity=starting_equity,
        vol_target_annual=0.10,  # 10% annualized target for the shadow curve (tweakable)
    )
    if extras_md:
        md.append("\n---\n")
        md.append(extras_md)

    # Save report & optional charts
    charts = _maybe_make_plots(out_dir, eq_curve, dd_curve, calib_tab, sweep_tab)
    if charts:
        md.append("\n## Charts\n")
        for label, path in charts.items():
            md.append(f"![{label}]({Path(path).name})")

    report_path = out_dir / "performance_audit.md"
    _write_markdown(report_path, "\n".join(md))
    print(f"Report written to {report_path}")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
