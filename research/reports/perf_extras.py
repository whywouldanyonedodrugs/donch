# research/reports/perf_extras.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


# ───────────────────────────── helpers ─────────────────────────────

def _ulcer_index(equity_curve: pd.Series) -> float:
    """Ulcer Index: sqrt(mean(drawdown_pct^2)), drawdown in percent."""
    if equity_curve is None or equity_curve.empty:
        return float("nan")
    run_max = equity_curve.cummax()
    dd_pct = (equity_curve / run_max - 1.0) * 100.0
    return float(np.sqrt(np.mean(np.square(dd_pct.fillna(0.0)))))


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve is None or equity_curve.empty:
        return float("nan")
    run_max = equity_curve.cummax()
    dd = equity_curve / run_max - 1.0
    return float(dd.min())


def _daily_equity(equity: pd.DataFrame,
                  positions: pd.DataFrame,
                  *,
                  starting_equity: float = 1000.0) -> pd.Series:
    """
    Return a daily equity series (last observation per day, forward-filled).
    If equity snapshots are missing, synthesize from positions.
    """
    if equity is not None and not equity.empty and "equity" in equity.columns:
        eq = equity.copy()
        eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
        eq.dropna(subset=["equity"], inplace=True)
        eq.sort_values("ts", inplace=True)
        eq.set_index("ts", inplace=True)
        daily = eq["equity"].resample("1D").last().ffill()
        return daily

    # Fallback – from closed trades
    df = positions.copy()
    df["fees_paid"] = pd.to_numeric(df.get("fees_paid", 0.0), errors="coerce").fillna(0.0)
    df["pnl"] = pd.to_numeric(df.get("pnl", 0.0), errors="coerce").fillna(0.0)
    df["net_pnl"] = df["pnl"] - df["fees_paid"]
    df.sort_values("closed_at", inplace=True)
    eq_curve = starting_equity + df["net_pnl"].cumsum()
    eq_curve.index = pd.to_datetime(df["closed_at"], utc=True, errors="coerce").fillna(pd.Timestamp.utcnow())
    daily = eq_curve.resample("1D").last().ffill()
    return daily


# ───────────────── volatility-targeted shadow equity ─────────────────

@dataclass
class VolShadowMetrics:
    twr: Optional[float]
    sharpe: Optional[float]
    ulcer: Optional[float]
    upi: Optional[float]
    max_dd_pct: Optional[float]


def volatility_targeted_shadow_equity(
    positions: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    starting_equity: float = 1000.0,
    vol_target_annual: float = 0.10,   # e.g. 10% annualized
    lookback_days: int = 20,
    lev_cap: float = 5.0,
) -> tuple[pd.Series, VolShadowMetrics]:
    """
    Build a parallel equity curve where each day's return is scaled so the *ex-ante*
    annualized volatility is near vol_target_annual. Vol estimate is a trailing
    sample std of daily returns, shifted by 1 day to avoid lookahead.

    Returns (shadow_equity_curve, metrics).
    """
    daily = _daily_equity(equity, positions, starting_equity=starting_equity)
    if len(daily) < max(lookback_days + 2, 5):
        return pd.Series(dtype=float), VolShadowMetrics(None, None, None, None, None)

    rets = daily.pct_change().dropna()
    if rets.empty:
        return pd.Series(dtype=float), VolShadowMetrics(None, None, None, None, None)

    # Realized ann vol (shifted to avoid lookahead)
    roll_vol = rets.rolling(lookback_days).std(ddof=1) * math.sqrt(365)
    target_lev = (vol_target_annual / roll_vol).shift(1).clip(lower=0, upper=lev_cap)
    target_lev = target_lev.fillna(0.0)

    shadow_rets = rets * target_lev
    start = float(daily.dropna().iloc[0])
    shadow_curve = pd.Series(start * (1.0 + shadow_rets).cumprod(), index=shadow_rets.index)

    # Metrics
    twr = float(shadow_curve.iloc[-1] / shadow_curve.iloc[0] - 1.0) if len(shadow_curve) else None
    d = shadow_curve.resample("1D").last().dropna()
    sharpe = None
    if len(d) >= 5:
        r = d.pct_change().dropna()
        sd = r.std(ddof=1)
        sharpe = float((r.mean() / sd) * math.sqrt(365)) if sd and sd > 0 else 0.0

    ulcer = _ulcer_index(shadow_curve) if len(shadow_curve) else float("nan")
    annual_ret = None
    if len(d) >= 2:
        yrs = max(1e-9, (d.index[-1] - d.index[0]).days / 365.0)
        annual_ret = float((d.iloc[-1] / d.iloc[0]) ** (1.0 / yrs) - 1.0) if d.iloc[0] > 0 else None
    upi = float(annual_ret / ulcer) if (annual_ret is not None and ulcer == ulcer and ulcer > 0) else None
    mdd = _max_drawdown(shadow_curve)
    mdp = float(mdd * 100) if mdd == mdd else None

    return shadow_curve, VolShadowMetrics(twr, sharpe, float(ulcer), upi, mdp)


# ───────────────── R-normalized quality (SQN, E-ratio) ─────────────────

def compute_sqn(r: pd.Series) -> Optional[float]:
    """
    Van Tharp's SQN-like: mean(R)/std(R) * sqrt(n), using R-multiples.
    """
    r = pd.to_numeric(r, errors="coerce").dropna()
    n = len(r)
    if n < 10:
        return None
    sd = r.std(ddof=1)
    if not sd or sd <= 0:
        return None
    return float(r.mean() / sd * math.sqrt(n))


def sqn_by(df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    if "r_multiple" not in df.columns:
        return pd.DataFrame()
    out = []
    g = df.dropna(subset=["r_multiple"]).groupby(list(by), dropna=False)
    for key, sub in g:
        val = compute_sqn(sub["r_multiple"])
        if val is None:
            continue
        key = key if isinstance(key, tuple) else (key,)
        out.append({**{by[i]: key[i] for i in range(len(by))}, "sqn": val, "n": int(len(sub))})
    return pd.DataFrame(out).sort_values("sqn", ascending=False)


def eratio_by(df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Edge/E-ratio ≈ median(MFE)/median(|MAE|); <1 is suspect."""
    if "mfe_usd" not in df.columns or "mae_usd" not in df.columns:
        return pd.DataFrame()
    z = df.copy()
    z["mfe_usd"] = pd.to_numeric(z["mfe_usd"], errors="coerce")
    z["mae_usd"] = pd.to_numeric(z["mae_usd"], errors="coerce").abs()
    g = z.dropna(subset=["mfe_usd", "mae_usd"]).groupby(list(by), dropna=False)
    out = []
    for key, sub in g:
        mfe_med = sub["mfe_usd"].median()
        mae_med = sub["mae_usd"].median()
        if mae_med <= 0:
            continue
        er = float(mfe_med / mae_med)
        key = key if isinstance(key, tuple) else (key,)
        out.append({**{by[i]: key[i] for i in range(len(by))}, "e_ratio": er, "n": int(len(sub))})
    return pd.DataFrame(out).sort_values("e_ratio", ascending=False)


# ───────────────── risk-of-ruin (Monte Carlo on R) ─────────────────

def risk_of_ruin_heatmap(
    r: pd.Series,
    *,
    f_grid: np.ndarray | None = None,
    horizon_trades: int = 200,
    trials: int = 5000,
    ruin_drawdown_pct: float = 0.5,  # 50% equity loss
    seed: int = 1337,
) -> pd.DataFrame:
    """
    Approximate risk-of-ruin when staking fraction f of equity per trade against
    the *empirical* R-multiple distribution.
    Ruin = hitting ruin_drawdown_pct drawdown at any time in the horizon.
    """
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 30:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    r_vals = r.values
    if f_grid is None:
        f_grid = np.linspace(0.25, 1.0, 16)

    results = []
    for f in f_grid:
        # sample [trials x horizon] R draws
        idx = rng.integers(low=0, high=len(r_vals), size=(trials, horizon_trades))
        samples = r_vals[idx]  # shape: (trials, horizon)
        # equity paths
        path_rets = f * samples
        equity = np.cumprod(1.0 + path_rets, axis=1)
        run_max = np.maximum.accumulate(equity, axis=1)
        dd = equity / run_max - 1.0
        ruined = (dd <= -ruin_drawdown_pct).any(axis=1)
        p_ruin = float(ruined.mean())
        results.append({"f": float(f), "p_ruin": p_ruin})
    return pd.DataFrame(results)


# ───────────────── stop / take-profit diagnostics ─────────────────

@dataclass
class StopTpDiag:
    winners_stop_violation_share: Optional[float]
    losers_stop_waste_share: Optional[float]
    tp_capture_median: Optional[float]
    mae_p80_winners: Optional[float]
    mfe_p50_winners: Optional[float]
    mfe_p70_winners: Optional[float]


def stop_tp_diagnostics(df: pd.DataFrame) -> StopTpDiag:
    """
    Uses mae_usd, mfe_usd, risk_usd (if present). Interprets risk_usd as
    initial dollar risk (≈ stop distance × size).
    """
    if "mae_usd" not in df.columns or "mfe_usd" not in df.columns or "risk_usd" not in df.columns:
        return StopTpDiag(None, None, None, None, None, None)

    z = df.copy()
    for c in ["mae_usd", "mfe_usd", "risk_usd", "pnl"]:
        if c in z.columns:
            z[c] = pd.to_numeric(z[c], errors="coerce")

    z = z.dropna(subset=["mae_usd", "mfe_usd", "risk_usd"])
    if z.empty:
        return StopTpDiag(None, None, None, None, None, None)

    winners = z[z["pnl"] > 0]
    losers  = z[z["pnl"] < 0]

    # If a winner's MAE exceeded the nominal stop, you'd have been stopped — stop too tight / or you re-entered.
    w_stop_violate = None
    if not winners.empty:
        w_stop_violate = float((winners["mae_usd"] > 0.95 * winners["risk_usd"]).mean())

    # Losers where price never traveled far against you → stop might be too wide (you paid extra).
    l_stop_waste = None
    if not losers.empty:
        l_stop_waste = float((losers["mae_usd"] < 0.5 * losers["risk_usd"]).mean())

    # Profit harvest vs available MFE (winners only)
    tp_capture = None
    if not winners.empty:
        cap = (winners["pnl"].clip(lower=0) / winners["mfe_usd"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        if not cap.dropna().empty:
            tp_capture = float(cap.median())

    mae_p80 = float(winners["mae_usd"].quantile(0.80)) if not winners.empty else None
    mfe_p50 = float(winners["mfe_usd"].quantile(0.50)) if not winners.empty else None
    mfe_p70 = float(winners["mfe_usd"].quantile(0.70)) if not winners.empty else None

    return StopTpDiag(w_stop_violate, l_stop_waste, tp_capture, mae_p80, mfe_p50, mfe_p70)


# ───────────────── rolling live calibration (cost-aware sweep) ─────────────────

def reliability_bins(probs: pd.Series, ywin: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Quantile bins → mean predicted vs observed win-rate."""
    df = pd.DataFrame({"p": pd.to_numeric(probs, errors="coerce").clip(0, 1),
                       "y": (pd.to_numeric(ywin, errors="coerce") > 0).astype(int)}).dropna()
    if df.empty or df["p"].nunique() < 2:
        return pd.DataFrame()
    df["bin"] = pd.qcut(df["p"], q=min(n_bins, max(2, df["p"].nunique())), duplicates="drop")
    tab = df.groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"),
        frac_positive=("y", "mean"),
        n=("y", "size"),
    ).reset_index(drop=True)
    return tab


def expected_calibration_error(tab: pd.DataFrame) -> Optional[float]:
    if tab is None or tab.empty:
        return None
    # ECE: sum over bins of |acc - conf| weighted by bin frequency
    w = tab["n"] / tab["n"].sum()
    ece = (w * (tab["frac_positive"] - tab["mean_pred"]).abs()).sum()
    return float(ece)


def cost_aware_threshold_sweep(sub: pd.DataFrame, prob_col: str, thresholds: Sequence[float]) -> pd.DataFrame:
    """
    EV(t) = p_hat * E[gain_R] − (1 − p_hat) * E[loss_R], using the subset with p≥t.
    """
    rows = []
    for t in thresholds:
        ss = sub[sub[prob_col] >= t]
        n = len(ss)
        if n == 0:
            rows.append({"thr": t, "n": 0, "win%": np.nan, "EV_R": np.nan})
            continue
        r = pd.to_numeric(ss.get("r_multiple", np.nan), errors="coerce")
        pnl = pd.to_numeric(ss.get("net_pnl", np.nan), errors="coerce")
        # fall back to sign from pnl if r_multiple missing
        wins = (r > 0) if r.notna().any() else (pnl > 0)
        gain_R = r[wins].mean(skipna=True)
        loss_R = (-r[~wins]).mean(skipna=True)
        # If r missing, cannot compute cost-aware EV
        if pd.isna(gain_R) or pd.isna(loss_R):
            rows.append({"thr": float(t), "n": int(n), "win%": float(wins.mean()*100.0), "EV_R": np.nan})
            continue
        p_hat = float(ss[prob_col].mean())
        ev = p_hat * gain_R - (1.0 - p_hat) * loss_R
        rows.append({"thr": float(t), "n": int(n), "win%": float(wins.mean()*100.0), "EV_R": float(ev)})
    return pd.DataFrame(rows)


# ───────────────── time-weighted return / IRR (cash-flow aware) ─────────────────

def time_weighted_return(daily_equity: pd.Series) -> Optional[float]:
    """TWR over the period (product of (1+r_d) − 1)."""
    if daily_equity is None or daily_equity.empty:
        return None
    r = daily_equity.pct_change().dropna()
    return float((1.0 + r).prod() - 1.0) if not r.empty else None


# ───────────────── markdown builder ─────────────────

def build_extras_markdown(
    positions: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    prob_col: Optional[str],
    rolling_window: int = 500,
    starting_equity: float = 1000.0,
    vol_target_annual: float = 0.10,
) -> str:
    """
    Returns a Markdown string with advanced analytics sections. All pieces are
    optional and skip themselves if required columns are missing.
    """
    md: list[str] = []

    # 1) Vol-targeted shadow equity
    shadow_curve, sm = volatility_targeted_shadow_equity(
        positions, equity, starting_equity=starting_equity, vol_target_annual=vol_target_annual
    )
    if not shadow_curve.empty:
        md.append("## Volatility-Targeted Shadow Equity")
        md.append(
            f"- Target: **{vol_target_annual*100:.1f}%** ann vol  |  TWR: **{(sm.twr or float('nan'))*100:.2f}%**  |  "
            f"Sharpe(d): **{(sm.sharpe if sm.sharpe is not None else float('nan')):.2f}**"
        )
        md.append(
            f"- Max DD: **{(sm.max_dd_pct if sm.max_dd_pct is not None else float('nan')):.2f}%**  |  "
            f"Ulcer: **{(sm.ulcer if sm.ulcer is not None else float('nan')):.2f}**  |  "
            f"UPI: **{(sm.upi if sm.upi is not None else float('nan')):.2f}**"
        )

    # 2) R-normalized quality (SQN)
    if "r_multiple" in positions.columns:
        md.append("\n## SQN (R-normalized)")
        overall_sqn = compute_sqn(positions["r_multiple"])
        if overall_sqn is not None:
            md.append(f"- **Overall SQN:** {overall_sqn:.2f}")
        for label, cols in [
            ("By session", ["session_tag_at_entry"]),
            ("By day of week", ["day_of_week_at_entry"]),
            ("By hour", ["hour_at_entry" if "hour_at_entry" in positions.columns else "opened_at"]),
        ]:
            if all(c in positions.columns for c in cols):
                tmp = positions.copy()
                if "opened_at" in cols and "opened_at" in tmp.columns:
                    tmp["hour_at_entry"] = tmp["opened_at"].dt.hour
                    cols = ["hour_at_entry"]
                tbl = sqn_by(tmp, cols)
                if not tbl.empty:
                    md.append(f"- **{label} (top 5)**")
                    for _, r in tbl.head(5).iterrows():
                        key = " / ".join(str(r[c]) for c in cols)
                        md.append(f"    - {key}: SQN={r['sqn']:.2f} (n={int(r['n'])})")

    # 3) E-ratio by setup (if columns available)
    candidate_cols = [c for c in ["setup", "signal", "signal_name", "strategy", "session_tag_at_entry"] if c in positions.columns]
    if candidate_cols:
        md.append("\n## Edge / E-ratio by setup")
        tbl = eratio_by(positions, [candidate_cols[0]])
        if not tbl.empty:
            for _, r in tbl.head(10).iterrows():
                md.append(f"- {candidate_cols[0]}={r[candidate_cols[0]]}: E-ratio={r['e_ratio']:.2f} (n={int(r['n'])})")
            bad = tbl[tbl["e_ratio"] < 1.0]
            if not bad.empty:
                md.append(f"\n_⚠ Buckets with E-ratio < 1.0: {len(bad)} (suspect)_")

    # 4) Stop/TP diagnostics
    diag = stop_tp_diagnostics(positions)
    if any(v is not None for v in diag.__dict__.values()):
        md.append("\n## Stop / Take-Profit Diagnostics")
        if diag.winners_stop_violation_share is not None:
            md.append(f"- Stop adequacy (winners with MAE > stop): **{diag.winners_stop_violation_share*100:.1f}%**")
        if diag.losers_stop_waste_share is not None:
            md.append(f"- Stop waste (losers with MAE < 0.5×stop): **{diag.losers_stop_waste_share*100:.1f}%**")
        if diag.tp_capture_median is not None:
            md.append(f"- TP harvest (median pnl/MFE on winners): **{diag.tp_capture_median*100:.1f}%**")
        if diag.mae_p80_winners is not None:
            md.append(f"- Suggest: initial stop near winners' MAE p80 ≈ **{diag.mae_p80_winners:+.2f}**")
        if diag.mfe_p50_winners is not None and diag.mfe_p70_winners is not None:
            md.append(f"- Consider partial TP around MFE p50 ≈ **{diag.mfe_p50_winners:+.2f}**, trail beyond p70 ≈ **{diag.mfe_p70_winners:+.2f}**")

    # 5) Risk-of-ruin heatmap on R-multiples
    if "r_multiple" in positions.columns and positions["r_multiple"].notna().any():
        md.append("\n## Risk-of-Ruin (Monte Carlo on R)")
        heat = risk_of_ruin_heatmap(positions["r_multiple"])
        if not heat.empty:
            # show compact grid
            show = heat.copy()
            show["p_ruin%"] = (show["p_ruin"] * 100).round(1)
            # sample a few evenly spaced rows
            for _, r in show.iloc[::3].iterrows():
                md.append(f"- f={r['f']:.2f} → ruin≈ **{r['p_ruin%']:.1f}%**")

    # 6) Rolling live calibration (last N trades)
    if prob_col and prob_col in positions.columns:
        md.append(f"\n## Live Calibration (last {rolling_window} trades)")
        last = positions.tail(rolling_window)
        tab = reliability_bins(last[prob_col], last["net_pnl"])
        if not tab.empty:
            ece = expected_calibration_error(tab)
            brier = float(((last[prob_col].clip(0, 1) - (last["net_pnl"] > 0).astype(int)) ** 2).mean())
            md.append(f"- Brier: **{brier:.3f}**  |  ECE: **{(ece if ece is not None else float('nan')):.3f}**")
            md.append("| bin | mean_pred | win_rate | n |")
            md.append("|---:|---:|---:|---:|")
            for _, r in tab.iterrows():
                md.append(f"|  | {r['mean_pred']:.3f} | {r['frac_positive']*100:5.2f}% | {int(r['n'])} |")

            thr = np.round(np.linspace(0.50, 0.95, 10), 2)
            sweep = cost_aware_threshold_sweep(last, prob_col, thr)
            if not sweep.empty:
                md.append("\n### Cost-aware Threshold Sweep (EV in R)")
                md.append("| thr | n | win% | EV(R) |")
                md.append("|---:|---:|---:|---:|")
                for _, r in sweep.iterrows():
                    md.append(f"| {r['thr']:.2f} | {int(r['n'])} | {r['win%']:.2f}% | {r['EV_R'] if r['EV_R']==r['EV_R'] else float('nan'):+.3f} |")
                # Suggest best EV threshold with min coverage
                cand = sweep[sweep["n"] >= 50].copy()
                if not cand.empty and cand["EV_R"].notna().any():
                    best = cand.sort_values("EV_R", ascending=False).iloc[0]
                    md.append(f"\n**Suggested threshold (cost-aware):** **{best['thr']:.2f}** (EV≈{best['EV_R']:+.3f} R, n={int(best['n'])})")

    # 7) Cash-flow aware return reporting (TWR only; IRR requires cashflows)
    daily = _daily_equity(equity, positions, starting_equity=starting_equity)
    twr = time_weighted_return(daily)
    if twr is not None:
        md.append("\n## Cash-flow-aware Returns")
        md.append(f"- **Time-Weighted Return (TWR): {twr*100:.2f}%**")

    return "\n".join(md)
