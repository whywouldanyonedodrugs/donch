import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _to_utc_ts(ts: Any) -> pd.Timestamp:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(t):
        raise ValueError(f"invalid timestamp: {ts}")
    return t


def _load_golden(path: Path) -> pd.DataFrame:
    g = pd.read_parquet(path)
    if "timestamp" in g.columns:
        ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
        g = g.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp")
        g = g.set_index("timestamp", drop=True)
    else:
        if not isinstance(g.index, pd.DatetimeIndex):
            raise ValueError("golden has no timestamp column and index is not DatetimeIndex")
        if g.index.tz is None:
            g.index = g.index.tz_localize("UTC")
        else:
            g.index = g.index.tz_convert("UTC")
        g = g.sort_index()
    return g


def _load_fixture(fixtures_dir: Path, symbol: str, tf: str) -> pd.DataFrame:
    p = fixtures_dir / f"{symbol}_{tf}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing fixture: {p}")

    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp", drop=True)
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"fixture {p} missing timestamp column and index is not DatetimeIndex")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index()

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "close" not in df.columns:
        raise ValueError(f"fixture {p} missing required 'close' column")

    df = df.dropna(subset=["close"]).sort_index()
    return df


def _first_nonnull_at_ts(g: pd.DataFrame, ts: pd.Timestamp, col: str) -> Optional[float]:
    if col not in g.columns:
        return None
    if ts not in g.index:
        return None
    v = g.loc[ts, col]
    if isinstance(v, pd.Series):
        v = v.dropna()
        if v.empty:
            return None
        v = v.iloc[0]
    if pd.isna(v):
        return None
    try:
        x = float(v)
        if not np.isfinite(x):
            return None
        return float(x)
    except Exception:
        return None


def _floor_bucket_close(asof_ts: pd.Timestamp, tf: str) -> pd.Timestamp:
    tf = str(tf).strip().lower()
    if tf in ("4h", "4hour", "h4"):
        return asof_ts.floor("4h")
    if tf in ("1d", "1day", "d", "day"):
        return asof_ts.floor("D")
    raise ValueError(f"unsupported tf: {tf}")


@dataclass(frozen=True)
class MarkovVariant:
    ret_kind: str                 # "log" | "pct"
    prob_kind: str                # "filtered" | "smoothed"
    trend: str                    # "c" | "n"
    switching_variance: bool      # True/False
    ewma_alpha: float             # 0 => no smoothing, else EWM(alpha, adjust=False)
    state_rule: str               # "prob>0.5" | "argmax"


def _compute_variant(
    df4h: pd.DataFrame,
    asof_ts: pd.Timestamp,
    tf: str,
    limit: int,
    min_obs: int,
    var: MarkovVariant,
) -> Optional[Tuple[float, int]]:
    """
    Strict as-of (no future leakage):
    - Slice bars by close_ts <= floor_bucket_close(asof_ts)
    - Use last `limit` bars (if limit>0)
    - Fit MarkovRegression on returns in that truncated window only
    """
    asof_ts = _to_utc_ts(asof_ts)
    cutoff = _floor_bucket_close(asof_ts, tf)

    df_use = df4h.loc[df4h.index <= cutoff]
    if df_use.empty:
        return None
    if limit and limit > 0:
        df_use = df_use.tail(int(limit))

    if len(df_use) < int(min_obs):
        return None

    close = df_use["close"].astype(float)
    if var.ret_kind == "log":
        ret = np.log(close).diff().dropna()
    elif var.ret_kind == "pct":
        ret = close.pct_change().dropna()
    else:
        raise ValueError(f"unknown ret_kind: {var.ret_kind}")

    if len(ret) < max(50, int(min_obs) - 1):
        return None

    try:
        mod = sm.tsa.MarkovRegression(
            ret,
            k_regimes=2,
            switching_variance=bool(var.switching_variance),
            trend=str(var.trend),
        )
        res = mod.fit(disp=False, maxiter=200)
    except Exception:
        return None

    if var.prob_kind == "filtered":
        probs = [res.filtered_marginal_probabilities[i] for i in range(2)]
    elif var.prob_kind == "smoothed":
        probs = [res.smoothed_marginal_probabilities[i] for i in range(2)]
    else:
        raise ValueError(f"unknown prob_kind: {var.prob_kind}")

    # Identify UP regime by higher weighted mean return under the chosen prob kind
    means: List[float] = []
    r = ret.reindex(probs[0].index).values
    for p in probs:
        w = p.values
        denom = max(float(np.sum(w)), 1e-12)
        mu = float(np.sum(w * r) / denom)
        means.append(mu)

    up_idx = int(np.argmax(means))
    p_up = probs[up_idx].clip(0.0, 1.0)

    if float(var.ewma_alpha) > 0.0:
        p_up = p_up.ewm(alpha=float(var.ewma_alpha), adjust=False).mean().clip(0.0, 1.0)

    last_prob = float(p_up.iloc[-1])

    if var.state_rule == "prob>0.5":
        last_state = int(last_prob > 0.5)
    elif var.state_rule == "argmax":
        # Most probable regime at last time; map to {0,1} via whether UP regime is more probable
        p0 = float(probs[0].iloc[-1])
        p1 = float(probs[1].iloc[-1])
        last_state = int((up_idx == 1 and p1 >= p0) or (up_idx == 0 and p0 >= p1))
    else:
        raise ValueError(f"unknown state_rule: {var.state_rule}")

    return last_prob, int(last_state)


def _build_variants(alpha_grid: List[float]) -> List[MarkovVariant]:
    vars: List[MarkovVariant] = []
    for ret_kind in ["log", "pct"]:
        for prob_kind in ["filtered", "smoothed"]:
            for trend in ["c", "n"]:
                for sv in [True, False]:
                    for a in alpha_grid:
                        for state_rule in ["prob>0.5", "argmax"]:
                            vars.append(
                                MarkovVariant(
                                    ret_kind=ret_kind,
                                    prob_kind=prob_kind,
                                    trend=trend,
                                    switching_variance=sv,
                                    ewma_alpha=float(a),
                                    state_rule=state_rule,
                                )
                            )
    return vars


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, type=str, help="Path to golden_features.parquet")
    ap.add_argument("--fixtures-dir", required=True, type=str, help="Path to tests/fixtures/regime")
    ap.add_argument("--symbol", required=True, type=str, help="Symbol, e.g. BTCUSDT")
    ap.add_argument("--tf", default="4H", type=str, help="Fixture TF (default: 4H)")
    ap.add_argument("--max-ts", default=60, type=int, help="Max timestamps to evaluate (default: 60)")
    ap.add_argument("--min-obs", default=80, type=int, help="Minimum bars required in window (default: 80)")
    ap.add_argument("--limit", default=500, type=int, help="Lookback bars per fit (default: 500)")
    ap.add_argument("--top", default=20, type=int, help="How many top variants to print (default: 20)")
    ap.add_argument("--golden-markov-prob-col", required=True, type=str, help="Golden prob column (e.g. markov_prob_up_4h)")
    ap.add_argument("--golden-markov-state-col", required=True, type=str, help="Golden state column (e.g. markov_state_4h)")
    ap.add_argument("--alpha-grid", default="0,0.1,0.2,0.3", type=str, help="Comma grid for EWMA alpha (default: 0,0.1,0.2,0.3)")

    args = ap.parse_args()

    golden_path = Path(args.golden)
    fixtures_dir = Path(args.fixtures_dir)
    symbol = str(args.symbol).upper()
    tf = str(args.tf).upper()
    prob_col = str(args.golden_markov_prob_col)
    state_col = str(args.golden_markov_state_col)

    g = _load_golden(golden_path)
    df4h = _load_fixture(fixtures_dir, symbol, tf)

    # Build candidate timestamps from golden (macro context => first non-null across symbols)
    cand: List[pd.Timestamp] = []
    for ts in g.index.unique():
        tsu = _to_utc_ts(ts)
        exp_p = _first_nonnull_at_ts(g, tsu, prob_col)
        exp_s = _first_nonnull_at_ts(g, tsu, state_col)
        if exp_p is None or exp_s is None:
            continue
        if tsu < df4h.index.min():
            continue
        if tsu > df4h.index.max() + pd.Timedelta("4h"):
            continue
        cand.append(tsu)

    cand = sorted(cand)[: int(args.max_ts)]
    if not cand:
        raise SystemExit("No comparable golden timestamps found in fixture window.")

    alpha_grid = []
    for s in str(args.alpha_grid).split(","):
        s = s.strip()
        if s == "":
            continue
        alpha_grid.append(float(s))

    variants = _build_variants(alpha_grid)

    results: List[Tuple[float, float, int, int, MarkovVariant]] = []
    # tuple: (prob_mae, 1-state_acc, n_eval, n_fitfail, variant)

    for var in variants:
        abs_err: List[float] = []
        ok_state = 0
        n_eval = 0
        n_fitfail = 0

        for ts in cand:
            exp_p = _first_nonnull_at_ts(g, ts, prob_col)
            exp_s = _first_nonnull_at_ts(g, ts, state_col)
            if exp_p is None or exp_s is None:
                continue

            out = _compute_variant(
                df4h=df4h,
                asof_ts=ts,
                tf="4h",
                limit=int(args.limit),
                min_obs=int(args.min_obs),
                var=var,
            )
            if out is None:
                n_fitfail += 1
                continue

            got_p, got_s = out
            abs_err.append(abs(float(got_p) - float(exp_p)))
            ok_state += int(int(got_s) == int(round(float(exp_s))))
            n_eval += 1

        if n_eval == 0:
            continue

        prob_mae = float(np.mean(abs_err)) if abs_err else float("inf")
        state_acc = float(ok_state) / float(n_eval)
        results.append((prob_mae, 1.0 - state_acc, n_eval, n_fitfail, var))

    if not results:
        raise SystemExit("All variants produced zero evaluated points; relax --min-obs/--limit or check fixtures.")

    results.sort(key=lambda t: (t[0], t[1], -t[2]))

    print(f"Evaluated timestamps: {len(cand)}")
    print(f"Variants with >=1 eval: {len(results)}")
    print("")
    print("TOP VARIANTS (sorted by prob MAE, then (1-acc), then n_eval desc):")
    print("prob_mae | state_acc | n_eval | n_fitfail | variant")
    for i, (mae, inv_acc, n_eval, n_fitfail, var) in enumerate(results[: int(args.top)]):
        print(
            f"{mae:9.6f} | {1.0-inv_acc:9.4f} | {n_eval:5d} | {n_fitfail:8d} | "
            f"ret={var.ret_kind} prob={var.prob_kind} trend={var.trend} "
            f"switch_var={var.switching_variance} alpha={var.ewma_alpha} state={var.state_rule}"
        )

    # Print some mismatch examples for the best variant
    best = results[0][-1]
    print("\nBEST VARIANT DETAIL:")
    print(best)

    rows: List[Tuple[pd.Timestamp, float, float, int, int]] = []
    for ts in cand:
        exp_p = _first_nonnull_at_ts(g, ts, prob_col)
        exp_s = _first_nonnull_at_ts(g, ts, state_col)
        if exp_p is None or exp_s is None:
            continue
        out = _compute_variant(df4h, ts, "4h", int(args.limit), int(args.min_obs), best)
        if out is None:
            continue
        got_p, got_s = out
        rows.append((ts, float(exp_p), float(got_p), int(round(float(exp_s))), int(got_s)))

    rows.sort(key=lambda r: abs(r[2] - r[1]), reverse=True)
    print("\nWorst 10 prob errors under BEST variant:")
    for ts, ep, gp, es, gs in rows[:10]:
        print(f"{ts} exp_p={ep:.6f} got_p={gp:.6f} | exp_s={es} got_s={gs} | abs_err={abs(gp-ep):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
