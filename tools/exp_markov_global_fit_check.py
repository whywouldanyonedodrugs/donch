#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

def read_parquet_with_timestamp(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.drop(columns=["timestamp"])
        df.index = ts
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
        else:
            raise SystemExit(f"{path} has no timestamp and index is not DatetimeIndex")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def normalize_golden(g: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in g.columns:
        g = g.copy()
        g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
    elif isinstance(g.index, pd.DatetimeIndex):
        g = g.reset_index().rename(columns={g.reset_index().columns[0]: "timestamp"})
        g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
    else:
        raise SystemExit("golden has no timestamp column and index not datetime")
    g = g.dropna(subset=["timestamp"]).sort_values(["timestamp"] + (["symbol"] if "symbol" in g.columns else []))
    return g

def pick_expected_rows(g: pd.DataFrame, prob_col: str, state_col: str | None, golden_symbol: str | None, max_ts: int) -> pd.DataFrame:
    if golden_symbol is not None:
        if "symbol" not in g.columns:
            raise SystemExit("--golden-symbol provided but golden has no symbol column")
        g = g[g["symbol"].astype(str).str.upper() == golden_symbol.upper()].copy()

    rows = []
    for ts, grp in g.groupby("timestamp", sort=True):
        grp2 = grp[grp[prob_col].notna()]
        if grp2.empty:
            continue
        if state_col is not None and state_col in grp2.columns:
            grp3 = grp2[grp2[state_col].notna()]
            if not grp3.empty:
                grp2 = grp3
        rows.append(grp2.iloc[0])
    out = pd.DataFrame(rows).reset_index(drop=True)
    if max_ts > 0 and len(out) > max_ts:
        out = out.iloc[:max_ts].reset_index(drop=True)
    return out

def ewma(p: pd.Series, alpha: float) -> pd.Series:
    if alpha <= 0:
        return p
    return p.ewm(alpha=alpha, adjust=False).mean()

def fit_and_score(ret: pd.Series, ts_cutoffs: pd.Series, exp_prob: np.ndarray, exp_state: np.ndarray | None,
                  use_log: bool, trend: str, switching_variance: bool, use_smoothed: bool, alpha: float, scale: float,
                  up_rule: str, state_rule: str):
    r = ret.dropna()
    if scale != 1.0:
        r = r * float(scale)

    mod = MarkovRegression(r, k_regimes=2, trend=trend, switching_variance=switching_variance)
    res = mod.fit(disp=False, maxiter=200)

    probs = res.smoothed_marginal_probabilities if use_smoothed else res.filtered_marginal_probabilities

    # Determine "up" regime
    if up_rule == "weighted_mean":
        means = (probs.mul(r, axis=0)).sum(axis=0) / probs.sum(axis=0)
        up_idx = int(np.argmax(means.values))
    else:
        # intercept rule
        names = list(getattr(res.model, "param_names", []))
        params = np.asarray(res.params, dtype=float)
        by_reg = {}
        for i, nm in enumerate(names):
            if str(nm).startswith("const[") and str(nm).endswith("]"):
                k = int(str(nm)[6:-1])
                by_reg[k] = float(params[i])
        if 0 in by_reg and 1 in by_reg:
            up_idx = int(np.argmax([by_reg[0], by_reg[1]]))
        else:
            means = (probs.mul(r, axis=0)).sum(axis=0) / probs.sum(axis=0)
            up_idx = int(np.argmax(means.values))

    p = probs.iloc[:, up_idx].astype(float)
    p = ewma(p, alpha)

    # align to cutoffs
    p_at = []
    s_at = []
    for ct in ts_cutoffs:
        if ct not in p.index:
            # exact match required for this test
            p_at.append(np.nan)
            s_at.append(np.nan)
            continue
        pval = float(p.loc[ct])
        p_at.append(pval)
        if state_rule == "argmax":
            arg = int(np.argmax(probs.loc[ct].values))
            s_at.append(1 if arg == up_idx else 0)
        else:
            s_at.append(1 if pval >= 0.5 else 0)

    p_at = np.asarray(p_at, dtype=float)
    s_at = np.asarray(s_at, dtype=float)

    mask = np.isfinite(p_at) & np.isfinite(exp_prob)
    mae = float(np.mean(np.abs(p_at[mask] - exp_prob[mask]))) if mask.any() else np.inf

    state_acc = np.nan
    if exp_state is not None:
        mask2 = mask & np.isfinite(s_at) & np.isfinite(exp_state)
        if mask2.any():
            state_acc = float(np.mean((s_at[mask2].astype(int) == exp_state[mask2].astype(int)).astype(float)))

    return mae, state_acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--fixtures-dir", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--golden-prob-col", required=True)
    ap.add_argument("--golden-state-col", default=None)
    ap.add_argument("--golden-symbol", default=None, help="Restrict expected rows to this symbol if golden has symbol column")
    ap.add_argument("--max-ts", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    g = normalize_golden(pd.read_parquet(args.golden))
    exp = pick_expected_rows(g, args.golden_prob_col, args.golden_state_col, args.golden_symbol, args.max_ts)

    if exp.empty:
        raise SystemExit("No expected rows after filtering. (prob col null or symbol filter eliminated all rows)")

    if "symbol" in exp.columns:
        print("Expected-row symbol distribution (top 10):", exp["symbol"].value_counts().head(10).to_dict())

    fx = Path(args.fixtures_dir) / f"{args.symbol.upper()}_4H.parquet"
    if not fx.exists():
        fx = Path(args.fixtures_dir) / f"{args.symbol.upper()}_4h.parquet"
    df4h = read_parquet_with_timestamp(fx)

    exp_ts = pd.to_datetime(exp["timestamp"], utc=True)
    exp_cut = exp_ts.dt.floor("4h")

    # restrict to fixture coverage
    min_ix, max_ix = df4h.index.min(), df4h.index.max()
    m = (exp_cut >= min_ix) & (exp_cut <= max_ix)
    exp = exp.loc[m.values].reset_index(drop=True)
    exp_cut = exp_cut.loc[m.values].reset_index(drop=True)

    if exp.empty:
        raise SystemExit("No expected timestamps are within BTC fixture coverage after 4h floor.")

    exp_prob = exp[args.golden_prob_col].astype(float).to_numpy()
    exp_state = None
    if args.golden_state_col is not None:
        exp_state = exp[args.golden_state_col].astype(float).to_numpy()

    close = df4h["close"].astype(float)

    # Global-fit experiment = fit ONCE, then score earlier timestamps
    grids = []
    for use_log in [False, True]:
        ret = (np.log(close).diff() if use_log else close.pct_change())
        for trend in ["n", "c"]:
            for swv in [True, False]:
                for use_sm in [False, True]:
                    for scale in [1.0, 100.0]:
                        for up_rule in ["weighted_mean", "intercept"]:
                            for state_rule in ["threshold", "argmax"]:
                                grids.append((use_log, trend, swv, use_sm, scale, up_rule, state_rule))

    print(f"Scoring global-fit variants: {len(grids)} (fit once each)")
    best = None
    for (use_log, trend, swv, use_sm, scale, up_rule, state_rule) in grids:
        ret = (np.log(close).diff() if use_log else close.pct_change())
        mae, acc = fit_and_score(
            ret=ret, ts_cutoffs=exp_cut, exp_prob=exp_prob, exp_state=exp_state,
            use_log=use_log, trend=trend, switching_variance=swv, use_smoothed=use_sm,
            alpha=float(args.alpha), scale=float(scale), up_rule=up_rule, state_rule=state_rule
        )
        rec = (mae, -1.0 if np.isnan(acc) else -acc, use_log, trend, swv, use_sm, scale, up_rule, state_rule, acc)
        if best is None or rec < best:
            best = rec

    mae, negacc, use_log, trend, swv, use_sm, scale, up_rule, state_rule, acc = best
    print("\nBEST global-fit variant:")
    print({
        "mae": mae,
        "state_acc": acc,
        "use_log_returns": use_log,
        "trend": trend,
        "switching_variance": swv,
        "use_smoothed_probs": use_sm,
        "scale": scale,
        "up_state_rule": up_rule,
        "state_rule": state_rule,
        "alpha": float(args.alpha),
        "n_eval": len(exp),
        "fixture": str(fx),
    })

if __name__ == "__main__":
    main()
