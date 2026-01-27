#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


# ----------------------------
# Utilities
# ----------------------------

def _to_utc_ts(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _read_parquet_with_timestamp(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop(columns=["timestamp"])
        df.index = ts
    else:
        # handle parquet metadata promoting timestamp to index
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
        else:
            raise AssertionError(f"{path} has no 'timestamp' col and index is not DatetimeIndex")
    df = df.sort_index()
    return df


def _floor_bucket_close(asof_ts: pd.Timestamp, tf: str) -> pd.Timestamp:
    """
    Map arbitrary decision_ts to the last completed bar CLOSE timestamp for tf.

    For 4h: a decision at 08:20 -> last completed 4h close is 08:00.
    """
    asof_ts = _to_utc_ts(asof_ts)
    if tf.lower() != "4h":
        raise AssertionError("This diagnostic tool currently only supports tf=4h.")
    return asof_ts.floor("4h")


def _extract_intercepts(res) -> Optional[np.ndarray]:
    """
    Try to extract per-regime intercepts from statsmodels params, if present.
    Works when trend='c' (or 'ct' etc).
    """
    names = None
    if hasattr(res, "model") and hasattr(res.model, "param_names"):
        names = list(res.model.param_names)
    elif hasattr(res, "param_names"):
        names = list(res.param_names)
    if not names:
        return None

    params = np.asarray(res.params, dtype=float)
    by_reg: Dict[int, float] = {}
    for i, nm in enumerate(names):
        m = re.match(r"^(?:const|intercept)\[(\d+)\]$", str(nm))
        if m:
            by_reg[int(m.group(1))] = float(params[i])

    if len(by_reg) >= 2 and 0 in by_reg and 1 in by_reg:
        return np.array([by_reg[0], by_reg[1]], dtype=float)
    return None


# ----------------------------
# Variant definition
# ----------------------------

@dataclass(frozen=True)
class MarkovVariant:
    use_log_returns: bool
    use_smoothed_probs: bool
    trend: str
    switching_variance: bool
    ewma_alpha: float
    scale: float
    up_state_rule: str   # "weighted_mean" or "intercept"
    state_rule: str      # "threshold" or "argmax"

    def key(self) -> Tuple[Any, ...]:
        return (
            self.use_log_returns,
            self.use_smoothed_probs,
            self.trend,
            self.switching_variance,
            float(self.ewma_alpha),
            float(self.scale),
            self.up_state_rule,
            self.state_rule,
        )


def _build_variants(args: argparse.Namespace) -> List[MarkovVariant]:
    use_log_grid = [True, False]
    use_smoothed_grid = [True, False]
    trend_grid = args.trend_grid
    switching_var_grid = [True, False]
    ewma_alpha_grid = args.ewma_alpha_grid
    scale_grid = args.scale_grid
    up_state_rule_grid = args.up_state_rule_grid
    state_rule_grid = args.state_rule_grid

    variants: List[MarkovVariant] = []
    for use_log in use_log_grid:
        for use_smoothed in use_smoothed_grid:
            for trend in trend_grid:
                for swv in switching_var_grid:
                    for alpha in ewma_alpha_grid:
                        for scale in scale_grid:
                            for up_rule in up_state_rule_grid:
                                for st_rule in state_rule_grid:
                                    variants.append(
                                        MarkovVariant(
                                            use_log_returns=use_log,
                                            use_smoothed_probs=use_smoothed,
                                            trend=trend,
                                            switching_variance=swv,
                                            ewma_alpha=float(alpha),
                                            scale=float(scale),
                                            up_state_rule=up_rule,
                                            state_rule=st_rule,
                                        )
                                    )
    return variants


# ----------------------------
# Scoring
# ----------------------------

def _score_one(
    df4h: pd.DataFrame,
    cutoff_ts: pd.Timestamp,
    limit: int,
    min_obs: int,
    maxiter: int,
    variant: MarkovVariant,
) -> Optional[Tuple[float, int]]:
    """
    Compute (prob_up, state_up) at "as-of cutoff_ts" for one variant.
    Returns None if insufficient data or fit fails.
    """
    cutoff_ts = _to_utc_ts(cutoff_ts)
    sub = df4h.loc[df4h.index <= cutoff_ts]
    if sub.empty:
        return None
    if limit is not None and int(limit) > 0 and len(sub) > int(limit):
        sub = sub.iloc[-int(limit):]

    if "close" not in sub.columns:
        raise AssertionError("Fixture 4h parquet must contain 'close'.")

    close = sub["close"].astype(float)
    if variant.use_log_returns:
        ret = np.log(close).diff()
    else:
        ret = close.pct_change()

    if variant.scale != 1.0:
        ret = ret * float(variant.scale)

    ret = ret.dropna()
    if len(ret) < int(min_obs):
        return None

    # Fit MarkovRegression
    try:
        mod = MarkovRegression(
            ret,
            k_regimes=2,
            trend=variant.trend,
            switching_variance=variant.switching_variance,
        )
        res = mod.fit(disp=False, maxiter=int(maxiter))
    except Exception:
        return None

    probs = res.smoothed_marginal_probabilities if variant.use_smoothed_probs else res.filtered_marginal_probabilities

    # Determine which regime is "UP"
    up_idx: Optional[int] = None
    if variant.up_state_rule == "intercept":
        intercepts = _extract_intercepts(res)
        if intercepts is not None:
            up_idx = int(np.argmax(intercepts))

    if up_idx is None:
        # weighted mean return by regime
        weights = probs.copy()
        means = (weights.mul(ret, axis=0)).sum(axis=0) / weights.sum(axis=0)
        up_idx = int(np.argmax(means.values))

    p_series = probs.iloc[:, up_idx].astype(float)

    # Optional EWMA smoothing of regime probability series
    if variant.ewma_alpha and variant.ewma_alpha > 0.0:
        p_series = p_series.ewm(alpha=float(variant.ewma_alpha), adjust=False).mean()

    prob_up = float(p_series.iloc[-1])

    # State definition
    if variant.state_rule == "argmax":
        # "up state" if the most likely regime equals up_idx
        last_argmax = int(np.argmax(probs.iloc[-1].values))
        state_up = 1 if last_argmax == up_idx else 0
    else:
        # threshold on prob_up
        state_up = 1 if prob_up >= 0.5 else 0

    return prob_up, int(state_up)


def _pick_first_nonnull_per_ts(golden: pd.DataFrame, ts_col: str, prob_col: str, state_col: Optional[str]) -> pd.DataFrame:
    """
    For each timestamp, pick a representative row:
      - prefer rows where prob_col is non-null
      - if state_col is provided, prefer rows where both are non-null
    Deterministic order within ts group.
    """
    g = golden.copy()
    g[ts_col] = pd.to_datetime(g[ts_col], utc=True)
    sort_cols = [ts_col]
    if "symbol" in g.columns:
        sort_cols.append("symbol")
    g = g.sort_values(sort_cols)

    rows = []
    for ts, grp in g.groupby(ts_col, sort=True):
        grp2 = grp[grp[prob_col].notna()]
        if grp2.empty:
            continue
        if state_col is not None and state_col in grp2.columns:
            grp3 = grp2[grp2[state_col].notna()]
            if not grp3.empty:
                grp2 = grp3
        rows.append(grp2.iloc[0])
    if not rows:
        return g.iloc[0:0]
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, type=str)
    ap.add_argument("--fixtures-dir", required=True, type=str)
    ap.add_argument("--symbol", required=True, type=str)
    ap.add_argument("--tf", default="4h", type=str)
    ap.add_argument("--max-ts", default=60, type=int)
    ap.add_argument("--min-obs", default=80, type=int)
    ap.add_argument("--limit", default=500, type=int)

    ap.add_argument("--golden-markov-prob-col", required=True, type=str)
    ap.add_argument("--golden-markov-state-col", default=None, type=str)

    ap.add_argument("--maxiter", default=120, type=int)
    ap.add_argument("--progress-every", default=10, type=int)

    ap.add_argument("--trend-grid", nargs="+", default=["c", "n"])
    ap.add_argument("--ewma-alpha-grid", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--scale-grid", nargs="+", type=float, default=[1.0, 100.0])
    ap.add_argument("--up-state-rule-grid", nargs="+", default=["weighted_mean", "intercept"])
    ap.add_argument("--state-rule-grid", nargs="+", default=["threshold", "argmax"])

    args = ap.parse_args()

    golden_path = Path(args.golden)
    fixtures_dir = Path(args.fixtures_dir)
    sym = args.symbol
    tf = args.tf.lower()

    # Load golden
    g = pd.read_parquet(golden_path)
    if "timestamp" in g.columns:
        ts_col = "timestamp"
    else:
        # handle parquet metadata promoting timestamp to index
        if isinstance(g.index, pd.DatetimeIndex):
            g = g.reset_index()
            ts_col = g.columns[0]
        else:
            raise AssertionError("Golden parquet must have timestamp column or DatetimeIndex.")

    prob_col = args.golden_markov_prob_col
    state_col = args.golden_markov_state_col

    if prob_col not in g.columns:
        raise AssertionError(f"Golden missing prob col: {prob_col}")
    if state_col is not None and state_col not in g.columns:
        raise AssertionError(f"Golden missing state col: {state_col}")

    picked = _pick_first_nonnull_per_ts(g, ts_col=ts_col, prob_col=prob_col, state_col=state_col)
    if picked.empty:
        raise AssertionError("No comparable golden rows found (prob col all null).")

    # Load 4h fixture
    fx_path = fixtures_dir / f"{sym.upper()}_4H.parquet"
    if not fx_path.exists():
        fx_path = fixtures_dir / f"{sym.upper()}_4h.parquet"
    if not fx_path.exists():
        raise AssertionError(f"Missing fixture parquet: {fx_path}")

    df4h = _read_parquet_with_timestamp(fx_path)

    # Filter picked timestamps to fixture coverage
    picked_ts = pd.to_datetime(picked[ts_col], utc=True)
    min_ix = df4h.index.min()
    max_ix = df4h.index.max()

    usable_rows = []
    for i in range(len(picked)):
        ts = _to_utc_ts(picked_ts.iloc[i])
        cutoff = _floor_bucket_close(ts, tf)
        if cutoff < min_ix or cutoff > max_ix:
            continue
        usable_rows.append(i)

    picked = picked.iloc[usable_rows].reset_index(drop=True)
    if picked.empty:
        raise AssertionError("No golden timestamps fall within fixture coverage after cutoff mapping.")

    if len(picked) > int(args.max_ts):
        picked = picked.iloc[: int(args.max_ts)].reset_index(drop=True)

    # Group timestamps by cutoff (cacheable)
    cutoff_to_rows: Dict[pd.Timestamp, List[int]] = {}
    for i in range(len(picked)):
        ts = _to_utc_ts(picked.loc[i, ts_col])
        cutoff = _floor_bucket_close(ts, tf)
        cutoff_to_rows.setdefault(cutoff, []).append(i)

    cutoffs = sorted(cutoff_to_rows.keys())
    print(f"Comparable golden timestamps: {len(picked)} rows, unique 4h cutoffs: {len(cutoffs)}")
    print(f"Fixture coverage: {min_ix} .. {max_ix}")

    variants = _build_variants(args)
    if int(args.limit) > 0 and len(variants) > int(args.limit):
        variants = variants[: int(args.limit)]
    print(f"Evaluating variants: {len(variants)} (maxiter={args.maxiter})")

    # Cache: (cutoff, variant.key()) -> (prob_up, state_up)
    cache: Dict[Tuple[pd.Timestamp, Tuple[Any, ...]], Optional[Tuple[float, int]]] = {}

    results = []
    t0 = time.time()

    for vi, var in enumerate(variants, start=1):
        abs_errs: List[float] = []
        state_hits = 0
        state_total = 0
        n_scored = 0

        for cutoff in cutoffs:
            ck = (cutoff, var.key())
            if ck not in cache:
                cache[ck] = _score_one(
                    df4h=df4h,
                    cutoff_ts=cutoff,
                    limit=int(args.limit),
                    min_obs=int(args.min_obs),
                    maxiter=int(args.maxiter),
                    variant=var,
                )
            got = cache[ck]
            if got is None:
                continue
            got_prob, got_state = got

            for row_i in cutoff_to_rows[cutoff]:
                exp_prob = picked.loc[row_i, prob_col]
                if pd.notna(exp_prob):
                    abs_errs.append(abs(float(exp_prob) - float(got_prob)))
                    n_scored += 1

                if state_col is not None and state_col in picked.columns:
                    exp_state = picked.loc[row_i, state_col]
                    if pd.notna(exp_state):
                        state_total += 1
                        if int(exp_state) == int(got_state):
                            state_hits += 1

        if n_scored == 0:
            mae = math.inf
        else:
            mae = float(np.mean(abs_errs))

        state_acc = float(state_hits / state_total) if state_total > 0 else float("nan")

        results.append((mae, state_acc, n_scored, state_total, var))

        if int(args.progress_every) > 0 and (vi == 1 or vi % int(args.progress_every) == 0):
            dt = time.time() - t0
            print(f"[{vi}/{len(variants)}] mae={mae:.6f} state_acc={state_acc:.3f} scored={n_scored} ({dt:.1f}s) var={dataclasses.asdict(var)}")

    results.sort(key=lambda x: (x[0], -x[1] if not math.isnan(x[1]) else 0.0))

    print("\nTop variants by prob MAE (lower is better):")
    for i, (mae, acc, n_scored, st_total, var) in enumerate(results[:25], start=1):
        print(f"{i:02d}) mae={mae:.6f} state_acc={acc:.3f} prob_n={n_scored} state_n={st_total} var={dataclasses.asdict(var)}")


if __name__ == "__main__":
    main()
