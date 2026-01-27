#!/usr/bin/env python3
"""
Diagnostic: brute-force a small set of MarkovRegression variants and compare to golden parquet.

This is for STRICT-PARITY debugging: we do not use wall-clock; everything is "as-of ts".
We map as-of ts to the last completed 4h bar (bar close <= ts) from fixtures.

Usage example:
  python tools/diag_markov_golden_variants.py \
    --golden /root/apps/donch/results/meta_export/golden_features.parquet \
    --fixtures-dir /root/apps/donch/tests/fixtures/regime \
    --symbol BTCUSDT \
    --max-ts 60 \
    --min-obs 80 \
    --golden-markov-prob-col markov_prob_up_4h \
    --golden-markov-state-col markov_state_4h
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def _load_golden(path: Path) -> pd.DataFrame:
    g = pd.read_parquet(path)
    if "timestamp" in g.columns:
        ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
        g = g.assign(timestamp=ts).dropna(subset=["timestamp"]).set_index("timestamp")
    elif not isinstance(g.index, pd.DatetimeIndex):
        raise ValueError("Golden parquet must have 'timestamp' column or DatetimeIndex.")
    else:
        g = g.copy()
        g.index = pd.to_datetime(g.index, utc=True, errors="coerce")
        g = g.dropna().sort_index()
    if "symbol" not in g.columns:
        raise ValueError("Golden parquet missing required column: 'symbol'")
    return g.sort_index()


def _load_fixture_4h(fixtures_dir: Path, symbol: str) -> pd.DataFrame:
    p = fixtures_dir / f"{symbol.upper()}_4H.parquet"
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_parquet(p)
    if "timestamp" not in df.columns:
        raise ValueError(f"{p.name} missing 'timestamp' column")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp", drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            raise ValueError(f"{p.name} missing required col {c!r}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    return df.sort_index()


def _exp_at_ts(g: pd.DataFrame, ts: pd.Timestamp, col: str) -> Optional[float]:
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
    return float(v)


def _exp_state_at_ts(g: pd.DataFrame, ts: pd.Timestamp, col: str) -> Optional[int]:
    v = _exp_at_ts(g, ts, col)
    if v is None:
        return None
    return int(v)


def _returns_from_closes(df4h: pd.DataFrame, asof_ts: pd.Timestamp, kind: str, scale: float) -> Optional[pd.Series]:
    # last completed 4h close <= asof_ts
    cut = df4h.loc[df4h.index <= asof_ts]
    if len(cut) < 3:
        return None
    close = cut["close"].astype(float)
    if kind == "log":
        ret = np.log(close).diff()
    elif kind == "pct":
        ret = close.pct_change()
    else:
        raise ValueError(f"unknown returns kind={kind!r}")
    ret = (ret * float(scale)).dropna()
    return ret


def _fit_markov(ret: pd.Series, trend: str, switching_variance: bool) -> MarkovRegression:
    # 2-regime Markov Regression, Gaussian innovations (statsmodels default)
    mod = MarkovRegression(
        ret.values,
        k_regimes=2,
        trend=trend,
        switching_variance=switching_variance,
    )
    # Use a deterministic fit configuration
    res = mod.fit(disp=False)
    return res


def _pick_up_regime(res: MarkovRegression) -> int:
    # Identify "up" as regime with higher mean return.
    # trend='c' -> there is a constant per regime.
    params = res.params
    # Statsmodels packs regime-specific intercepts first; safest is to read regime means via expected_value if present.
    # For MarkovRegression, params ordering for trend='c': [const[0], const[1], (variance params...)]
    if len(params) >= 2:
        mu0 = float(params[0])
        mu1 = float(params[1])
        return 1 if mu1 > mu0 else 0
    return 0


@dataclass(frozen=True)
class Variant:
    name: str
    ret_kind: str              # 'log' or 'pct'
    ret_scale: float           # 1.0 or 100.0
    trend: str                 # 'c' or 'n'
    switching_variance: bool   # True/False
    probs_source: str          # 'filtered' or 'smoothed'
    prob_mapping: str          # 'up' or 'state1'
    state_mapping: str         # 'raw' or 'up01'


def _compute_variant(df4h: pd.DataFrame, asof_ts: pd.Timestamp, v: Variant, min_obs: int) -> Optional[Tuple[float, int]]:
    ret = _returns_from_closes(df4h, asof_ts, v.ret_kind, v.ret_scale)
    if ret is None or len(ret) < int(min_obs):
        return None

    res = _fit_markov(ret, trend=v.trend, switching_variance=v.switching_variance)

    if v.probs_source == "filtered":
        probs = res.filtered_marginal_probabilities
    elif v.probs_source == "smoothed":
        probs = res.smoothed_marginal_probabilities
    else:
        raise ValueError(f"bad probs_source={v.probs_source!r}")

    # probs is (T x k_regimes)
    p_last = probs.iloc[-1].astype(float).values
    state_raw = int(np.argmax(p_last))
    up_regime = _pick_up_regime(res)

    if v.prob_mapping == "up":
        prob = float(p_last[up_regime])
    elif v.prob_mapping == "state1":
        prob = float(p_last[1])
    else:
        raise ValueError(f"bad prob_mapping={v.prob_mapping!r}")

    if v.state_mapping == "raw":
        state = state_raw
    elif v.state_mapping == "up01":
        state = 1 if state_raw == up_regime else 0
    else:
        raise ValueError(f"bad state_mapping={v.state_mapping!r}")

    return prob, state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--fixtures-dir", required=True)
    ap.add_argument("--symbol", required=True, help="Fixture symbol to use (usually BTCUSDT for macro Markov)")
    ap.add_argument("--max-ts", type=int, default=60)
    ap.add_argument("--min-obs", type=int, default=80)
    ap.add_argument("--golden-markov-prob-col", default="markov_prob_4h")
    ap.add_argument("--golden-markov-state-col", default="markov_state_4h")
    args = ap.parse_args()

    golden_path = Path(args.golden).expanduser().resolve()
    fixtures_dir = Path(args.fixtures_dir).expanduser().resolve()

    g = _load_golden(golden_path)
    df4h = _load_fixture_4h(fixtures_dir, args.symbol)

    prob_col = args.golden_markov_prob_col
    state_col = args.golden_markov_state_col

    if prob_col not in g.columns:
        raise ValueError(f"golden missing column {prob_col!r}. Available: {sorted(g.columns)[:60]}")
    if state_col not in g.columns:
        raise ValueError(f"golden missing column {state_col!r}. Available: {sorted(g.columns)[:60]}")

    # Candidate timestamps: any timestamp with non-null expected prob (first non-null across symbols).
    cand: List[pd.Timestamp] = []
    for ts in g.index.unique():
        exp_prob = _exp_at_ts(g, ts, prob_col)
        exp_state = _exp_state_at_ts(g, ts, state_col)
        if exp_prob is None or exp_state is None:
            continue
        # must be in fixture coverage (allow a small tail because asof can be slightly after last 4h close)
        if ts < df4h.index.min():
            continue
        if ts > df4h.index.max() + pd.Timedelta("4h"):
            continue
        cand.append(ts)

    cand = sorted(cand)[: int(args.max_ts)]
    if not cand:
        raise AssertionError("No comparable timestamps found between golden and fixture 4h window.")

    # Small but meaningful variant grid (keep runtime bounded).
    variants: List[Variant] = []
    for ret_kind in ["log", "pct"]:
        for probs_source in ["filtered", "smoothed"]:
            for switching_variance in [True, False]:
                for trend in ["c", "n"]:
                    for ret_scale in [1.0, 100.0]:
                        for prob_mapping in ["up", "state1"]:
                            for state_mapping in ["raw", "up01"]:
                                name = f"{ret_kind}|{ret_scale:g}|{trend}|sv={int(switching_variance)}|{probs_source}|p={prob_mapping}|s={state_mapping}"
                                variants.append(
                                    Variant(
                                        name=name,
                                        ret_kind=ret_kind,
                                        ret_scale=ret_scale,
                                        trend=trend,
                                        switching_variance=switching_variance,
                                        probs_source=probs_source,
                                        prob_mapping=prob_mapping,
                                        state_mapping=state_mapping,
                                    )
                                )

    results: List[Tuple[str, int, float, float]] = []
    # tuple: (name, n, mean_abs_prob_err, state_acc)

    for v in variants:
        abs_errs: List[float] = []
        state_hits = 0
        n = 0
        for ts in cand:
            exp_prob = _exp_at_ts(g, ts, prob_col)
            exp_state = _exp_state_at_ts(g, ts, state_col)
            if exp_prob is None or exp_state is None:
                continue

            out = _compute_variant(df4h, ts, v, min_obs=int(args.min_obs))
            if out is None:
                continue
            prob, state = out
            abs_errs.append(abs(prob - float(exp_prob)))
            state_hits += int(state == int(exp_state))
            n += 1

        if n >= 5:
            mean_abs = float(np.mean(abs_errs)) if abs_errs else float("inf")
            acc = float(state_hits / n)
            results.append((v.name, n, mean_abs, acc))

    if not results:
        raise AssertionError("No variant produced >=5 comparable cases. Try lowering --min-obs or increasing --max-ts.")

    # Rank by prob error then by state accuracy.
    results.sort(key=lambda x: (x[2], -x[3], -x[1]))

    print("Top 15 variants by (mean_abs_prob_err asc, state_acc desc):")
    for name, n, mean_abs, acc in results[:15]:
        print(f"  n={n:3d} mean_abs_prob_err={mean_abs:.6f} state_acc={acc:.3f} :: {name}")

    best = results[0]
    print("\nBEST:", best[0], "n=", best[1], "mean_abs_prob_err=", best[2], "state_acc=", best[3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
