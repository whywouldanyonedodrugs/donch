#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

def _pick_col_by_suffix(cols, suffixes):
    for suf in suffixes:
        cands = [c for c in cols if c.endswith(suf)]
        if cands:
            # shortest tends to pick non-prefixed before long variants, but still finds S1_*
            return sorted(cands, key=len)[0]
    return None

def _pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def _load_fixture_as_open_ts(fixtures_dir: Path, symbol: str, tf_tag: str) -> pd.DataFrame:
    """
    Loads <SYMBOL>_<TF>.parquet where 'timestamp' is BAR CLOSE (UTC),
    then shifts index back by TF to get BAR OPEN timestamps (CCXT-style).
    """
    p = fixtures_dir / f"{symbol.upper()}_{tf_tag.upper()}.parquet"
    if not p.exists():
        raise FileNotFoundError(str(p))

    df = pd.read_parquet(p)
    if "timestamp" not in df.columns:
        raise ValueError(f"{p} has no 'timestamp' column")

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp", drop=True)

    need = {"open", "high", "low", "close", "volume"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise ValueError(f"{p} missing required OHLCV columns: {missing}")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).sort_index()

    td = pd.Timedelta(tf_tag)  # '1D' and '4H' are valid
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True) - td
    df.index.name = "timestamp"
    return df


def _slice_completed_open_bars(df_open: pd.DataFrame, asof_ts: pd.Timestamp, tf_tag: str) -> pd.DataFrame:
    """
    Keep only fully-completed bars as-of asof_ts, assuming df is indexed by BAR OPEN timestamps.
    The bar whose open == asof_ts.floor(tf) is IN-PROGRESS and must be excluded.
    """
    asof_ts = pd.to_datetime(asof_ts, utc=True)
    floor = asof_ts.floor(tf_tag)
    return df_open.loc[df_open.index < floor].copy()


def _fit_markov(ret: pd.Series, min_obs: int) -> Tuple[Optional[object], Dict[str, float]]:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    info: Dict[str, float] = {"nobs": float(len(ret))}
    if len(ret) < min_obs:
        return None, None  # caller should skip this ts


    mod = MarkovRegression(ret, k_regimes=2, trend="c", switching_variance=True)

    # Try to keep fitting deterministic across runs.
    try:
        res = mod.fit(disp=False, search_reps=0)
    except TypeError:
        res = mod.fit(disp=False)

    return res, info


def _prob_variant(res, kind: str, which_state: int) -> pd.Series:
    """
    kind: 'filtered' or 'smoothed'
    returns a probability series for the chosen latent regime index (0 or 1)
    """
    if kind == "filtered":
        probs = res.filtered_marginal_probabilities
    elif kind == "smoothed":
        probs = res.smoothed_marginal_probabilities
    else:
        raise ValueError(kind)

    # probs may be DataFrame keyed by 0/1
    return pd.Series(probs[which_state], index=probs.index)


def _extract_params(res) -> Dict[str, float]:
    names = getattr(res, "param_names", None)
    params = getattr(res, "params", None)
    if names is None or params is None:
        return {}
    return {str(k): float(v) for k, v in zip(list(names), list(params))}


def _pick_state_by_mean(params: Dict[str, float]) -> int:
    # statsmodels commonly uses const[0]/const[1] for trend='c'
    m0 = params.get("const[0]", params.get("intercept[0]", np.nan))
    m1 = params.get("const[1]", params.get("intercept[1]", np.nan))
    if np.isnan(m0) or np.isnan(m1):
        return 1  # fallback
    return 0 if m0 >= m1 else 1


def _pick_state_by_variance(params: Dict[str, float]) -> int:
    v0 = params.get("sigma2[0]", np.nan)
    v1 = params.get("sigma2[1]", np.nan)
    if np.isnan(v0) or np.isnan(v1):
        return 1  # fallback
    return 0 if v0 >= v1 else 1


def _series_last(x: pd.Series) -> float:
    if x.empty:
        return float("nan")
    return float(x.iloc[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="golden_features.parquet")
    ap.add_argument("--fixtures-dir", required=True, help="Directory with <SYM>_1D.parquet and <SYM>_4H.parquet")
    ap.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    ap.add_argument("--max-ts", type=int, default=60, help="How many golden timestamps to test (from start)")
    ap.add_argument("--min-obs", type=int, default=200, help="Minimum returns required to fit Markov")
    ap.add_argument("--alpha", type=float, default=0.20, help="EWM alpha for smoothing (0 disables)")
    ap.add_argument("--golden-markov-prob-col", default="markov_prob_4h")
    args = ap.parse_args()

    golden_p = Path(args.golden).expanduser().resolve()
    fixtures_dir = Path(args.fixtures_dir).expanduser().resolve()
    sym = args.symbol.upper()

    g = pd.read_parquet(golden_p)
    if "timestamp" in g.columns:
        g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
        g = g.dropna(subset=["timestamp"]).set_index("timestamp", drop=True)
    elif isinstance(g.index, pd.DatetimeIndex):
        g.index = pd.to_datetime(g.index, utc=True, errors="coerce")
    else:
        raise ValueError("golden parquet must have timestamp column or DatetimeIndex")

    cols = list(g.columns)

    # Prefer explicit canonical names, but fall back to suffix search.
    gold_prob_col = _pick_col(cols, [args.golden_markov_prob_col]) if args.golden_markov_prob_col else None
    if gold_prob_col is None:
        gold_prob_col = _pick_col(cols, ["markov_prob_up_4h", "markov_prob_4h"])
    if gold_prob_col is None:
        gold_prob_col = _pick_col_by_suffix(cols, ["markov_prob_up_4h", "markov_prob_4h"])

    gold_state_col = _pick_col(cols, [args.golden_markov_state_col]) if args.golden_markov_state_col else None
    if gold_state_col is None:
        gold_state_col = _pick_col(cols, ["markov_state_up_4h", "markov_state_4h"])
    if gold_state_col is None:
        gold_state_col = _pick_col_by_suffix(cols, ["markov_state_up_4h", "markov_state_4h"])

    if gold_prob_col is None:
        raise ValueError(
            "Could not find Markov prob column in golden. "
            "Expected something like markov_prob_up_4h. "
            f"Available (sample): {sorted(cols)[:60]}"
        )


    if "symbol" in g.columns:
        g = g[g["symbol"].astype(str).str.upper() == sym]

    g = g.sort_index()
    if args.golden_markov_prob_col not in g.columns:
        raise ValueError(f"golden missing column {args.golden_markov_prob_col!r}. Available: {sorted(g.columns)[:50]}")

    ts_list = list(g.index[: int(args.max_ts)])
    if not ts_list:
        raise ValueError("No timestamps selected from golden")

    daily = _load_fixture_as_open_ts(fixtures_dir, sym, "1D")
    h4 = _load_fixture_as_open_ts(fixtures_dir, sym, "4H")

    print(f"Golden rows used: {len(ts_list)}  |  Fixture daily rows={len(daily)}  4h rows={len(h4)}")
    print(f"Fixture daily min={daily.index.min()} max={daily.index.max()}")
    print(f"Fixture 4h   min={h4.index.min()} max={h4.index.max()}")

    for ts in ts_list:
        exp_prob = float(g.loc[ts, args.golden_markov_prob_col])

        h4_use = _slice_completed_open_bars(h4, ts, "4H")
        close = h4_use["close"]

        # Two return definitions
        ret_log = np.log(close).diff().dropna()
        ret_pct = close.pct_change().dropna()

        print("\nTS:", ts, "| exp_prob=", f"{exp_prob:.6f}", "| 4h_bars=", len(h4_use), "log_ret_n=", len(ret_log), "pct_ret_n=", len(ret_pct))

        for ret_name, ret in [("log", ret_log), ("pct", ret_pct)]:
            res, params = _fit_markov(ret, min_obs=args.min_obs)
            if res is None:
                return {"skip": f"insufficient returns: {len(ret)} (<{args.min_obs})"}

                continue

            params = _extract_params(res)
            up_by_mean = _pick_state_by_mean(params)
            hi_by_var = _pick_state_by_variance(params)

            for prob_kind in ["filtered", "smoothed"]:
                for pick_name, state_idx in [("state=1", 1), ("up_by_mean", up_by_mean), ("hi_by_var", hi_by_var)]:
                    pser = _prob_variant(res, prob_kind, state_idx)
                    if args.alpha and args.alpha > 0:
                        pser = pser.ewm(alpha=float(args.alpha), adjust=False).mean()
                    got = _series_last(pser)
                    if np.isnan(got):
                        continue
                    err = abs(got - exp_prob)
                    print(f"  {ret_name} {prob_kind} {pick_name}: got={got:.6f} | abs_err={err:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
