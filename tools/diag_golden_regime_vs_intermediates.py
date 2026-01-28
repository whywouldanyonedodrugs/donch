#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_parquet_ts(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df[df["timestamp"].notna()].copy()
        df = df.sort_values("timestamp")
        df = df.set_index("timestamp", drop=True)
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise RuntimeError(f"{path} has no timestamp column and index is not DatetimeIndex")
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[df.index.notna()].copy()
        df = df.sort_index()
    return df


def _pick_golden_symbol(g: pd.DataFrame, user_symbol: str | None) -> str | None:
    if "symbol" not in g.columns:
        return None
    if user_symbol:
        return user_symbol
    vc = g["symbol"].value_counts()
    return str(vc.index[0]) if len(vc) else None


def _asof_last(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    # expects df indexed by timestamp sorted
    if df.empty:
        return None
    # fast path: if ts < first index
    if ts < df.index[0]:
        return None
    # position via searchsorted
    pos = df.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return df.iloc[int(pos)]


def eval_daily(golden: pd.DataFrame, daily_inter: pd.DataFrame) -> tuple[float, float, int]:
    # golden expected columns
    code_col = "regime_code_1d" if "regime_code_1d" in golden.columns else "regime_code"
    prob_col = "vol_prob_low_1d" if "vol_prob_low_1d" in golden.columns else "vol_prob_low"

    if code_col not in golden.columns or prob_col not in golden.columns:
        raise RuntimeError(f"Golden missing {code_col} or {prob_col}")

    # offline semantics: day = ts.floor("D"), then last daily row <= day
    # daily_inter index is daily labels; ensure we have a daily table
    di = daily_inter.copy()
    di["_day"] = di.index.floor("D")
    di_day = di.groupby("_day").last().sort_index()
    di_day.index.name = "day"

    exp_code = []
    got_code = []
    exp_prob = []
    got_prob = []

    for ts, row in golden.iterrows():
        day = ts.floor("D")
        r = _asof_last(di_day, day)
        if r is None:
            continue
        if pd.isna(row[code_col]) or pd.isna(row[prob_col]):
            continue

        exp_code.append(int(row[code_col]))
        got_code.append(int(r["regime_code"] if "regime_code" in r.index else r.get("regime_code", np.nan)))

        exp_prob.append(float(row[prob_col]))
        got_prob.append(float(r["vol_prob_low"] if "vol_prob_low" in r.index else r.get("vol_prob_low", np.nan)))

    if not exp_code:
        return (np.nan, np.nan, 0)

    exp_code = np.array(exp_code, dtype=int)
    got_code = np.array(got_code, dtype=int)
    exp_prob = np.array(exp_prob, dtype=float)
    got_prob = np.array(got_prob, dtype=float)

    code_acc = float((exp_code == got_code).mean())
    prob_mae = float(np.mean(np.abs(exp_prob - got_prob)))
    return (code_acc, prob_mae, int(len(exp_code)))


def eval_markov(
    golden: pd.DataFrame,
    markov_inter: pd.DataFrame,
    shift_hours: int,
    alpha: float | None,
    use_raw: bool,
) -> tuple[float, float, int]:
    # golden expected columns
    prob_col = "markov_prob_up_4h"
    state_col = "markov_state_4h"
    if prob_col not in golden.columns or state_col not in golden.columns:
        raise RuntimeError(f"Golden missing {prob_col} or {state_col}")

    mi = markov_inter.copy()
    mi.index = mi.index + pd.Timedelta(hours=int(shift_hours))
    mi = mi.sort_index()

    # choose base probability series
    if "prob_up_raw" in mi.columns:
        base = mi["prob_up_raw"].astype(float)
    elif "prob_up" in mi.columns:
        base = mi["prob_up"].astype(float)
    else:
        raise RuntimeError("Markov intermediate missing prob_up_raw/prob_up")

    if use_raw:
        prob = base
    else:
        a = float(alpha)
        prob = base.ewm(alpha=a, adjust=False).mean().clip(0, 1)

    exp_p = []
    got_p = []
    exp_s = []
    got_s = []

    # as-of lookup: last markov row <= ts
    for ts, row in golden.iterrows():
        if pd.isna(row[prob_col]) or pd.isna(row[state_col]):
            continue
        r = _asof_last(pd.DataFrame({"prob": prob}).dropna(), ts)
        if r is None:
            continue
        p = float(r["prob"])
        exp_p.append(float(row[prob_col]))
        got_p.append(p)
        exp_s.append(int(row[state_col]))
        got_s.append(int(p > 0.5))

    if not exp_p:
        return (np.nan, np.nan, 0)

    exp_p = np.array(exp_p, dtype=float)
    got_p = np.array(got_p, dtype=float)
    exp_s = np.array(exp_s, dtype=int)
    got_s = np.array(got_s, dtype=int)

    prob_mae = float(np.mean(np.abs(exp_p - got_p)))
    state_acc = float((exp_s == got_s).mean())
    return (state_acc, prob_mae, int(len(exp_p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--intermediate-dir", required=True)
    ap.add_argument("--golden-symbol", default=None)
    ap.add_argument("--max-rows", type=int, default=300)
    args = ap.parse_args()

    golden_path = Path(args.golden).resolve()
    idir = Path(args.intermediate_dir).resolve()

    g_raw = pd.read_parquet(golden_path)
    # timestamp handling for golden
    if "timestamp" in g_raw.columns:
        g_raw["timestamp"] = pd.to_datetime(g_raw["timestamp"], utc=True, errors="coerce")
        g_raw = g_raw[g_raw["timestamp"].notna()].copy()
        g_raw = g_raw.sort_values("timestamp").set_index("timestamp", drop=True)
    else:
        if not isinstance(g_raw.index, pd.DatetimeIndex):
            raise RuntimeError("Golden has no timestamp and index is not DatetimeIndex")
        g_raw = g_raw.copy()
        g_raw.index = pd.to_datetime(g_raw.index, utc=True, errors="coerce")
        g_raw = g_raw[g_raw.index.notna()].copy().sort_index()

    sym = _pick_golden_symbol(g_raw, args.golden_symbol)
    if sym is not None:
        g = g_raw[g_raw["symbol"] == sym].copy()
    else:
        g = g_raw.copy()

    if args.max_rows and len(g) > args.max_rows:
        g = g.iloc[: args.max_rows].copy()

    print(f"Golden rows used: {len(g)} symbol={sym}")

    # DAILY: try BTC/ETH intermediates
    for s in ["BTCUSDT", "ETHUSDT"]:
        p = idir / f"{s}_daily_regime_intermediate.parquet"
        if not p.exists():
            print(f"DAILY {s}: missing {p}")
            continue
        di = _load_parquet_ts(p)
        code_acc, prob_mae, n = eval_daily(g, di)
        print(f"DAILY {s}: n={n} code_acc={code_acc:.4f} volprob_mae={prob_mae:.6f}")

    # MARKOV: try BTC/ETH, shift 0/4, raw vs ewm(alpha grid)
    alpha_grid = [None] + [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]
    for s in ["BTCUSDT", "ETHUSDT"]:
        p = idir / f"{s}_markov4h_intermediate.parquet"
        if not p.exists():
            print(f"MARKOV {s}: missing {p}")
            continue
        mi = _load_parquet_ts(p)
        best = None
        for shift in [0, 4, -4]:
            # raw
            st_acc, mae, n = eval_markov(g, mi, shift_hours=shift, alpha=0.2, use_raw=True)
            best = min(best, (mae, st_acc, n, s, shift, "raw", None), default=(mae, st_acc, n, s, shift, "raw", None))
            # ewm
            for a in alpha_grid:
                if a is None:
                    continue
                st_acc, mae, n = eval_markov(g, mi, shift_hours=shift, alpha=a, use_raw=False)
                cand = (mae, st_acc, n, s, shift, "ewm", a)
                best = cand if (best is None or cand[0] < best[0]) else best

        if best is None:
            continue
        mae, st_acc, n, s, shift, mode, a = best
        print(f"MARKOV BEST {s}: n={n} prob_mae={mae:.6f} state_acc={st_acc:.4f} shift={shift} mode={mode} alpha={a}")


if __name__ == "__main__":
    main()
