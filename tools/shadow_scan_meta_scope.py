#!/usr/bin/env python3
"""
tools/shadow_scan_meta_scope.py

Shadow-scan historical timestamps to find cases where meta scope passes (scope_ok=True),
and optionally where p_cal is above/below pstar.

Designed to run on the data machine with local parquet + meta bundle.
- No wall-clock leakage: everything is evaluated "as-of decision_ts".
- Uses eval_meta_scope() to mirror live scope logic.
- Best-effort introspection to call FeatureBuilder/WinProbScorer without hardcoding exact signatures.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable

import numpy as np
import pandas as pd

# Ensure project root is importable when running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg  # noqa: E402
from live.shared_utils import load_parquet_data  # noqa: E402
from live.feature_builder import FeatureBuilder  # noqa: E402
from live.parity_utils import eval_meta_scope  # noqa: E402

try:
    from live.winprob_loader import WinProbScorer  # noqa: E402
except Exception:
    WinProbScorer = None  # type: ignore


def _parse_ts(s: str) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _floor_5m(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.floor("5min")


def _call_with_signature(fn: Callable[..., Any], **candidates: Any) -> Any:
    """
    Call fn by matching provided candidate kwargs to its signature.
    Unmatched candidates are ignored.
    """
    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name in candidates:
            kwargs[name] = candidates[name]
    return fn(**kwargs)


def _build_meta_row(builder: Any, *, df5: pd.DataFrame, decision_ts: pd.Timestamp, symbol: str,
                    ctx: Optional[Dict[str, Any]] = None, univ: Optional[Dict[str, Any]] = None,
                    signal: Any = None) -> Dict[str, Any]:
    """
    Try common meta-row builder method names, introspecting signatures.
    Fails loudly if none found.
    """
    ctx = ctx or {}
    univ = univ or {}

    method_names = [
        "build_meta_row",
        "compute_meta_row",
        "compute_meta_features",
        "build_row",
        "compute_row",
    ]

    for name in method_names:
        fn = getattr(builder, name, None)
        if callable(fn):
            out = _call_with_signature(
                fn,
                df5=df5, df=df5, ohlcv=df5, ohlcv_5m=df5, df_5m=df5,
                decision_ts=decision_ts, ts=decision_ts, timestamp=decision_ts, t=decision_ts,
                symbol=symbol, sym=symbol,
                ctx=ctx, context=ctx,
                univ=univ, universe=univ, universe_ctx=univ,
                signal=signal, signal_obj=signal,
            )
            if not isinstance(out, dict):
                raise RuntimeError(f"{name}() returned {type(out)}; expected dict meta_row")
            return out

    avail = [x for x in dir(builder) if "meta" in x.lower() or "row" in x.lower()]
    raise RuntimeError(
        "Could not find a meta-row builder method on FeatureBuilder.\n"
        f"Tried: {method_names}\n"
        f"Available (filtered): {avail}"
    )


def _init_scorer(meta_dir: Path) -> Any:
    if WinProbScorer is None:
        raise RuntimeError("WinProbScorer import failed; cannot score p_cal/pstar")

    sig = inspect.signature(WinProbScorer)
    kwargs: Dict[str, Any] = {}

    # Try common constructor params
    if "meta_dir" in sig.parameters:
        kwargs["meta_dir"] = meta_dir
    elif "model_dir" in sig.parameters:
        kwargs["model_dir"] = meta_dir
    elif "bundle_dir" in sig.parameters:
        kwargs["bundle_dir"] = meta_dir

    return WinProbScorer(**kwargs)


def _score_meta_row(scorer: Any, meta_row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], bool, Optional[str], Optional[str]]:
    """
    Returns: (p_cal, pstar, schema_ok, pstar_scope, err)
    Supports multiple scorer return shapes.
    """
    # pick a scoring method
    fn = None
    for name in ("score_row", "score", "predict", "predict_row"):
        cand = getattr(scorer, name, None)
        if callable(cand):
            fn = cand
            break
    if fn is None:
        raise RuntimeError("WinProbScorer has no usable scoring method (score_row/score/predict/...)")

    out = fn(meta_row)

    p_cal = None
    pstar = getattr(scorer, "pstar", None)
    pstar_scope = getattr(scorer, "pstar_scope", None)
    schema_ok = True
    err = None

    if isinstance(out, dict):
        # common dict keys
        p_cal = out.get("p_cal", out.get("p", out.get("prob", None)))
        schema_ok = bool(out.get("schema_ok", True))
        err = out.get("err", None)
        pstar = out.get("pstar", pstar)
        pstar_scope = out.get("pstar_scope", pstar_scope)
    elif isinstance(out, tuple) or isinstance(out, list):
        # best-effort: look for floats + schema_ok + err
        vals = list(out)
        # pull schema_ok if present
        for v in vals:
            if isinstance(v, bool):
                schema_ok = v
                break
        # pull err if present
        for v in vals:
            if isinstance(v, str) and v:
                err = v
                break
        # pull first finite float as p_cal
        for v in vals:
            try:
                fv = float(v)
                if np.isfinite(fv):
                    p_cal = fv
                    break
            except Exception:
                continue

    # normalize p_cal
    try:
        p_cal = float(p_cal) if p_cal is not None else None
    except Exception:
        p_cal = None

    try:
        pstar = float(pstar) if pstar is not None else None
    except Exception:
        pstar = None

    return p_cal, pstar, schema_ok, (str(pstar_scope) if pstar_scope is not None else None), err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True, help="Root parquet directory (sets cfg.PARQUET_DIR)")
    ap.add_argument("--symbol", required=True, help="Symbol, e.g. BTCUSDT")
    ap.add_argument("--start", required=True, help="Start timestamp (UTC), e.g. 2025-12-01T00:00:00Z")
    ap.add_argument("--end", required=True, help="End timestamp (UTC), e.g. 2026-01-01T00:00:00Z")
    ap.add_argument("--step-min", type=int, default=60, help="Step minutes between evaluated decision_ts")
    ap.add_argument("--strategy-yaml", required=True, help="Strategy YAML to derive FeatureBuilder cfg")
    ap.add_argument("--meta-dir", default=None, help="Meta bundle dir (e.g. results/meta_export). If set, will score p_cal vs pstar.")
    ap.add_argument("--max-print", type=int, default=50, help="Max rows to print")
    args = ap.parse_args()

    # Set parquet dir for loader
    cfg.PARQUET_DIR = args.parquet_dir

    symbol = args.symbol.upper()
    start = _floor_5m(_parse_ts(args.start))
    end = _floor_5m(_parse_ts(args.end))
    step = pd.Timedelta(minutes=int(args.step_min))

    # Load base 5m bars (no wall-clock)
    df5 = load_parquet_data(
        symbol,
        start_date=start,
        end_date=end,
        drop_last_partial=True,
        columns=["open", "high", "low", "close", "volume"],
    )
    if df5 is None or df5.empty:
        print(f"[shadow_scan] No data for {symbol} in range.", flush=True)
        return 2
    if not isinstance(df5.index, pd.DatetimeIndex):
        raise RuntimeError("load_parquet_data did not return a DatetimeIndex")
    if df5.index.tz is None:
        df5.index = df5.index.tz_localize("UTC")
    else:
        df5.index = df5.index.tz_convert("UTC")

    # Load strategy cfg for FeatureBuilder
    import yaml
    strat = yaml.safe_load(Path(args.strategy_yaml).read_text(encoding="utf-8"))
    builder = FeatureBuilder(strat if isinstance(strat, dict) else {})

    scorer = None
    pstar_scope = None
    pstar = None
    if args.meta_dir:
        scorer = _init_scorer(Path(args.meta_dir))
        pstar_scope = getattr(scorer, "pstar_scope", None)
        pstar = getattr(scorer, "pstar", None)

    ts_list = []
    t = start
    while t <= end:
        ts_list.append(t)
        t = t + step

    printed = 0
    n_scope_ok = 0
    n_scored = 0

    for decision_ts in ts_list:
        if decision_ts not in df5.index:
            # find nearest prior 5m
            prior = df5.index[df5.index <= decision_ts]
            if len(prior) == 0:
                continue
            decision_ts = prior[-1]

        # Build meta_row (introspected)
        try:
            meta_row = _build_meta_row(builder, df5=df5, decision_ts=decision_ts, symbol=symbol)
        except Exception as e:
            raise RuntimeError(f"Failed building meta_row at {decision_ts.isoformat()}: {e!r}")

        # Scope evaluation (mirror live)
        scope_ok, info = eval_meta_scope(pstar_scope, meta_row)

        if scope_ok:
            n_scope_ok += 1

        # Optional scoring
        p_cal = None
        schema_ok = True
        err = None
        if scorer is not None:
            p_cal, pstar, schema_ok, pstar_scope2, err = _score_meta_row(scorer, meta_row)
            if pstar_scope is None and pstar_scope2 is not None:
                pstar_scope = pstar_scope2
            n_scored += 1

        if scope_ok and printed < args.max_print:
            if p_cal is not None and pstar is not None:
                side = "ABOVE" if (p_cal >= pstar) else "BELOW"
                print(
                    f"{decision_ts.isoformat()} scope_ok=True "
                    f"risk_on_1={info.get('risk_on_1_raw')} risk_on={info.get('risk_on_raw')} "
                    f"scope_val={info.get('scope_val')} src={info.get('scope_src')} "
                    f"p_cal={p_cal:.4f} pstar={pstar:.4f} {side} schema_ok={schema_ok} err={err}",
                    flush=True
                )
            else:
                print(
                    f"{decision_ts.isoformat()} scope_ok=True "
                    f"risk_on_1={info.get('risk_on_1_raw')} risk_on={info.get('risk_on_raw')} "
                    f"scope_val={info.get('scope_val')} src={info.get('scope_src')}",
                    flush=True
                )
            printed += 1

    print(
        f"[shadow_scan] done: checked={len(ts_list)} scope_ok={n_scope_ok} "
        f"scored={n_scored} pstar_scope={pstar_scope} pstar={pstar}",
        flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
