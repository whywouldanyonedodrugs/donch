#!/usr/bin/env python3
"""
Generate a deterministic sizing parity fixture (live-chain vs offline-reference chain).

This is an evidence tool for JT-003. It does not place orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Case:
    case_id: int
    equity: float
    regime_up: int
    btc_trend_slope: float
    btc_vol_regime_level: float
    p_cal: float
    eth_macd_hist_4h: float
    risk_scale: float | None
    entry: float
    sl_initial: float


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _f(cfg: dict, key: str, default: float) -> float:
    try:
        v = float(cfg.get(key, default))
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _b(cfg: dict, key: str, default: bool) -> bool:
    v = cfg.get(key, default)
    if isinstance(v, bool):
        return v
    x = str(v).strip().lower()
    if x in {"1", "true", "yes", "on"}:
        return True
    if x in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def valid_prob_01(p: float) -> bool:
    return np.isfinite(p) and 0.0 <= p <= 1.0


def linear_meta_map(cfg: dict, p: float) -> float:
    p0 = _f(cfg, "META_SIZING_P0", _f(cfg, "WINPROB_PROB_FLOOR", 0.50))
    p1 = _f(cfg, "META_SIZING_P1", _f(cfg, "WINPROB_PROB_CAP", 0.90))
    m0 = _f(cfg, "META_SIZING_MIN", _f(cfg, "WINPROB_MIN_MULTIPLIER", 0.70))
    m1 = _f(cfg, "META_SIZING_MAX", _f(cfg, "WINPROB_MAX_MULTIPLIER", 1.30))
    if p1 <= p0:
        p1 = p0 + 1e-9
    pp = float(np.clip(p, p0, p1))
    x = (pp - p0) / (p1 - p0)
    return float(m0 + x * (m1 - m0))


def dyn_size_multiplier(cfg: dict, p: float, eth_hist: float) -> float:
    size_mult = 1.0
    if _b(cfg, "META_SIZING_ENABLED", True) and valid_prob_01(p):
        size_mult = linear_meta_map(cfg, p)

    hist_thresh = _f(cfg, "DYN_MACD_HIST_THRESH", 0.0)
    regime_down_mult = _f(cfg, "REGIME_DOWNSIZE_MULT", 1.0)
    if np.isfinite(eth_hist) and np.isfinite(hist_thresh) and np.isfinite(regime_down_mult):
        if eth_hist < hist_thresh:
            size_mult *= regime_down_mult

    size_min_cap_raw = cfg.get("SIZE_MIN_CAP", cfg.get("SIZING_MULT_MIN", None))
    size_max_cap_raw = cfg.get("SIZE_MAX_CAP", cfg.get("SIZING_MULT_MAX", None))
    size_min_cap = 0.0 if size_min_cap_raw is None else float(size_min_cap_raw)
    size_max_cap = float("inf") if size_max_cap_raw is None else float(size_max_cap_raw)
    if np.isfinite(size_min_cap) and size_min_cap > 0.0:
        size_mult = max(size_mult, size_min_cap)
    if np.isfinite(size_max_cap) and size_max_cap > 0.0:
        size_mult = min(size_mult, size_max_cap)

    if (not np.isfinite(size_mult)) or size_mult < 0.0:
        return 0.0
    return float(size_mult)


def compute_risk_on(case: Case, btc_vol_hi: float) -> int:
    btc_trend_up = 1 if case.btc_trend_slope > 0.0 else 0
    btc_vol_high = 1 if case.btc_vol_regime_level >= btc_vol_hi else 0
    return 1 if (case.regime_up == 1 and btc_trend_up == 1 and btc_vol_high == 0) else 0


def live_chain(cfg: dict, case: Case) -> dict:
    regime_block = _b(cfg, "REGIME_BLOCK_WHEN_DOWN", True)
    regime_size_down = _f(cfg, "REGIME_SIZE_WHEN_DOWN", 1.0)
    mode = str(cfg.get("RISK_MODE", "fixed") or "fixed").lower()
    fixed_risk = _f(cfg, "FIXED_RISK_CASH", _f(cfg, "RISK_USD", 10.0))
    risk_pct = _f(cfg, "RISK_PCT", _f(cfg, "RISK_EQUITY_PCT", 0.01))
    probe = _f(cfg, "RISK_OFF_PROBE_MULT", 0.01)
    probe_enabled = _b(cfg, "META_RISK_OFF_PROBE_ENABLED", True)
    btc_vol_hi = _f(cfg, "BTC_VOL_HI", 0.75)
    notional_cap_pct = _f(cfg, "NOTIONAL_CAP_PCT_OF_EQUITY", 1.0)
    max_lev = _f(cfg, "MAX_LEVERAGE", 1.0)

    if regime_block and case.regime_up == 0:
        return {
            "skipped": True,
            "skip_reason": "regime_down",
            "risk_on": compute_risk_on(case, btc_vol_hi),
            "size_mult_pre_probe": 0.0,
            "size_mult_final": 0.0,
            "equity_for_sizing": case.equity,
            "risk_usd": 0.0,
            "qty": 0.0,
        }

    equity_for_sizing = float(case.equity)
    if case.regime_up == 0 and not regime_block:
        equity_for_sizing *= regime_size_down

    if case.risk_scale is not None and np.isfinite(case.risk_scale):
        size_mult_pre_probe = max(0.0, float(case.risk_scale))
    else:
        size_mult_pre_probe = max(0.0, float(dyn_size_multiplier(cfg, case.p_cal, case.eth_macd_hist_4h)))

    risk_on = compute_risk_on(case, btc_vol_hi)
    size_mult_final = float(size_mult_pre_probe)
    if probe_enabled and risk_on != 1 and np.isfinite(probe) and probe > 0.0:
        size_mult_final = min(size_mult_final, probe)
    size_mult_final = max(0.0, float(size_mult_final))

    if mode in {"percent", "pct"}:
        risk_pct_override = risk_pct * size_mult_final
        risk_usd = max(0.0, equity_for_sizing * risk_pct_override)
    else:
        fixed_cash_override = fixed_risk * size_mult_final
        risk_usd = max(0.0, fixed_cash_override)

    risk_per_unit = abs(float(case.entry) - float(case.sl_initial))
    qty = 0.0 if risk_per_unit <= 0.0 else (risk_usd / risk_per_unit)
    max_notional = equity_for_sizing * max(0.0, notional_cap_pct) * max(0.0, max_lev)
    if case.entry > 0 and qty * case.entry > max_notional:
        qty = max_notional / case.entry

    return {
        "skipped": False,
        "skip_reason": "",
        "risk_on": risk_on,
        "size_mult_pre_probe": size_mult_pre_probe,
        "size_mult_final": size_mult_final,
        "equity_for_sizing": equity_for_sizing,
        "risk_usd": float(max(0.0, risk_usd)),
        "qty": float(max(0.0, qty)),
    }


def offline_reference_chain(cfg: dict, case: Case) -> dict:
    # Independent implementation from ticket semantics.
    regime_block = _b(cfg, "REGIME_BLOCK_WHEN_DOWN", True)
    regime_size_when_down = _f(cfg, "REGIME_SIZE_WHEN_DOWN", 1.0)
    risk_mode = str(cfg.get("RISK_MODE", "fixed") or "fixed").lower()
    fixed_risk_cash = _f(cfg, "FIXED_RISK_CASH", _f(cfg, "RISK_USD", 10.0))
    risk_pct = _f(cfg, "RISK_PCT", _f(cfg, "RISK_EQUITY_PCT", 0.01))
    btc_vol_hi = _f(cfg, "BTC_VOL_HI", 0.75)
    probe = _f(cfg, "RISK_OFF_PROBE_MULT", 0.01)
    notional_cap = _f(cfg, "NOTIONAL_CAP_PCT_OF_EQUITY", 1.0)
    max_lev = _f(cfg, "MAX_LEVERAGE", 1.0)

    if regime_block and case.regime_up == 0:
        return {
            "skipped": True,
            "skip_reason": "regime_down",
            "risk_on": compute_risk_on(case, btc_vol_hi),
            "size_mult_pre_probe": 0.0,
            "size_mult_final": 0.0,
            "equity_for_sizing": case.equity,
            "risk_usd": 0.0,
            "qty": 0.0,
        }

    equity_for_sizing = float(case.equity)
    if case.regime_up == 0 and not regime_block:
        equity_for_sizing *= regime_size_when_down

    if case.risk_scale is not None and np.isfinite(case.risk_scale):
        size_mult = float(case.risk_scale)
    else:
        size_mult = dyn_size_multiplier(cfg, case.p_cal, case.eth_macd_hist_4h)

    size_mult = max(0.0, float(size_mult))

    risk_on = compute_risk_on(case, btc_vol_hi)
    if risk_on == 0 and np.isfinite(probe) and probe > 0.0:
        size_mult = min(size_mult, probe)

    if risk_mode in {"percent", "pct"}:
        risk_pct_override = risk_pct * size_mult
        cash_risk = max(0.0, equity_for_sizing * risk_pct_override)
    else:
        fixed_cash_override = fixed_risk_cash * size_mult
        cash_risk = max(0.0, fixed_cash_override)

    risk_per_unit = abs(float(case.entry) - float(case.sl_initial))
    qty = 0.0 if risk_per_unit <= 0.0 else (cash_risk / risk_per_unit)

    max_notional = equity_for_sizing * max(0.0, notional_cap) * max(0.0, max_lev)
    if case.entry > 0 and qty * case.entry > max_notional:
        qty = max_notional / case.entry

    return {
        "skipped": False,
        "skip_reason": "",
        "risk_on": risk_on,
        "size_mult_pre_probe": size_mult,
        "size_mult_final": size_mult,
        "equity_for_sizing": equity_for_sizing,
        "risk_usd": float(cash_risk),
        "qty": float(max(0.0, qty)),
    }


def make_cases(n: int, seed: int) -> list[Case]:
    rng = np.random.default_rng(seed)
    out: list[Case] = []
    for i in range(n):
        equity = float(rng.uniform(500.0, 5000.0))
        regime_up = int(rng.integers(0, 2))
        btc_trend_slope = float(rng.normal(0.0, 0.02))
        btc_vol_level = float(rng.uniform(0.2, 1.4))
        p_cal = float(rng.uniform(0.0, 1.0))
        eth_hist = float(rng.normal(0.0, 0.4))
        if rng.uniform() < 0.25:
            risk_scale = float(rng.uniform(0.01, 2.5))
        else:
            risk_scale = None

        entry = float(rng.uniform(0.2, 300.0))
        dist = max(1e-4, float(entry * rng.uniform(0.002, 0.08)))
        if rng.uniform() < 0.5:
            sl_initial = entry - dist
        else:
            sl_initial = entry + dist

        out.append(
            Case(
                case_id=i,
                equity=equity,
                regime_up=regime_up,
                btc_trend_slope=btc_trend_slope,
                btc_vol_regime_level=btc_vol_level,
                p_cal=p_cal,
                eth_macd_hist_4h=eth_hist,
                risk_scale=risk_scale,
                entry=entry,
                sl_initial=sl_initial,
            )
        )
    return out


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    arr = np.array(xs, dtype=float)
    return float(np.percentile(arr, q))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate live-vs-offline sizing parity fixture CSV.")
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config-path", default="config.yaml")
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--summary-json", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(Path(args.config_path))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_csv = Path(args.output_csv or f"results/autopar_exports/sizing_parity_fixture_{stamp}.csv")
    out_json = Path(args.summary_json or f"results/autopar_exports/sizing_parity_fixture_{stamp}.summary.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict] = []
    err_size: list[float] = []
    err_risk: list[float] = []

    for c in make_cases(int(args.rows), int(args.seed)):
        lv = live_chain(cfg, c)
        bt = offline_reference_chain(cfg, c)

        size_err = abs(float(lv["size_mult_final"]) - float(bt["size_mult_final"]))
        risk_err = abs(float(lv["risk_usd"]) - float(bt["risk_usd"]))
        qty_err = abs(float(lv["qty"]) - float(bt["qty"]))
        err_size.append(size_err)
        err_risk.append(risk_err)

        rows_out.append(
            {
                "case_id": c.case_id,
                "equity": c.equity,
                "regime_up": c.regime_up,
                "btc_trend_slope": c.btc_trend_slope,
                "btc_vol_regime_level": c.btc_vol_regime_level,
                "p_cal": c.p_cal,
                "eth_macd_hist_4h": c.eth_macd_hist_4h,
                "risk_scale": "" if c.risk_scale is None else c.risk_scale,
                "entry": c.entry,
                "sl_initial": c.sl_initial,
                "risk_on_live": lv["risk_on"],
                "risk_on_bt": bt["risk_on"],
                "size_mult_live": lv["size_mult_final"],
                "size_mult_bt": bt["size_mult_final"],
                "size_mult_abs_err": size_err,
                "risk_usd_live": lv["risk_usd"],
                "risk_usd_bt": bt["risk_usd"],
                "risk_usd_abs_err": risk_err,
                "qty_live": lv["qty"],
                "qty_bt": bt["qty"],
                "qty_abs_err": qty_err,
            }
        )

    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows_out:
            wr.writerow(r)

    summary = {
        "rows": len(rows_out),
        "size_mult_mean_abs_error": float(np.mean(err_size)) if err_size else 0.0,
        "size_mult_p90_abs_error": percentile(err_size, 90),
        "risk_usd_mean_abs_error": float(np.mean(err_risk)) if err_risk else 0.0,
        "risk_usd_p90_abs_error": percentile(err_risk, 90),
        "max_size_mult_abs_error": max(err_size) if err_size else 0.0,
        "max_risk_usd_abs_error": max(err_risk) if err_risk else 0.0,
        "config_path": str(args.config_path),
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[OK] fixture_csv={out_csv}")
    print(f"[OK] summary_json={out_json}")
    print(
        "[OK] size_mult_mean_abs_error=%.12f size_mult_p90_abs_error=%.12f risk_usd_mean_abs_error=%.12f risk_usd_p90_abs_error=%.12f"
        % (
            summary["size_mult_mean_abs_error"],
            summary["size_mult_p90_abs_error"],
            summary["risk_usd_mean_abs_error"],
            summary["risk_usd_p90_abs_error"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
