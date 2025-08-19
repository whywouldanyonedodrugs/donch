# calibrate_and_export.py
# ------------------------------------------------------------
# Fit & select a probability calibrator using OOS predictions,
# following the "simple + improves ECE/Brier + preserves resolution"
# decision rule. Exports:
#   - calibrator.joblib    (call on p_raw in live)
#   - calibration_report.json
#   - oos_predictions_calibrated.parquet  (with y_proba_cal)
#   - (optional) ev_curve_calibrated.csv and pstar.txt, if trades supplied
#
# Usage (basic):
#   python calibrate_and_export.py \
#       --oos results/meta_export/oos_predictions.parquet \
#       --out results/meta_export
#
# With trades EV + threshold selection:
#   python calibrate_and_export.py \
#       --oos results/meta_export/oos_predictions.parquet \
#       --trades results/trades.csv \
#       --pred-ts entry_ts --pred-sym symbol \
#       --trades-ts entry_ts --trades-sym symbol \
#       --out results/meta_export --min-trades 150
#
# ------------------------------------------------------------
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

# ------------------------
# Utility metrics
# ------------------------
def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width bins on [0,1]."""
    p = np.asarray(p)
    y = np.asarray(y_true).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    ece = 0.0
    m = len(p)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        weight = mask.mean()
        ece += weight * abs(acc - conf)
    return float(ece)

def resolution_stats(p_raw: np.ndarray, p_cal: np.ndarray) -> Dict[str, float]:
    """How much resolution is preserved by calibration."""
    p_raw = np.asarray(p_raw)
    p_cal = np.asarray(p_cal)
    std_ratio = float(np.std(p_cal) / (np.std(p_raw) + 1e-12))
    # round to reduce floating noise when counting unique
    uraw = len(np.unique(np.round(p_raw, 6)))
    ucal = len(np.unique(np.round(p_cal, 6)))
    unique_ratio = float(ucal / max(1, uraw))
    return {"std_ratio": std_ratio, "unique_ratio": unique_ratio}

# ------------------------
# Calibrator interfaces
# ------------------------
class IdentityCalibrator:
    kind = "raw"
    def fit(self, p: np.ndarray, y: np.ndarray):
        return self
    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p).astype(float)

class PlattCalibrator:
    kind = "platt"
    def __init__(self):
        self.lr = LogisticRegression(max_iter=5000, solver="lbfgs")
    def fit(self, p: np.ndarray, y: np.ndarray):
        self.lr.fit(np.asarray(p).reshape(-1,1), np.asarray(y).astype(int))
        return self
    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(np.asarray(p).reshape(-1,1))[:,1]

class IsotonicCalibrator:
    kind = "isotonic"
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    def fit(self, p: np.ndarray, y: np.ndarray):
        self.iso.fit(np.asarray(p), np.asarray(y).astype(int))
        return self
    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        return self.iso.predict(np.asarray(p))

class BinnedCalibrator:
    """
    Quantile-binned reliability mapping + monotone smoothing + linear interpolation.
    More data efficient than full isotonic; keeps continuity between bins.
    """
    kind = "binned"
    def __init__(self, n_bins: int = 12, min_bin: int = 40, enforce_monotone: bool = True):
        self.n_bins = int(n_bins)
        self.min_bin = int(min_bin)
        self.enforce_monotone = bool(enforce_monotone)
        self.bin_centers_: Optional[np.ndarray] = None
        self.bin_rates_: Optional[np.ndarray] = None

    @staticmethod
    def _merge_small_bins(edges: np.ndarray, idx: np.ndarray, min_bin: int) -> Tuple[np.ndarray, np.ndarray]:
        """Merge adjacent bins until each has at least min_bin items."""
        n_bins = len(edges) - 1
        counts = np.array([(idx == b).sum() for b in range(n_bins)], dtype=int)
        lefts = list(edges[:-1])
        rights = list(edges[1:])
        # Greedy merge smallest bin with its heavier neighbor
        while True:
            if len(counts) == 1 or (counts >= min_bin).all():
                break
            b = int(np.argmin(counts))
            if b == 0:
                # merge 0 -> 1
                counts[1] += counts[0]
                lefts.pop(0); rights.pop(0)
                counts = counts[1:]
            elif b == len(counts) - 1:
                # merge last-1 -> last
                counts[-2] += counts[-1]
                lefts.pop(-1); rights.pop(-1)
                counts = counts[:-1]
            else:
                # merge into heavier neighbor
                if counts[b-1] >= counts[b+1]:
                    # merge b into b-1
                    counts[b-1] += counts[b]
                    lefts.pop(b); rights.pop(b)
                    counts = np.delete(counts, b)
                else:
                    # merge b into b+1
                    counts[b+1] += counts[b]
                    lefts.pop(b); rights.pop(b)
                    counts = np.delete(counts, b)
        new_edges = np.array(lefts + [rights[-1]], dtype=float)
        # rebuild new idx for each sample
        return new_edges, counts

    def fit(self, p: np.ndarray, y: np.ndarray):
        p = np.asarray(p).astype(float)
        y = np.asarray(y).astype(int)

        # initial quantile edges
        q = np.linspace(0, 1, self.n_bins + 1)
        edges = np.quantile(p, q)
        edges[0], edges[-1] = 0.0, 1.0  # make well-defined support
        # ensure strictly increasing edges (handle ties)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = np.nextafter(edges[i-1], 1.0)

        # initial binning
        idx = np.digitize(p, edges, right=True) - 1
        idx = np.clip(idx, 0, len(edges) - 2)

        # merge bins to enforce min_bin samples/bin
        edges2, _ = self._merge_small_bins(edges, idx, self.min_bin)
        idx = np.digitize(p, edges2, right=True) - 1
        idx = np.clip(idx, 0, len(edges2) - 2)

        # empirical rates per bin
        centers, rates, weights = [], [], []
        for b in range(len(edges2) - 1):
            mask = idx == b
            if not np.any(mask):
                continue
            centers.append(0.5 * (edges2[b] + edges2[b+1]))
            rates.append(y[mask].mean())
            weights.append(mask.sum())
        centers = np.asarray(centers, dtype=float)
        rates = np.asarray(rates, dtype=float)
        weights = np.asarray(weights, dtype=float)

        # optional monotone smoothing (PAV on bin centers)
        if self.enforce_monotone and len(centers) >= 3:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            rates = iso.fit_predict(centers, rates, sample_weight=weights)

        self.bin_centers_ = centers
        self.bin_rates_ = rates
        return self

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        if self.bin_centers_ is None or self.bin_rates_ is None:
            raise RuntimeError("BinnedCalibrator not fitted.")
        x = np.asarray(p).astype(float)
        # continuous linear interpolation; clip to [0,1]
        y = np.interp(x, self.bin_centers_, self.bin_rates_, left=self.bin_rates_[0], right=self.bin_rates_[-1])
        return np.clip(y, 0.0, 1.0)

# ------------------------
# Evaluation + selection
# ------------------------
@dataclass
class RuleParams:
    min_ece_gain: float = 0.005
    min_brier_gain: float = 0.002
    min_std_ratio: float = 0.50
    min_unique_ratio: float = 0.50
    n_bins_ece: int = 15

def evaluate_calibration(y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray, rule: RuleParams) -> Dict[str, float]:
    ece_raw  = expected_calibration_error(y, p_raw, n_bins=rule.n_bins_ece)
    ece_cal  = expected_calibration_error(y, p_cal, n_bins=rule.n_bins_ece)
    brier_raw = brier_score_loss(y, p_raw)
    brier_cal = brier_score_loss(y, p_cal)
    resol = resolution_stats(p_raw, p_cal)
    return {
        "ece_raw": ece_raw,
        "ece_cal": ece_cal,
        "ece_gain": ece_raw - ece_cal,
        "brier_raw": brier_raw,
        "brier_cal": brier_cal,
        "brier_gain": brier_raw - brier_cal,
        "std_ratio": resol["std_ratio"],
        "unique_ratio": resol["unique_ratio"],
    }

def passes_rule(m: Dict[str, float], rule: RuleParams) -> bool:
    return (
        (m["ece_gain"]   >= rule.min_ece_gain) and
        (m["brier_gain"] >= rule.min_brier_gain) and
        (m["std_ratio"]  >= rule.min_std_ratio) and
        (m["unique_ratio"] >= rule.min_unique_ratio)
    )

# ------------------------
# OOS I/O + optional EV
# ------------------------
def read_oos_preds(path: Path, proba_col: str = "y_proba", y_col: str = "y_true",
                   ts_col: str = "entry_ts", sym_col: str = "symbol",
                   dedup: str = "mean") -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Make sure required columns exist
    if proba_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"OOS file must contain '{proba_col}' and '{y_col}' columns.")
    # Optional: keep single row per (ts,symbol)
    if ts_col in df.columns and sym_col in df.columns and dedup:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        if dedup == "mean":
            df = (df
                  .sort_values([ts_col, sym_col])
                  .groupby([ts_col, sym_col], as_index=False)
                  .agg({proba_col:"mean", y_col:"first"}))
        elif dedup == "first":
            df = (df
                  .sort_values([ts_col, sym_col])
                  .drop_duplicates([ts_col, sym_col], keep="first"))
        else:
            raise ValueError("dedup must be 'mean', 'first', or ''")
    return df

def merge_trades(trades_csv: Path, preds: pd.DataFrame,
                 pred_ts: str, pred_sym: str, trades_ts: str, trades_sym: str) -> pd.DataFrame:
    tr = pd.read_csv(trades_csv)
    tr[trades_ts] = pd.to_datetime(tr[trades_ts], utc=True, errors="coerce")
    m = pd.merge(
        preds[[pred_ts, pred_sym, "y_proba_cal"]].rename(columns={pred_ts:"entry_ts", pred_sym:"symbol"}),
        tr,
        on=["entry_ts","symbol"],
        how="inner",
        validate="one_to_one"
    )
    return m

def ev_curve_on_trades(df: pd.DataFrame, thresholds=None, min_trades: int = 150) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 0.91, 0.05), 2)
    out = []
    for t in thresholds:
        pick = df[df["y_proba_cal"] >= t]
        n = len(pick)
        if n < min_trades:
            out.append({"threshold": float(t), "n": int(n), "ev_R": np.nan})
            continue
        ev = float(pick["pnl_R"].mean()) if "pnl_R" in pick.columns else np.nan
        out.append({"threshold": float(t), "n": int(n), "ev_R": ev})
    return pd.DataFrame(out)

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", required=True, help="Path to oos_predictions.parquet")
    ap.add_argument("--out", required=True, help="Output directory")

    # columns
    ap.add_argument("--proba-col", default="y_proba")
    ap.add_argument("--y-col", default="y_true")
    ap.add_argument("--pred-ts", default="entry_ts")
    ap.add_argument("--pred-sym", default="symbol")
    ap.add_argument("--dedup", default="mean", choices=["", "mean", "first"])

    # rule thresholds
    ap.add_argument("--min-ece-gain", type=float, default=0.005)
    ap.add_argument("--min-brier-gain", type=float, default=0.002)
    ap.add_argument("--min-std-ratio", type=float, default=0.50)
    ap.add_argument("--min-unique-ratio", type=float, default=0.50)
    ap.add_argument("--ece-bins", type=int, default=15)

    # optional EV computation & p*
    ap.add_argument("--trades", default=None, help="results/trades.csv to recompute EV on calibrated probs")
    ap.add_argument("--trades-ts", default="entry_ts")
    ap.add_argument("--trades-sym", default="symbol")
    ap.add_argument("--min-trades", type=int, default=150)

    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load OOS predictions
    oos = read_oos_preds(
        Path(args.oos),
        proba_col=args.proba_col, y_col=args.y_col,
        ts_col=args.pred_ts, sym_col=args.pred_sym, dedup=args.dedup
    )
    y = oos[args.y_col].astype(int).values
    p_raw = oos[args.proba_col].astype(float).values

    # Decision rule parameters
    rule = RuleParams(
        min_ece_gain=args.min_ece_gain,
        min_brier_gain=args.min_brier_gain,
        min_std_ratio=args.min_std_ratio,
        min_unique_ratio=args.min_unique_ratio,
        n_bins_ece=args.ece_bins,
    )

    # Baseline metrics
    base = evaluate_calibration(y, p_raw, p_raw, rule)

    # Candidates in preference order: Platt -> Binned -> Isotonic
    cands = [
        PlattCalibrator(),
        BinnedCalibrator(n_bins=12, min_bin=40, enforce_monotone=True),
        IsotonicCalibrator(),
    ]

    picked = IdentityCalibrator()
    picked_metrics = base
    results = {"raw": {"kind": "raw", **base}}

    for cand in cands:
        try:
            model = cand.fit(p_raw, y)
            p_cal = model.predict_proba(p_raw)
            met = evaluate_calibration(y, p_raw, p_cal, rule)
            results[cand.kind] = {"kind": cand.kind, **met}
            if passes_rule(met, rule):
                picked = model
                picked_metrics = met
                break  # pick the simplest that passes
        except Exception as e:
            results[cand.kind] = {"kind": cand.kind, "error": str(e)}

    # Apply chosen calibrator to OOS preds and save (robust to 1D/2D outputs)
    _tmp = picked.predict_proba(p_raw)
    p_cal = _tmp[:, 1] if (isinstance(_tmp, np.ndarray) and _tmp.ndim == 2) else np.asarray(_tmp).ravel()

    oos["y_proba_cal"] = p_cal
    oos_path = outdir / "oos_predictions_calibrated.parquet"
    oos.to_parquet(oos_path, index=False)

    # Export calibrator (skip file if RAW; live should use p_raw)
    with open(outdir / "calibration_kind.txt", "w") as f:
        f.write(getattr(picked, "kind", type(picked).__name__) + "\n")
    if getattr(picked, "kind", "") != "raw":
        joblib.dump(picked, outdir / "calibrator.joblib")
    else:
        # remove any stale calibrator so live doesn’t apply a flat map
        (outdir / "calibrator.joblib").unlink(missing_ok=True)

    # Report
    report = {
        "picked": getattr(picked, "kind", type(picked).__name__),
        "rule": vars(rule),
        "metrics": results,
    }
    (outdir / "calibration_report.json").write_text(json.dumps(report, indent=2))
    print("[calibration] picked:", report["picked"])
    picked_key = report["picked"] if report["picked"] in results else "raw"
    print(json.dumps(results[picked_key], indent=2))

    # Optional EV & p*
    if args.trades:
        try:
            merged = merge_trades(
                Path(args.trades),
                oos.rename(columns={args.pred_ts: "entry_ts", args.pred_sym: "symbol"}),
                pred_ts="entry_ts", pred_sym="symbol",
                trades_ts=args.trades_ts, trades_sym=args.trades_sym,
            )
            ev = ev_curve_on_trades(merged, min_trades=args.min_trades)
            ev_path = outdir / "ev_curve_calibrated.csv"
            ev.to_csv(ev_path, index=False)
            # choose best threshold with enough trades
            ev_valid = ev.dropna().sort_values("ev_R", ascending=False)
            if not ev_valid.empty:
                pstar = float(ev_valid.iloc[0]["threshold"])
                (outdir / "pstar.txt").write_text(f"{pstar:.2f}\n")
                print(f"[EV] p* = {pstar:.2f} (EV={float(ev_valid.iloc[0]['ev_R']):.4f}, n={int(ev_valid.iloc[0]['n'])})")
            else:
                print("[EV] no threshold met min-trades; p* not written.")
        except Exception as e:
            print(f"[EV] skipped due to error: {e}")

if __name__ == "__main__":
    main()
