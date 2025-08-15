# research/cli/train_eval.py
import argparse, json, os, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

from research.data.loader import load_dataset
from research.modeling.splits import PurgedKFold
from research.modeling.calibration import fit_with_best_calibration

# --- small helpers ---
def _hour_cyc(x: pd.Series) -> pd.DataFrame:
    h = x.astype(float).to_numpy().reshape(-1)
    return pd.DataFrame({"hour_sin": np.sin(2*np.pi*h/24.0), "hour_cos": np.cos(2*np.pi*h/24.0)})

def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    # Expected Calibration Error (Guo et al., 2017)
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not np.any(m): 
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)

def _drop_all_nan_features(df: pd.DataFrame, cols: list[str]) -> list[str]:
    keep = []
    dropped = []
    for c in cols:
        if c in df.columns and not df[c].isna().all():
            keep.append(c)
        else:
            dropped.append(c)
    if dropped:
        print(f"[WARN] Dropping all-NaN (or missing) features: {dropped}")
    return keep

@dataclass
class ModelBundle:
    feature_names: list[str]
    calibrator: CalibratedClassifierCV  # sklearn object (pickle-safe)

def _make_pipeline(numeric_cols: list[str], hour_col: str | None) -> Pipeline:
    numeric_cols = [c for c in numeric_cols if c != hour_col]
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler(with_mean=True, with_std=True)),
    ])
    transformers = [("num", num_pipe, numeric_cols)]
    if hour_col and hour_col in (numeric_cols + [hour_col]):
        transformers.append(("hour", FunctionTransformer(_hour_cyc), [hour_col]))
    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    est = LogisticRegression(
        penalty="l2", solver="liblinear", C=1.0, max_iter=1000, class_weight=None
    )
    return Pipeline([("ct", ct), ("clf", est)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="research/data/dataset.parquet")
    ap.add_argument("--report_dir", default="research/results")
    ap.add_argument("--model_out", default="win_probability_model.pkl")
    ap.add_argument("--embargo_bars", type=int, default=48)
    ap.add_argument("--hour_col", default="hour_of_day_at_entry")
    args = ap.parse_args()

    df = load_dataset(args.dataset)

    # Label and candidate feature set (explicit list keeps us honest)
    label_col = "win"
    base_feats = [
        "rsi_at_entry","adx_at_entry","atr_pct_at_entry",
        "price_boom_pct_at_entry","price_slowdown_pct_at_entry",
        "vwap_z_at_entry","ema_spread_pct_at_entry","is_ema_crossed_down_at_entry",
        "day_of_week_at_entry","hour_of_day_at_entry","eth_macdhist_at_entry",
        # VWAP-stack audit (may be all-NaN in older rows; we will auto-drop if so)
        "vwap_stack_frac_at_entry","vwap_stack_expansion_pct_at_entry","vwap_stack_slope_pph_at_entry",
    ]
    feature_names = _drop_all_nan_features(df, base_feats)

    # data
    y = df[label_col].astype(int).to_numpy()
    X = df[feature_names].copy()

    # CV
    cv = PurgedKFold(n_splits=5, embargo=args.embargo_bars)
    oof_pred = np.zeros(len(X), dtype=float)
    for tr, te in cv.split(df):
        pipe = _make_pipeline(numeric_cols=feature_names, hour_col=args.hour_col)
        # small 3-fold inner CV to pick sigmoid vs isotonic calibration
        calibrated, best = fit_with_best_calibration(pipe, X.iloc[tr], y[tr], cv=3)
        oof_pred[te] = calibrated.predict_proba(X.iloc[te])[:, 1]

    # metrics
    auc = float(roc_auc_score(y, oof_pred))
    brier = float(brier_score_loss(y, oof_pred))
    pos_rate = float(y.mean())
    baseline_brier = float(pos_rate * (1.0 - pos_rate))  # best constant-prob Brier
    ece = _ece(y, oof_pred, n_bins=10)
    dsr = float(np.mean((oof_pred - 0.5) * (y - 0.5) > 0))  # rough directional success rate

    print(f"OOF AUC={auc:.4f}  Brier={brier:.5f}  (baseline={baseline_brier:.5f})  N={len(y)}")
    print(f"ECE={ece:.5f}")
    print(f"DSR (approx)={dsr:.3f}")

    # final calibrator on full data
    final_pipe = _make_pipeline(numeric_cols=feature_names, hour_col=args.hour_col)
    final_cal, best = fit_with_best_calibration(final_pipe, X, y, cv=3)
    print(f"Calibration scores: {best} → using {min(best, key=best.get)}")

    bundle = ModelBundle(feature_names=feature_names, calibrator=final_cal)
    joblib.dump(bundle, args.model_out)
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.report_dir) / "train_eval_summary.json", "w") as f:
        json.dump({
            "auc_oof": auc,
            "brier_oof": brier,
            "brier_baseline": baseline_brier,
            "ece": ece,
            "dsr": dsr,
            "n": int(len(y)),
            "features_used": feature_names
        }, f, indent=2)
    print(f"Saved calibrated model to {args.model_out}")
    print(f"Reports written to {args.report_dir}")

if __name__ == "__main__":
    main()
