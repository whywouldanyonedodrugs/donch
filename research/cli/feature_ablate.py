# research/cli/feature_ablate.py
import argparse
import sys
import pathlib
from typing import List
import numpy as np
import pandas as pd

# --- import guards so "python -m research.cli.feature_ablate" always finds the pkg ---
THIS = pathlib.Path(__file__).resolve()
PKG_ROOT = THIS.parents[1]   # ./research
SRC_ROOT = THIS.parents[2]   # ./src
for p in (str(SRC_ROOT), str(PKG_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from research.modeling.splits import PurgedKFold
from research.modeling.pipeline import build_pipeline
from research.modeling.calibration import fit_with_best_calibration
from research.modeling.metrics import ece
from sklearn.metrics import roc_auc_score, brier_score_loss


def _oof_preds(df: pd.DataFrame, feats: List[str], embargo: int) -> np.ndarray:
    """
    Compute OOF predictions with a pipeline that is rebuilt on the *subset* of features.
    This avoids ColumnTransformer 'column not in dataframe' errors during ablation.
    """
    # Build a minimal frame the pipeline can inspect
    cols_needed = list(feats)
    # keep order_idx only for the splitter
    if "order_idx" in df.columns and "order_idx" not in cols_needed:
        cols_needed.append("order_idx")
    if "y" in df.columns and "y" not in cols_needed:
        cols_needed.append("y")

    df_small = df.loc[:, cols_needed].copy()

    # Build the pipeline on the subset (so the ColumnTransformer expects only these)
    pipe, cont, cat, cyc = build_pipeline(df_small)
    # features that the *subset* pipeline will actually use (intersection with feats)
    used_feats = [c for c in (cont + cat + cyc) if c in feats]

    if not used_feats:
        # If nothing remains, return constant preds = class prior
        prior = float(df["y"].mean())
        return np.full(len(df), prior, dtype=float)

    X = df_small[used_feats]
    y = df["y"].astype(int).values
    order = df["order_idx"].values if "order_idx" in df.columns else np.arange(len(df))

    cv = list(PurgedKFold(n_splits=5, embargo=embargo).split(X, y, order=order))
    oof = np.zeros(len(df), dtype=float)

    for tr, te in cv:
        y_tr = y[tr]
        # guard: if a fold becomes single-class, fallback to prior for that fold
        if np.unique(y_tr).size < 2:
            oof[te] = float(np.mean(y_tr))
            continue
        model, _ = fit_with_best_calibration(pipe, X.iloc[tr], y_tr, cv=3)
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]

    return oof


def _scores(y: np.ndarray, p: np.ndarray):
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    br  = brier_score_loss(y, p)
    ec  = ece(y, p)
    return auc, br, ec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/data/dataset.parquet")
    ap.add_argument("--embargo_bars", type=int, default=48)
    ap.add_argument("--out_csv", default="research/results/ablation.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.data).dropna(subset=["y", "opened_at"]).reset_index(drop=True)

    # Build once on full df to get the base feature set the pipeline *would* use
    _, cont, cat, cyc = build_pipeline(df)
    base_features = cont + cat + cyc
    if not base_features:
        raise SystemExit("No usable features found in dataset for the current pipeline.")

    y = df["y"].astype(int).values

    # Baseline OOF with full feature set
    base_oof = _oof_preds(df, base_features, args.embargo_bars)
    base_auc, base_brier, base_ece = _scores(y, base_oof)

    rows = [{
        "feature": "__BASELINE__",
        "AUC": base_auc,
        "Brier": base_brier,
        "ECE": base_ece,
        "dAUC": 0.0,
        "dBrier": 0.0,
        "dECE": 0.0,
        "coverage": 1.0,
    }]

    # Drop-one ablation
    for f in base_features:
        fset = [x for x in base_features if x != f]
        oof = _oof_preds(df, fset, args.embargo_bars)
        auc, brier, ec = _scores(y, oof)
        rows.append({
            "feature": f,
            "AUC": auc, "Brier": brier, "ECE": ec,
            "dAUC": auc - base_auc,
            "dBrier": brier - base_brier,
            "dECE": ec - base_ece,
            "coverage": float(len(df)) / float(len(df)),
        })

    out = pd.DataFrame(rows)
    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved ablation table → {args.out_csv}")


if __name__ == "__main__":
    main()
