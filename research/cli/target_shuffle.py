# research/cli/target_shuffle.py
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.utils import resample
from research.data.loader import load_dataset
from research.modeling.splits import PurgedKFold
from research.cli.train_eval import _make_pipeline, _drop_all_nan_features

def _oof_preds(df: pd.DataFrame, features: list[str], y: np.ndarray, embargo_bars: int, random_state: int = 0):
    X = df[features].copy()
    cv = PurgedKFold(n_splits=5, embargo=embargo_bars, random_state=random_state)
    oof = np.zeros(len(df), dtype=float)
    for tr, te in cv.split(df):
        pipe = _make_pipeline(numeric_cols=features, hour_col="hour_of_day_at_entry")
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return oof

def _bootstrap_ci(y, p, metric, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        vals.append(metric(y[idx], p[idx]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="research/data/dataset.parquet")
    ap.add_argument("--embargo_bars", type=int, default=48)
    ap.add_argument("--n_reps", type=int, default=200)
    ap.add_argument("--out_csv", default="research/results/target_shuffle.csv")
    ap.add_argument("--summary_out", default="research/results/target_shuffle_summary.json")
    args = ap.parse_args()

    df = load_dataset(args.dataset)
    label = df["win"].astype(int).to_numpy()

    base_feats = [
        "rsi_at_entry","adx_at_entry","atr_pct_at_entry",
        "price_boom_pct_at_entry","price_slowdown_pct_at_entry",
        "vwap_z_at_entry","ema_spread_pct_at_entry","is_ema_crossed_down_at_entry",
        "day_of_week_at_entry","hour_of_day_at_entry","eth_macdhist_at_entry",
        "vwap_stack_frac_at_entry","vwap_stack_expansion_pct_at_entry","vwap_stack_slope_pph_at_entry",
    ]
    feats = _drop_all_nan_features(df, base_feats)

    # Real OOF
    pred_real = _oof_preds(df, feats, label, args.embargo_bars, random_state=1)
    auc_real = float(roc_auc_score(label, pred_real))
    brier_real = float(brier_score_loss(label, pred_real))

    # Bootstrap CIs
    auc_ci = _bootstrap_ci(label, pred_real, roc_auc_score, n_boot=2000, seed=7)
    brier_ci = _bootstrap_ci(label, pred_real, brier_score_loss, n_boot=2000, seed=7)

    # Null by y-shuffle (y-randomization / target shuffling)
    rng = np.random.RandomState(42)
    rows = [{"trial": 0, "type": "real", "auc": auc_real, "brier": brier_real}]
    for i in range(1, args.n_reps + 1):
        y_perm = rng.permutation(label)
        pred_perm = _oof_preds(df, feats, y_perm, args.embargo_bars, random_state=1+i)
        rows.append({
            "trial": i, "type": "shuffled",
            "auc": float(roc_auc_score(y_perm, pred_perm)),
            "brier": float(brier_score_loss(y_perm, pred_perm))
        })

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    # p-value (conservative): fraction of shuffled AUC >= real AUC
    pval = float(np.mean(out.loc[out["type"]=="shuffled", "auc"] >= auc_real))
    summary = {
        "auc_real": auc_real,
        "brier_real": brier_real,
        "auc_ci_95": {"lo": auc_ci[0], "hi": auc_ci[1]},
        "brier_ci_95": {"lo": brier_ci[0], "hi": brier_ci[1]},
        "n_shuffles": args.n_reps,
        "p_value_auc": pval,
        "features_used": feats,
        "embargo_bars": args.embargo_bars,
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Real AUC={auc_real:.4f}  (95% CI {auc_ci[0]:.4f}-{auc_ci[1]:.4f}); p(shuffled >= real)={pval:.3f}")
    print(f"Wrote shuffle table → {args.out_csv}")
    print(f"Wrote summary → {args.summary_out}")

if __name__ == "__main__":
    main()
