# research/modeling/metrics.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import roc_auc_score, brier_score_loss

def calibration_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": p})
    df["bucket"] = pd.qcut(df["p"].rank(method="first") / len(df), q=bins, labels=False)
    g = df.groupby("bucket")
    out = pd.DataFrame({
        "bucket": g.size().index,
        "count": g.size().values,
        "p_mean": g["p"].mean().values,
        "y_rate": g["y"].mean().values,
        "lift": g["y"].mean().values / (df["y"].mean() + 1e-9),
    })
    return out

def ece(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    tab = calibration_table(y_true, p, bins=bins)
    w = tab["count"] / tab["count"].sum()
    return float(np.sum(w.values * np.abs(tab["p_mean"].values - tab["y_rate"].values)))

def auc_brier(y_true: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    try:
        auc = roc_auc_score(y_true, p)
    except ValueError:
        auc = np.nan
    brier = brier_score_loss(y_true, p)
    return float(auc), float(brier)
