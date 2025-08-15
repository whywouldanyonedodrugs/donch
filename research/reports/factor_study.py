# research/reports/factor_study.py
import os
import numpy as np
import pandas as pd

def univariate_bins(df: pd.DataFrame, feature: str, y_col: str = "y", q: int = 5) -> pd.DataFrame:
    tmp = df[[feature, y_col]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[feature].rank(method="first"), q=q, labels=False)
    g = tmp.groupby("bin").agg(count=(y_col,"size"), win_rate=(y_col,"mean"), feat_mean=(feature,"mean"))
    g["lift"] = g["win_rate"] / (tmp[y_col].mean() + 1e-9)
    return g.reset_index()
