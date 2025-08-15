# research/reports/winprob_eval.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..modeling.metrics import calibration_table, ece, auc_brier

def save_reliability_plot(y, p, out_png: str):
    tab = calibration_table(y, p, bins=10)
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(tab["p_mean"], tab["y_rate"], marker="o")
    plt.xlabel("Predicted prob (bin mean)")
    plt.ylabel("Empirical win-rate")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def write_reports(y, p, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tab = calibration_table(y, p, bins=10)
    tab.to_csv(os.path.join(out_dir, "winprob_deciles.csv"), index=False)
    # aggregation by coarse buckets (0-0.2, 0.2-0.4, ...)
    df = pd.DataFrame({"y":y, "p":p})
    df["bucket20"] = (df["p"] * 5).clip(0, 4).astype(int)  # 5 buckets
    g = df.groupby("bucket20").agg(count=("y","size"), p_mean=("p","mean"), y_rate=("y","mean"))
    g.to_csv(os.path.join(out_dir, "winprob_buckets.csv"))
    auc, br = auc_brier(y, p)
    with open(os.path.join(out_dir, "winprob_summary.txt"), "w") as f:
        f.write(f"AUC={auc:.4f}  Brier={br:.5f}  ECE={ece(y,p):.5f}\n")
    save_reliability_plot(y, p, os.path.join(out_dir, "winprob_calibration.png"))
