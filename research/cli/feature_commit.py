# research/cli/feature_commit.py
import argparse, pathlib, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation_csv", default="research/results/ablation.csv")
    ap.add_argument("--out_txt", default="research/config/disable_features.txt")
    ap.add_argument("--dBrier_max", type=float, default=0.0,
                    help="Drop features whose removal lowers Brier (dBrier < 0 → harmful).")
    ap.add_argument("--dAUC_max_increase", type=float, default=0.005,
                    help="Optionally require AUC not to worsen much when dropping the feature.")
    args = ap.parse_args()

    df = pd.read_csv(args.ablation_csv)
    df = df[df["feature"] != "__BASELINE__"].copy()

    # Keep features whose drop makes Brier *worse* (dBrier > 0): helpful → do NOT disable
    # We will DISABLE features with dBrier < args.dBrier_max and dAUC >= -args.dAUC_max_increase
    mask_disable = (df["dBrier"] < args.dBrier_max) & (df["dAUC"] >= -args.dAUC_max_increase)
    to_disable = df.loc[mask_disable, "feature"].sort_values().tolist()

    path = pathlib.Path(args.out_txt)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Auto-generated from ablation; edit manually as needed\n")
        for feat in to_disable:
            f.write(f"{feat}\n")

    print(f"Wrote {len(to_disable)} features to disable → {args.out_txt}")
    if to_disable:
        print("Disabled examples:", ", ".join(to_disable[:10]))

if __name__ == "__main__":
    main()
