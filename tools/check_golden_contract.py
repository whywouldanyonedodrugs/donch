from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_manifest_features(manifest_path: Path) -> list[str]:
    m = json.loads(manifest_path.read_text())
    feats = (m or {}).get("features") or {}
    numeric = list(feats.get("numeric_cols") or [])
    cat = list(feats.get("cat_cols") or [])
    return numeric + cat


def main() -> int:
    repo_root = Path(".").resolve()

    meta_dir = repo_root / "results" / "meta_export"
    manifest_path = meta_dir / "feature_manifest.json"
    golden_path = repo_root / "golden_features.parquet"

    if not manifest_path.exists():
        raise SystemExit(f"Missing: {manifest_path}")
    if not golden_path.exists():
        raise SystemExit(f"Missing: {golden_path}")

    req = load_manifest_features(manifest_path)
    if len(req) != 73:
        print(f"[WARN] Manifest features count is {len(req)} (expected 73 from your exporter)")

    # Only load needed columns
    cols = ["symbol", "timestamp"] + req
    df = pd.read_parquet(golden_path, columns=cols)

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    missing = sorted(set(req) - set(df.columns))
    extra = sorted(set(df.columns) - set(cols))

    print(f"manifest_features={len(req)} golden_rows={len(df)}")
    print(f"missing_in_golden={len(missing)}")
    if missing:
        print("  sample_missing:", missing[:25])

    # "extra" here only refers to extra among the selected columns; should be 0
    print(f"unexpected_extra_selected_cols={len(extra)}")
    if extra:
        print("  sample_extra:", extra[:25])

    # Quick null diagnostics
    null_counts = df[req].isna().sum().sort_values(ascending=False)
    top_null = null_counts[null_counts > 0].head(10)
    print(f"features_with_any_nulls={int((null_counts > 0).sum())}")
    if not top_null.empty:
        print("top_null_features:")
        for k, v in top_null.items():
            print(f"  {k}: {int(v)}")

    # Timestamp coverage
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    print(f"ts_range_utc=[{ts_min.isoformat()} .. {ts_max.isoformat()}]")

    # Return non-zero if contract broken
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
