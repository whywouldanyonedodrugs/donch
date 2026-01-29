from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DAILY_OUT_DEFAULT = "regime_daily_truth.parquet"
MARKOV_OUT_DEFAULT = "regime_markov4h_truth.parquet"


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_golden(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
    else:
        ts = pd.to_datetime(df.index, utc=True)
    df = df.copy()
    df["timestamp"] = ts
    # normalize required columns
    req = ["regime_code_1d", "vol_prob_low_1d", "markov_state_4h", "markov_prob_up_4h"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise SystemExit(f"golden missing required columns: {missing}")
    return df


def _reduce_macro_per_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    golden has many symbols; macro columns are constant per timestamp.
    Reduce to one row per timestamp deterministically.
    """
    use_cols = ["timestamp", "regime_code_1d", "vol_prob_low_1d", "markov_state_4h", "markov_prob_up_4h"]
    df2 = df[use_cols].sort_values(["timestamp"])
    # deterministic: take first row per timestamp
    df2 = df2.groupby("timestamp", as_index=False).first()
    return df2


def _build_daily_truth(macro: pd.DataFrame) -> pd.DataFrame:
    ts = macro["timestamp"]
    day = ts.dt.floor("D")

    daily = macro.copy()
    daily["day"] = day

    # Expect constant within day; take first per day (deterministic).
    daily = daily.sort_values(["timestamp"]).groupby("day", as_index=False).first()

    out = daily[["day", "regime_code_1d", "vol_prob_low_1d"]].rename(columns={"day": "timestamp"}).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.set_index("timestamp").sort_index()

    # dtypes
    out["regime_code_1d"] = pd.to_numeric(out["regime_code_1d"], errors="coerce").astype("Int64")
    out["vol_prob_low_1d"] = pd.to_numeric(out["vol_prob_low_1d"], errors="coerce").astype(float)

    if out.index.has_duplicates:
        raise SystemExit("daily truth index has duplicates")
    return out


def _build_markov_truth(macro: pd.DataFrame) -> pd.DataFrame:
    ts = macro["timestamp"]
    bucket = ts.dt.floor("4h")  # use lowercase to avoid pandas FutureWarning

    m = macro.copy()
    m["bucket"] = bucket
    m = m.sort_values(["timestamp"]).groupby("bucket", as_index=False).first()

    out = m[["bucket", "markov_state_4h", "markov_prob_up_4h"]].rename(columns={"bucket": "timestamp"}).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.set_index("timestamp").sort_index()

    out["markov_state_4h"] = pd.to_numeric(out["markov_state_4h"], errors="coerce").astype("Int64")
    out["markov_prob_up_4h"] = pd.to_numeric(out["markov_prob_up_4h"], errors="coerce").astype(float)

    if out.index.has_duplicates:
        raise SystemExit("markov truth index has duplicates")
    return out


def _update_checksums(checksums_path: Path, updates: Dict[str, str]) -> None:
    obj = json.loads(checksums_path.read_text(encoding="utf-8"))
    # accept common shapes; store back in same shape
    if isinstance(obj, dict) and isinstance(obj.get("files"), dict):
        files = obj["files"]
        container_key = "files"
    elif isinstance(obj, dict) and isinstance(obj.get("sha256"), dict):
        files = obj["sha256"]
        container_key = "sha256"
    elif isinstance(obj, dict):
        files = obj
        container_key = None
    else:
        raise SystemExit("checksums file must be a dict-like JSON")

    if not isinstance(files, dict):
        raise SystemExit("checksums container is not a dict")

    for k, v in updates.items():
        files[k] = v

    if container_key is None:
        out_obj = files
    else:
        obj[container_key] = files
        out_obj = obj

    checksums_path.write_text(json.dumps(out_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", required=True, help="e.g. results/meta_export")
    ap.add_argument("--golden", default=None, help="override golden parquet path")
    ap.add_argument("--out-daily", default=DAILY_OUT_DEFAULT)
    ap.add_argument("--out-markov", default=MARKOV_OUT_DEFAULT)
    ap.add_argument("--update-checksums", action="store_true")
    ap.add_argument("--checksums", default="checksums_sha256.json")
    args = ap.parse_args()

    meta_dir = Path(args.meta_dir).resolve()
    golden_path = Path(args.golden).resolve() if args.golden else (meta_dir / "golden_features.parquet")
    if not golden_path.exists():
        raise SystemExit(f"golden parquet not found: {golden_path}")

    df = _load_golden(golden_path)
    macro = _reduce_macro_per_ts(df)

    daily_truth = _build_daily_truth(macro)
    markov_truth = _build_markov_truth(macro)

    out_daily = (meta_dir / args.out_daily).resolve()
    out_markov = (meta_dir / args.out_markov).resolve()

    daily_truth.to_parquet(out_daily, index=True)
    markov_truth.to_parquet(out_markov, index=True)

    daily_sha = _sha256_file(out_daily)
    markov_sha = _sha256_file(out_markov)

    print(f"WROTE {out_daily} sha256={daily_sha}")
    print(f"WROTE {out_markov} sha256={markov_sha}")

    if args.update_checksums:
        checksums_path = (meta_dir / args.checksums).resolve()
        if not checksums_path.exists():
            raise SystemExit(f"checksums not found: {checksums_path}")
        _update_checksums(
            checksums_path,
            {
                out_daily.name: daily_sha,
                out_markov.name: markov_sha,
            },
        )
        print(f"UPDATED {checksums_path} with new entries")


if __name__ == "__main__":
    main()
