from __future__ import annotations

import argparse
from pathlib import Path

from live.manifest_diag import build_manifest_coverage_report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diagnose feature_manifest keys (OI/funding/cross-asset) and static code references."
    )
    ap.add_argument("--meta-dir", required=True, help="Meta export dir (contains feature_manifest.json).")
    ap.add_argument("--repo-root", default=".", help="Repo root to scan for feature references.")
    ap.add_argument(
        "--include-dirs",
        default="live,strategies",
        help="Comma-separated dirs under repo-root to scan (default: live,strategies).",
    )
    ap.add_argument(
        "--fail-on-unreferenced-derivatives",
        action="store_true",
        help="Exit non-zero if any derivative (OI/funding) features are unreferenced in scanned code.",
    )
    ap.add_argument(
        "--fail-on-unreferenced-cross",
        action="store_true",
        help="Exit non-zero if any cross-asset (BTC/ETH) features are unreferenced in scanned code.",
    )
    args = ap.parse_args()

    meta_dir = Path(args.meta_dir)
    repo_root = Path(args.repo_root)
    include_dirs = tuple([x.strip() for x in str(args.include_dirs).split(",") if x.strip()])

    r = build_manifest_coverage_report(meta_dir=meta_dir, repo_root=repo_root, include_dirs=include_dirs)

    print(f"meta_dir: {r.meta_dir}")
    print(f"total manifest features: {r.n_features}")
    print("")

    print(f"derivatives-like features: {len(r.derivatives)}")
    for f in r.derivatives:
        print(f"  {f}  (refs={r.ref_counts.get(f, 0)})")
    print("")

    print(f"cross-asset-like features: {len(r.cross_asset)}")
    for f in r.cross_asset:
        print(f"  {f}  (refs={r.ref_counts.get(f, 0)})")
    print("")

    print(f"unreferenced features in scanned dirs: {len(r.unreferenced)}")
    for f in r.unreferenced[:200]:
        print(f"  {f}")
    if len(r.unreferenced) > 200:
        print(f"  ... ({len(r.unreferenced) - 200} more)")
    print("")

    if r.unreferenced_derivatives:
        print(f"UNREFERENCED DERIVATIVES: {len(r.unreferenced_derivatives)}")
        for f in r.unreferenced_derivatives:
            print(f"  {f}")
        print("")
    if r.unreferenced_cross_asset:
        print(f"UNREFERENCED CROSS-ASSET: {len(r.unreferenced_cross_asset)}")
        for f in r.unreferenced_cross_asset:
            print(f"  {f}")
        print("")

    if args.fail_on_unreferenced_derivatives and r.unreferenced_derivatives:
        return 2
    if args.fail_on_unreferenced_cross and r.unreferenced_cross_asset:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
