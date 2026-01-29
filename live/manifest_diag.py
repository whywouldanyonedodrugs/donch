from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .artifact_bundle import _extract_feature_names_from_manifest


_DERIV_RE = re.compile(
    r"(^|_)(oi|open[_]?interest|funding|funding[_]?rate|basis|premium|carry|perp)(_|$)",
    re.IGNORECASE,
)
_CROSS_ASSET_RE = re.compile(r"(^|_)(btc|eth)(_|$)", re.IGNORECASE)


def is_derivatives_feature(name: str) -> bool:
    return bool(_DERIV_RE.search(str(name)))


def is_cross_asset_feature(name: str) -> bool:
    return bool(_CROSS_ASSET_RE.search(str(name)))


def _boundary_pattern(feature_name: str) -> re.Pattern:
    """
    Match the feature name as a token, not as part of a longer identifier.
    Works for typical usages where feature keys appear as strings in dicts/lists.
    """
    esc = re.escape(feature_name)
    return re.compile(rf"(?<![A-Za-z0-9_]){esc}(?![A-Za-z0-9_])")


def scan_python_files_for_feature_refs(
    repo_root: Path,
    feature_names: Sequence[str],
    *,
    include_dirs: Sequence[str] = ("live", "strategies"),
    exclude_dir_names: Sequence[str] = (".venv", "venv", "__pycache__", "results", ".git"),
) -> Dict[str, int]:
    """
    Very conservative static scan: counts token-boundary occurrences of each feature name
    across .py files under include_dirs. This does NOT prove runtime coverage, but it is a
    safe, reproducible first diagnostic to identify likely-missing/stubbed keys.
    """
    repo_root = Path(repo_root).resolve()

    include_roots: List[Path] = []
    for d in include_dirs:
        p = (repo_root / d).resolve()
        if p.exists() and p.is_dir():
            include_roots.append(p)

    patterns: Dict[str, re.Pattern] = {f: _boundary_pattern(f) for f in feature_names}

    counts: Dict[str, int] = {f: 0 for f in feature_names}

    def _should_skip(path: Path) -> bool:
        parts = set(path.parts)
        return any(x in parts for x in exclude_dir_names)

    for root in include_roots:
        for py in root.rglob("*.py"):
            if _should_skip(py):
                continue
            try:
                txt = py.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for f, pat in patterns.items():
                # count token matches
                m = pat.findall(txt)
                if m:
                    counts[f] += len(m)

    return counts


@dataclass(frozen=True)
class ManifestCoverageReport:
    meta_dir: Path
    n_features: int
    derivatives: List[str]
    cross_asset: List[str]
    ref_counts: Dict[str, int]
    unreferenced: List[str]
    unreferenced_derivatives: List[str]
    unreferenced_cross_asset: List[str]


def build_manifest_coverage_report(
    meta_dir: Path,
    repo_root: Path,
    *,
    include_dirs: Sequence[str] = ("live", "strategies"),
) -> ManifestCoverageReport:
    meta_dir = Path(meta_dir).resolve()
    repo_root = Path(repo_root).resolve()

    manifest_path = meta_dir / "feature_manifest.json"
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names = _extract_feature_names_from_manifest(obj)

    deriv = [f for f in feature_names if is_derivatives_feature(f)]
    cross = [f for f in feature_names if is_cross_asset_feature(f)]

    counts = scan_python_files_for_feature_refs(
        repo_root=repo_root,
        feature_names=feature_names,
        include_dirs=include_dirs,
    )

    unref = [f for f in feature_names if counts.get(f, 0) == 0]
    unref_deriv = [f for f in deriv if counts.get(f, 0) == 0]
    unref_cross = [f for f in cross if counts.get(f, 0) == 0]

    return ManifestCoverageReport(
        meta_dir=meta_dir,
        n_features=len(feature_names),
        derivatives=deriv,
        cross_asset=cross,
        ref_counts=counts,
        unreferenced=unref,
        unreferenced_derivatives=unref_deriv,
        unreferenced_cross_asset=unref_cross,
    )
