from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib


class BundleError(RuntimeError):
    pass


# Regime truth artifacts (Option A)
REGIME_DAILY_TRUTH_FILE = "regime_daily_truth.parquet"
REGIME_MARKOV4H_TRUTH_FILE = "regime_markov4h_truth.parquet"

REGIME_FEATURE_KEYS = {
    "regime_code_1d",
    "vol_prob_low_1d",
    "markov_state_4h",
    "markov_prob_up_4h",
}


@dataclass(frozen=True)
class ArtifactBundle:
    meta_dir: Path
    bundle_id: str
    model_kind: str  # "sklearn_pipeline" or "legacy_lgbm"
    feature_manifest: dict

    # Artifacts
    model: Any
    calibrator: Optional[Any]
    calibrator_path: Optional[Path]
    pstar: Optional[float]
    pstar_scope: Optional[str]  # e.g. "RISK_ON_1"

    # optional extras (used by other live components)
    thresholds: Optional[dict]
    sizing_curve_path: Optional[Path]
    deployment_config: Optional[dict]

    # Regime truth artifact paths (loaded elsewhere; included in bundle_id)
    regime_daily_truth_path: Optional[Path]
    regime_markov4h_truth_path: Optional[Path]

    # verification info
    file_hashes: Dict[str, str]


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def bundle_id_from_hashes(file_hashes: Dict[str, str]) -> str:
    """
    Stable bundle id from (filename, sha256) pairs.
    """
    items = sorted((k, v) for k, v in file_hashes.items())
    h = hashlib.sha256()
    for name, digest in items:
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(digest.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(meta_dir: Path, names: Iterable[str]) -> Optional[Path]:
    for n in names:
        p = meta_dir / n
        if p.exists():
            return p
    return None


def _load_checksums_map(obj: Any) -> Dict[str, str]:
    """
    Accepts common shapes:
      - {"file": "sha", ...}
      - {"files": {"file": "sha", ...}}
      - {"sha256": {"file": "sha", ...}}
    """
    if isinstance(obj, dict):
        if isinstance(obj.get("files"), dict):
            obj = obj["files"]
        elif isinstance(obj.get("sha256"), dict):
            obj = obj["sha256"]

    if not isinstance(obj, dict):
        raise BundleError(f"checksums_sha256.json must be dict-like, got {type(obj)}")

    out: Dict[str, str] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v

    if not out:
        raise BundleError("checksums_sha256.json had no usable {filename: sha256} entries")

    return out


def _validate_checksums(*, meta_dir: Path, computed: Dict[str, str], strict: bool) -> None:
    """
    Validate computed sha256 for the files in 'computed' against checksums_sha256.json.
    If strict=False, only warns via exceptions for obviously bad structure; does not enforce equality.
    """
    chk_path = meta_dir / "checksums_sha256.json"
    if not chk_path.exists():
        raise BundleError(f"Missing checksums file: {chk_path}")

    chk_raw = _read_json(chk_path)
    chk_map = _load_checksums_map(chk_raw)

    if not strict:
        return

    missing = [k for k in computed.keys() if k not in chk_map]
    if missing:
        raise BundleError(f"checksums_sha256.json missing entries for: {missing}")

    mismatched = []
    for name, digest in computed.items():
        exp = chk_map.get(name)
        if exp is None:
            continue
        if str(exp).lower() != str(digest).lower():
            mismatched.append((name, exp, digest))

    if mismatched:
        lines = ["Checksum mismatch:"]
        for name, exp, got in mismatched[:20]:
            lines.append(f"  {name}: expected {exp} got {got}")
        raise BundleError("\n".join(lines))


def load_bundle(meta_dir: str | Path, strict: bool = True, required_extra_files: Optional[Iterable[str]] = None) -> ArtifactBundle:
    meta_dir_p = Path(meta_dir).resolve()
    if not meta_dir_p.exists():
        raise BundleError(f"Meta export dir not found: {meta_dir_p}")

    # Core required artifacts
    required = [
        "feature_manifest.json",
        "feature_names.json",
        "checksums_sha256.json",
    ]

    # Model candidates
    model_path = _first_existing(meta_dir_p, ["model.joblib", "model.pkl", "model.joblib.gz"])
    if model_path is None:
        raise BundleError("No model found (expected model.joblib / model.pkl / model.joblib.gz)")

    # Calibration candidates
    calibrator_path = _first_existing(meta_dir_p, ["calibrator.joblib", "isotonic.joblib", "calibration.json"])
    thresholds_path = _first_existing(meta_dir_p, ["thresholds.json"])
    sizing_curve_path = _first_existing(meta_dir_p, ["sizing_curve.csv"])
    deployment_config_path = _first_existing(meta_dir_p, ["deployment_config.json"])

    required.append(model_path.name)
    if thresholds_path is not None:
        required.append(thresholds_path.name)
    if sizing_curve_path is not None:
        required.append(sizing_curve_path.name)
    if deployment_config_path is not None:
        required.append(deployment_config_path.name)
    if calibrator_path is not None:
        required.append(calibrator_path.name)

    # Verify base required exist
    for name in required:
        p = meta_dir_p / name
        if not p.exists():
            raise BundleError(f"Missing required bundle file: {p}")

    feature_manifest = _read_json(meta_dir_p / "feature_manifest.json")
    manifest_keys = set(feature_manifest.keys()) if isinstance(feature_manifest, dict) else set()

    # If regimes are in the manifest, require truth artifacts (Option A).
    auto_extra: list[str] = []
    if manifest_keys & REGIME_FEATURE_KEYS:
        auto_extra.extend([REGIME_DAILY_TRUTH_FILE, REGIME_MARKOV4H_TRUTH_FILE])

    user_extra = list(required_extra_files) if required_extra_files is not None else []
    for n in auto_extra:
        if n not in user_extra:
            user_extra.append(n)

    # Enforce extras existence
    for n in user_extra:
        p = meta_dir_p / n
        if not p.exists():
            raise BundleError(f"Missing required extra bundle file: {p}")

    # Compute sha256 for required files (excluding checksums file itself)
    to_hash = [n for n in (required + user_extra) if n != "checksums_sha256.json"]
    file_hashes: Dict[str, str] = {name: sha256_file(meta_dir_p / name) for name in to_hash}

    _validate_checksums(meta_dir=meta_dir_p, computed=file_hashes, strict=strict)

    b_id = bundle_id_from_hashes(file_hashes)

    # Load thresholds/p* if present
    thresholds = _read_json(thresholds_path) if thresholds_path is not None else None
    pstar = None
    pstar_scope = None
    if isinstance(thresholds, dict):
        pstar = thresholds.get("pstar") or thresholds.get("p*") or thresholds.get("p_star")
        pstar_scope = thresholds.get("scope")

    deployment_config = _read_json(deployment_config_path) if deployment_config_path is not None else None

    # Load model (sklearn pipeline)
    model = joblib.load(model_path)
    model_kind = "sklearn_pipeline"

    calibrator = None
    if calibrator_path is not None:
        if calibrator_path.suffix.lower() == ".json":
            calibrator = _read_json(calibrator_path)
        else:
            calibrator = joblib.load(calibrator_path)

    regime_daily_truth_path = meta_dir_p / REGIME_DAILY_TRUTH_FILE if (REGIME_DAILY_TRUTH_FILE in user_extra) else None
    regime_markov4h_truth_path = meta_dir_p / REGIME_MARKOV4H_TRUTH_FILE if (REGIME_MARKOV4H_TRUTH_FILE in user_extra) else None

    return ArtifactBundle(
        meta_dir=meta_dir_p,
        bundle_id=b_id,
        model_kind=model_kind,
        feature_manifest=feature_manifest,
        model=model,
        calibrator=calibrator,
        calibrator_path=calibrator_path,
        pstar=pstar,
        pstar_scope=pstar_scope,
        thresholds=thresholds,
        sizing_curve_path=sizing_curve_path,
        deployment_config=deployment_config,
        regime_daily_truth_path=regime_daily_truth_path,
        regime_markov4h_truth_path=regime_markov4h_truth_path,
        file_hashes=file_hashes,
    )
