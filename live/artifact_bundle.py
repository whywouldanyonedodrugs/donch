# live/artifact_bundle.py
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import joblib

LOG = logging.getLogger("bundle")


class BundleError(RuntimeError):
    pass

class SchemaError(BundleError):
    """
    Raised when a raw feature row does not match the required manifest schema.
    Kept as a distinct type because winprob_loader imports it explicitly.
    """
    pass


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

    # optional extras (used by other live components)
    thresholds: Optional[dict]
    sizing_curve_path: Optional[Path]
    deployment_config: Optional[dict]

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

    If strict=True:
      - checksums_sha256.json must exist
      - all computed filenames must exist in the checksums map
      - all sha must match
    """
    chk_path = meta_dir / "checksums_sha256.json"
    if not chk_path.exists():
        if strict:
            raise BundleError(f"Missing required checksums_sha256.json in {meta_dir}")
        LOG.warning("checksums_sha256.json missing in %s (non-strict).", meta_dir)
        return

    try:
        expected_map = _load_checksums_map(_read_json(chk_path))
    except Exception as e:
        raise BundleError(f"Failed to read/parse checksums_sha256.json: {e}") from e

    missing = []
    mismatched = []
    for name, got in computed.items():
        exp = expected_map.get(name)
        if exp is None:
            missing.append(name)
        elif str(exp).lower() != str(got).lower():
            mismatched.append((name, exp, got))

    if mismatched:
        msg = "; ".join(
            [f"{n}: expected={e[:12]}.. got={g[:12]}.." for (n, e, g) in mismatched[:10]]
        )
        raise BundleError(f"Bundle checksum mismatch for {len(mismatched)} file(s): {msg}")

    if strict and missing:
        raise BundleError(
            f"checksums_sha256.json missing entries for required files: {missing[:20]}"
            + (" ..." if len(missing) > 20 else "")
        )


def _joblib_load_guarded(path: Path, *, strict_versions: bool) -> Any:
    """
    Load joblib artifact and (optionally) hard-fail on sklearn InconsistentVersionWarning.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj = joblib.load(path)

    bad = [wi for wi in w if getattr(wi.category, "__name__", "") == "InconsistentVersionWarning"]
    if bad:
        msg = "; ".join([str(wi.message) for wi in bad[:3]])
        if strict_versions:
            raise BundleError(f"Inconsistent sklearn version while loading {path.name}: {msg}")
        LOG.warning("Inconsistent sklearn version while loading %s: %s", path.name, msg)

    return obj


def load_bundle(
    meta_dir: Union[str, Path],
    strict: bool = True,
    required_extra_files: Optional[List[str]] = None,
) -> ArtifactBundle:
    """
    Load and validate the deployment bundle.

    Canonical contract (Option A / pipeline bundle):
      - model.joblib
      - feature_manifest.json
      - deployment_config.json
      - checksums_sha256.json
      - thresholds.json
      - sizing_curve.csv
      - calibrator: one of {isotonic.joblib, calibration.json, calibrator.json, calibrator.joblib}

    Legacy contract (only if model.joblib absent):
      - model.txt + ohe.joblib + feature_names.json
      - feature_manifest.json
      - (optional) calibration

    If strict=True:
      - required files must exist
      - checksums must match
      - sklearn InconsistentVersionWarning becomes a hard error
    """
    meta_dir_p = Path(meta_dir).expanduser().resolve()
    if not meta_dir_p.exists() or not meta_dir_p.is_dir():
        raise BundleError(f"Bundle directory does not exist: {meta_dir_p}")

    required_extra_files = required_extra_files or []

    # Decide contract by presence of model.joblib (this matches your exporter outputs)
    model_joblib = meta_dir_p / "model.joblib"
    is_pipeline = model_joblib.exists()

    # Common required
    required: List[str] = ["feature_manifest.json"]

    # Pipeline required (Option A)
    pipeline_required: List[str] = [
        "deployment_config.json",
        "checksums_sha256.json",
        "thresholds.json",
        "sizing_curve.csv",
        "model.joblib",
    ]

    # Legacy required (Option B)
    legacy_required: List[str] = [
        "model.txt",
        "ohe.joblib",
        "feature_names.json",
    ]

    if is_pipeline:
        required += pipeline_required
    else:
        required += legacy_required

    # Add any caller-required extras
    required += list(required_extra_files)

    # Calibrator: accept any of these names (your export uses calibration.json + isotonic.joblib)
    calib_joblib = _first_existing(meta_dir_p, ["isotonic.joblib", "calibrator.joblib"])
    calib_json = _first_existing(meta_dir_p, ["calibration.json", "calibrator.json"])

    if strict and (calib_joblib is None and calib_json is None):
        raise BundleError(
            "Missing calibrator artifact: expected one of "
            "{isotonic.joblib, calibrator.joblib, calibration.json, calibrator.json}"
        )

    # Resolve required paths
    missing = [n for n in required if not (meta_dir_p / n).exists()]
    if missing:
        raise BundleError(f"Missing required bundle files in {meta_dir_p}: {missing}")

    required_paths: Dict[str, Path] = {n: (meta_dir_p / n) for n in required}

    # Compute hashes for required set (+ calibrators if present) for validation and bundle id
    to_hash: Dict[str, Path] = dict(required_paths)
    if calib_joblib is not None:
        to_hash[calib_joblib.name] = calib_joblib
    if calib_json is not None:
        to_hash[calib_json.name] = calib_json

    file_hashes: Dict[str, str] = {}
    for name, path in sorted(to_hash.items()):
        file_hashes[name] = sha256_file(path)

    # Validate checksums against required+calibration set
    _validate_checksums(meta_dir=meta_dir_p, computed=file_hashes, strict=strict)

    bid = bundle_id_from_hashes(file_hashes)

    # Load manifest
    feature_manifest = _read_json(required_paths["feature_manifest.json"])
    deployment_config = _read_json(required_paths["deployment_config.json"]) if is_pipeline else None
    thresholds = _read_json(required_paths["thresholds.json"]) if (meta_dir_p / "thresholds.json").exists() else None
    sizing_curve_path = (meta_dir_p / "sizing_curve.csv") if (meta_dir_p / "sizing_curve.csv").exists() else None

    # Load model/calibrator
    pstar: Optional[float] = None
    calibrator: Optional[Any] = None
    calibrator_path: Optional[Path] = None

    if is_pipeline:
        model = _joblib_load_guarded(model_joblib, strict_versions=strict)
        model_kind = "sklearn_pipeline"
    else:
        # legacy: keep returning raw text paths; live code can handle it if still used
        model = (meta_dir_p / "model.txt").read_text(encoding="utf-8")
        model_kind = "legacy_lgbm"

    # Prefer joblib calibrator if present (isotonic.joblib)
    if calib_joblib is not None:
        calibrator_path = calib_joblib
        calibrator = _joblib_load_guarded(calib_joblib, strict_versions=strict)

    # Load calibration.json if present (even if isotonic exists) to read pstar metadata
    if calib_json is not None:
        try:
            cal_obj = _read_json(calib_json)
            # If no joblib calibrator was loaded, use json calibrator as calibrator object
            if calibrator is None:
                calibrator_path = calib_json
                calibrator = cal_obj
            # Extract pstar if present
            for k in ("pstar", "p_star", "p*", "pStar"):
                if isinstance(cal_obj, dict) and k in cal_obj:
                    try:
                        pstar = float(cal_obj[k])
                    except Exception:
                        pstar = None
                    break
        except Exception as e:
            if strict:
                raise BundleError(f"Failed to load/parse {calib_json.name}: {e}") from e
            LOG.warning("Could not parse %s: %s", calib_json.name, e)

    # Log what we validated (useful in systemd logs)
    LOG.info("Loaded bundle: dir=%s id=%s kind=%s", str(meta_dir_p), bid, model_kind)
    for name in sorted(file_hashes.keys()):
        LOG.info("Bundle sha256 %s=%s", name, file_hashes[name])

    return ArtifactBundle(
        meta_dir=meta_dir_p,
        bundle_id=bid,
        model_kind=model_kind,
        feature_manifest=feature_manifest,
        model=model,
        calibrator=calibrator,
        calibrator_path=calibrator_path,
        pstar=pstar,
        thresholds=thresholds,
        sizing_curve_path=sizing_curve_path,
        deployment_config=deployment_config,
        file_hashes=file_hashes,
    )
