# live/artifact_bundle.py
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib

LOG = logging.getLogger("bundle")


class BundleError(RuntimeError):
    """Fatal artifact-bundle error: missing files, hash mismatch, invalid format."""


class SchemaError(ValueError):
    """Strict schema mismatch for raw features."""


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Return SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def bundle_id_from_hashes(file_hashes: Dict[str, str]) -> str:
    """Deterministic bundle id derived from per-file hashes."""
    h = hashlib.sha256()
    for name in sorted(file_hashes.keys()):
        h.update(name.encode("utf-8"))
        h.update(b":")
        h.update(file_hashes[name].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise BundleError(f"Failed to parse JSON: {path} ({e})") from e


def _resolve_required(meta_dir: Path, names: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    missing = []
    for n in names:
        p = meta_dir / n
        if not p.exists():
            missing.append(n)
        else:
            out[n] = p
    if missing:
        raise BundleError(f"Missing required bundle files in {meta_dir}: {missing}")
    return out


@dataclass(frozen=True)
class ArtifactBundle:
    """Immutable meta artifact bundle (strict-parity live)."""

    meta_dir: Path
    file_hashes: Dict[str, str]
    bundle_id: str

    # Loaded artifacts
    feature_manifest: Any
    feature_names: List[str]
    ohe: Any
    model: Any
    calibrator: Optional[Any]
    pstar: Optional[float]
    model_kind: str  # "lgbm_sklearn" | "lgb_booster"


def load_bundle(
    meta_dir: str | Path,
    *,
    required_extra_files: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> ArtifactBundle:
    """
    Load and validate the exported meta-model artifact bundle.

    Required core files (strict-parity):
      - feature_manifest.json
      - feature_names.json
      - ohe.joblib
      - model.txt  (preferred) OR donch_meta_lgbm.joblib (legacy)
    Optional:
      - calibrator.json OR calibrator.joblib
      - pstar.txt

    Extra files can be required via required_extra_files.
    """
    meta_dir_p = Path(meta_dir).expanduser().resolve()
    if not meta_dir_p.exists():
        raise BundleError(f"META_EXPORT_DIR does not exist: {meta_dir_p}")

    required = ["feature_manifest.json", "feature_names.json", "ohe.joblib"]
    required_paths = _resolve_required(meta_dir_p, required)

    # Model: prefer model.txt (LightGBM booster); allow legacy joblib wrapper.
    model_txt = meta_dir_p / "model.txt"
    model_joblib = meta_dir_p / "donch_meta_lgbm.joblib"

    model_kind: Optional[str] = None
    model_path: Optional[Path] = None
    if model_txt.exists():
        model_kind, model_path = "lgb_booster", model_txt
    elif model_joblib.exists():
        model_kind, model_path = "lgbm_sklearn", model_joblib
        if strict:
            LOG.warning("Bundle using legacy model joblib (%s). Prefer exporting model.txt for strict parity.", model_joblib)
    else:
        raise BundleError(
            f"Missing model artifact in {meta_dir_p}: expected model.txt (preferred) or donch_meta_lgbm.joblib"
        )

    # Optional calibrator
    calibrator = None
    cal_json = meta_dir_p / "calibrator.json"
    cal_joblib = meta_dir_p / "calibrator.joblib"
    cal_path: Optional[Path] = None
    if cal_json.exists():
        cal_path = cal_json
        calibrator = _read_json(cal_json)
    elif cal_joblib.exists():
        cal_path = cal_joblib
        try:
            calibrator = joblib.load(cal_joblib)
        except Exception as e:
            raise BundleError(f"Failed to load calibrator.joblib: {e}") from e

    # Optional p*
    pstar = None
    pstar_path = meta_dir_p / "pstar.txt"
    if pstar_path.exists():
        try:
            pstar = float(pstar_path.read_text(encoding="utf-8").strip())
        except Exception:
            pstar = None

    # Extra required files (regime/sizing artifacts, etc.)
    extra_paths: Dict[str, Path] = {}
    if required_extra_files:
        extra = list(required_extra_files)
        extra_paths = _resolve_required(meta_dir_p, extra)

    # Load JSON manifest + feature names
    feature_manifest = _read_json(required_paths["feature_manifest.json"])
    feature_names_raw = _read_json(required_paths["feature_names.json"])
    if not isinstance(feature_names_raw, list) or not all(isinstance(x, str) for x in feature_names_raw):
        raise BundleError("feature_names.json must be a JSON list of strings.")
    feature_names = list(feature_names_raw)

    # Load OHE
    try:
        ohe = joblib.load(required_paths["ohe.joblib"])
    except Exception as e:
        raise BundleError(f"Failed to load ohe.joblib: {e}") from e

    # Load model
    if model_kind == "lgb_booster":
        try:
            import lightgbm as lgb  # type: ignore
        except Exception as e:
            raise BundleError("lightgbm is required to load model.txt but is not installed.") from e
        try:
            model = lgb.Booster(model_file=str(model_path))
        except Exception as e:
            raise BundleError(f"Failed to load LightGBM Booster from {model_path}: {e}") from e
    else:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            raise BundleError(f"Failed to load joblib model from {model_path}: {e}") from e

    # Hash all included artifacts
    file_hashes: Dict[str, str] = {}
    for name, p in {**required_paths, **extra_paths}.items():
        file_hashes[name] = sha256_file(p)

    file_hashes[model_path.name] = sha256_file(model_path)
    if cal_path is not None:
        file_hashes[cal_path.name] = sha256_file(cal_path)
    if pstar_path.exists():
        file_hashes[pstar_path.name] = sha256_file(pstar_path)

    bid = bundle_id_from_hashes(file_hashes)

    return ArtifactBundle(
        meta_dir=meta_dir_p,
        file_hashes=file_hashes,
        bundle_id=bid,
        feature_manifest=feature_manifest,
        feature_names=feature_names,
        ohe=ohe,
        model=model,
        calibrator=calibrator,
        pstar=pstar,
        model_kind=model_kind,
    )
