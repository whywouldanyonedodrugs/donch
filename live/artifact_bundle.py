# live/artifact_bundle.py
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib

LOG = logging.getLogger("bundle")


class BundleError(RuntimeError):
    """Fatal artifact-bundle error: missing files, invalid format, load failure."""


class SchemaError(ValueError):
    """Strict schema mismatch for raw features."""


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


def _first_existing(meta_dir: Path, candidates: List[str]) -> Optional[Path]:
    for n in candidates:
        p = meta_dir / n
        if p.exists():
            return p
    return None


def _require(meta_dir: Path, path: Optional[Path], label: str) -> Path:
    if path is None or (not path.exists()):
        present = sorted([p.name for p in meta_dir.glob("*") if p.is_file()])
        raise BundleError(f"Missing required {label} in {meta_dir}. Present files: {present}")
    return path


def _is_sklearn_pipeline(obj: Any) -> bool:
    # Heuristic: sklearn Pipeline exposes "steps" / "named_steps"
    return hasattr(obj, "steps") or hasattr(obj, "named_steps")


@dataclass(frozen=True)
class ArtifactBundle:
    meta_dir: Path
    file_hashes: Dict[str, str]
    bundle_id: str

    feature_manifest: Any

    # One of:
    # - external_ohe mode: feature_names + ohe + model (txt or joblib)
    # - pipeline_raw mode: model.joblib pipeline handles preprocessing internally
    feature_names: List[str]
    ohe: Optional[Any]

    model: Any
    model_kind: str  # "lgb_booster" | "lgbm_sklearn" | "pipeline_joblib"

    calibrator: Optional[Any]
    pstar: Optional[float]


def load_bundle(
    meta_dir: str | Path,
    *,
    required_extra_files: Optional[Iterable[str]] = None,
    strict: bool = True,
) -> ArtifactBundle:
    """
    Load and validate exported meta-model artifacts.

    Supported export layouts:

    A) External-OHE layout (older):
      - feature_manifest.json
      - feature_names.json
      - ohe.joblib
      - model.txt (preferred) OR donch_meta_lgbm.joblib

    B) Pipeline layout (current in your folder):
      - feature_manifest.json
      - model.joblib (sklearn pipeline that includes preprocessing/OHE)
      - optional calibrators: isotonic.joblib and/or calibration.json

    Calibrator accepted names:
      - calibration.json OR calibrator.json
      - isotonic.joblib OR calibrator.joblib

    strict=True means: hard-fail if required artifacts for the detected layout are missing.
    """
    meta_dir_p = Path(meta_dir).expanduser().resolve()
    if not meta_dir_p.exists():
        raise BundleError(f"META_EXPORT_DIR does not exist: {meta_dir_p}")

    # Always required
    manifest_path = _require(meta_dir_p, _first_existing(meta_dir_p, ["feature_manifest.json"]), "feature_manifest.json")
    feature_manifest = _read_json(manifest_path)

    # Detect model format
    model_txt = _first_existing(meta_dir_p, ["model.txt"])
    model_legacy = _first_existing(meta_dir_p, ["donch_meta_lgbm.joblib"])
    model_joblib = _first_existing(meta_dir_p, ["model.joblib"])  # current export

    model_kind: Optional[str] = None
    model_path: Optional[Path] = None

    if model_txt is not None:
        model_kind, model_path = "lgb_booster", model_txt
    elif model_legacy is not None:
        model_kind, model_path = "lgbm_sklearn", model_legacy
        if strict:
            LOG.warning("Using legacy model joblib (%s). Prefer model.txt or a full pipeline model.joblib.", model_legacy)
    elif model_joblib is not None:
        model_kind, model_path = "pipeline_joblib", model_joblib
    else:
        present = sorted([p.name for p in meta_dir_p.glob("*") if p.is_file()])
        raise BundleError(f"No model artifact found in {meta_dir_p}. Present files: {present}")

    # Calibrator (optional)
    cal_json_path = _first_existing(meta_dir_p, ["calibrator.json", "calibration.json"])
    cal_joblib_path = _first_existing(meta_dir_p, ["calibrator.joblib", "isotonic.joblib"])

    calibrator: Optional[Any] = None
    if cal_joblib_path is not None:
        try:
            calibrator = joblib.load(cal_joblib_path)
        except Exception as e:
            raise BundleError(f"Failed to load calibrator joblib {cal_joblib_path}: {e}") from e
    elif cal_json_path is not None:
        calibrator = _read_json(cal_json_path)

    # p* (optional, best-effort from deployment_config.json or thresholds.json)
    pstar: Optional[float] = None
    dep_cfg_path = _first_existing(meta_dir_p, ["deployment_config.json"])
    thr_path = _first_existing(meta_dir_p, ["thresholds.json"])
    for p in [dep_cfg_path, thr_path]:
        if p and p.exists():
            try:
                obj = _read_json(p)
                if isinstance(obj, dict):
                    for k in ("pstar", "p_star", "p*"):
                        if k in obj:
                            pstar = float(obj[k])
                            break
                if pstar is not None:
                    break
            except Exception:
                pass

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
            model = joblib.load(model_path)  # lgbm_sklearn or pipeline_joblib
        except Exception as e:
            raise BundleError(f"Failed to load joblib model from {model_path}: {e}") from e

    # External-OHE requirements (only for those layouts)
    feature_names: List[str] = []
    ohe = None

    if model_kind in ("lgb_booster", "lgbm_sklearn"):
        fn_path = _first_existing(meta_dir_p, ["feature_names.json"])
        ohe_path = _first_existing(meta_dir_p, ["ohe.joblib"])
        fn_path = _require(meta_dir_p, fn_path, "feature_names.json")
        ohe_path = _require(meta_dir_p, ohe_path, "ohe.joblib")

        fn_obj = _read_json(fn_path)
        if not isinstance(fn_obj, list) or not all(isinstance(x, str) for x in fn_obj):
            raise BundleError("feature_names.json must be a JSON list of strings.")
        feature_names = list(fn_obj)

        try:
            ohe = joblib.load(ohe_path)
        except Exception as e:
            raise BundleError(f"Failed to load ohe.joblib: {e}") from e

    elif model_kind == "pipeline_joblib":
        # Pipeline layout: feature_names/ohe are not required.
        # If the object exposes feature_names_in_, we can use it for diagnostics.
        try:
            fni = getattr(model, "feature_names_in_", None)
            if fni is not None:
                feature_names = [str(x) for x in list(fni)]
        except Exception:
            feature_names = []

        # Strong safety check: pipeline must accept DataFrame with raw columns.
        # If it is not a Pipeline, we still support it if it has feature_names_in_ (above).
        if (not _is_sklearn_pipeline(model)) and (not feature_names):
            if strict:
                raise BundleError(
                    "model.joblib is not a sklearn Pipeline and does not expose feature_names_in_. "
                    "Cannot safely construct inputs. Export a pipeline model.joblib or provide feature_names.json + ohe.joblib."
                )
            LOG.warning(
                "model.joblib not a Pipeline and lacks feature_names_in_. Scoring may be disabled in non-strict mode."
            )

    # Extra required files (regime/sizing artifacts, etc.)
    if required_extra_files:
        missing_extra = []
        for n in required_extra_files:
            if not (meta_dir_p / n).exists():
                missing_extra.append(n)
        if missing_extra:
            raise BundleError(f"Missing required extra files in {meta_dir_p}: {missing_extra}")

    # Build bundle_id from a curated set + whatever is present that matters
    include_candidates = [
        "feature_manifest.json",
        "feature_names.json",
        "ohe.joblib",
        "model.txt",
        "donch_meta_lgbm.joblib",
        "model.joblib",
        "calibrator.json",
        "calibration.json",
        "calibrator.joblib",
        "isotonic.joblib",
        "thresholds.json",
        "sizing_curve.csv",
        "ev_thresholds.csv",
        "regimes_report.json",
        "deployment_config.json",
        "checksums_sha256.json",
        "bundle.tar.gz",
    ]
    file_hashes: Dict[str, str] = {}
    for name in include_candidates:
        p = meta_dir_p / name
        if p.exists():
            file_hashes[name] = sha256_file(p)

    # Also include required extras in hash set
    if required_extra_files:
        for n in required_extra_files:
            p = meta_dir_p / n
            if p.exists():
                file_hashes[n] = sha256_file(p)

    bid = bundle_id_from_hashes(file_hashes)

    return ArtifactBundle(
        meta_dir=meta_dir_p,
        file_hashes=file_hashes,
        bundle_id=bid,
        feature_manifest=feature_manifest,
        feature_names=feature_names,
        ohe=ohe,
        model=model,
        model_kind=model_kind,
        calibrator=calibrator,
        pstar=pstar,
    )
