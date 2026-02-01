from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np

class BundleError(RuntimeError):
    pass
class SchemaError(BundleError):
    """Raised when raw feature schema/dtypes do not match the feature manifest."""


# Regime truth artifacts (Option A)
REGIME_DAILY_TRUTH_FILE = "regime_daily_truth.parquet"
REGIME_MARKOV4H_TRUTH_FILE = "regime_markov4h_truth.parquet"

REGIME_FEATURE_KEYS = {
    "regime_code_1d",
    "vol_prob_low_1d",
    "markov_state_4h",
    "markov_prob_up_4h",
}



def _extract_feature_names_from_manifest(feature_manifest: dict) -> list[str]:
    """
    Extract feature names from feature_manifest.json in a schema-robust way.

    Supported shapes:
      - {"features": {"numeric_cols": [...], "cat_cols": [...]} }   (current)
      - {"feature_names": [...]} / {"columns": [...]} / {"feature_cols": [...]}
      - {"features": {feature_name: {...}, ...}}                   (legacy)
      - {"features": [{"name": "..."}, ...]} / {"features": ["a", "b", ...]}
      - {"schema": {...}}                                          (recursive)
    """

    def _dedupe(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _is_str_list(v: object) -> bool:
        return isinstance(v, list) and all(isinstance(x, str) and x for x in v)

    def _from(obj: object) -> list[str]:
        if not isinstance(obj, dict):
            return []

        # Direct lists
        for k in ("feature_names", "columns", "feature_list", "feature_cols"):
            v = obj.get(k)
            if _is_str_list(v):
                return _dedupe(list(v))

        feats = obj.get("features")

        # Current schema: {"features": {"numeric_cols": [...], "cat_cols": [...]} }
        if isinstance(feats, dict):
            num = feats.get("numeric_cols")
            cat = feats.get("cat_cols")
            if _is_str_list(num) or _is_str_list(cat):
                out: list[str] = []
                if _is_str_list(num):
                    out.extend(list(num))
                if _is_str_list(cat):
                    out.extend(list(cat))
                return _dedupe(out)

            # Legacy schema: {"features": {feature_name: {...}, ...}}
            # Guard against accidentally returning structural keys.
            structural = {"numeric_cols", "cat_cols"}
            keys = [str(k) for k in feats.keys() if isinstance(k, str) and k not in structural]
            if keys:
                return _dedupe(keys)

        # List of strings or dicts containing a name-like key
        if isinstance(feats, list):
            out: list[str] = []
            for item in feats:
                if isinstance(item, str) and item:
                    out.append(item)
                    continue
                if isinstance(item, dict):
                    for nk in ("name", "key", "feature", "col", "column"):
                        nv = item.get(nk)
                        if isinstance(nv, str) and nv:
                            out.append(nv)
                            break
            if out:
                return _dedupe(out)

        # Some manifests nest under "schema"
        schema = obj.get("schema")
        if isinstance(schema, dict):
            got = _from(schema)
            if got:
                return got

        return []

    return _from(feature_manifest)


@dataclass(frozen=True)
class ArtifactBundle:
    meta_dir: Path
    bundle_id: str
    model_kind: str  # "sklearn_pipeline" or "legacy_lgbm"
    feature_manifest: dict
    feature_names: list[str]

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

    feature_names = _extract_feature_names_from_manifest(feature_manifest)
    if strict and (not feature_names):
        raise ValueError(
            f"feature_manifest.json yielded 0 feature names (schema mismatch). "
            f"path={meta_dir_p / 'feature_manifest.json'}"
        )
    
    feature_set = set(feature_names)


    daily_truth_p = meta_dir_p / REGIME_DAILY_TRUTH_FILE
    markov_truth_p = meta_dir_p / REGIME_MARKOV4H_TRUTH_FILE

    include_regimes = bool(feature_manifest.get("include_regimes_as_features", True))
    needs_regime_truth = bool(feature_set & REGIME_FEATURE_KEYS) if include_regimes else False

    has_daily = daily_truth_p.exists()
    has_markov = markov_truth_p.exists()
    has_both = has_daily and has_markov

    auto_extra: list[str] = []

    # If the files exist, attach them (this is your current situation).
    # If the manifest needs them, require them in strict mode.
    if has_both or needs_regime_truth:
        auto_extra.extend([REGIME_DAILY_TRUTH_FILE, REGIME_MARKOV4H_TRUTH_FILE])

    if strict:
        # Fail closed on partial presence (prevents silent drift).
        if has_daily != has_markov:
            raise BundleError(
                "Inconsistent regime truth artifacts: "
                f"{REGIME_DAILY_TRUTH_FILE} exists={has_daily}, "
                f"{REGIME_MARKOV4H_TRUTH_FILE} exists={has_markov}"
            )
        if needs_regime_truth and not has_both:
            missing = []
            if not has_daily:
                missing.append(str(daily_truth_p))
            if not has_markov:
                missing.append(str(markov_truth_p))
            raise BundleError("Regime truth artifacts required by manifest are missing: " + ", ".join(missing))


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

    def _safe_float01(x: Any) -> Optional[float]:
        try:
            v = float(x)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        if v < 0.0 or v > 1.0:
            return None
        return v

    def _thr_from_scope_entry(entry: Any) -> Optional[float]:
        """
        thresholds_by_scope[scope] can be:
          - float
          - {"threshold": 0.xx, ...}
          - {"best": {..., "threshold": 0.xx}, "top5": [...]}   (your current export)
        """
        if isinstance(entry, (float, int)):
            return _safe_float01(entry)

        if isinstance(entry, dict):
            if "threshold" in entry:
                return _safe_float01(entry.get("threshold"))

            best = entry.get("best")
            if isinstance(best, dict) and ("threshold" in best):
                return _safe_float01(best.get("threshold"))

            top5 = entry.get("top5")
            if isinstance(top5, list) and top5:
                t0 = top5[0]
                if isinstance(t0, dict) and ("threshold" in t0):
                    return _safe_float01(t0.get("threshold"))

        return None

    pstar: Optional[float] = None
    pstar_scope: Optional[str] = None

    # 1) Direct legacy keys (if present)
    if isinstance(thresholds, dict):
        pstar = _safe_float01(thresholds.get("pstar") or thresholds.get("p*") or thresholds.get("p_star"))
        sc = thresholds.get("scope") or thresholds.get("selected_scope") or thresholds.get("pstar_scope")
        pstar_scope = str(sc).strip() if isinstance(sc, str) and sc.strip() else None

    # 2) thresholds_by_scope structure (your current export)
    if (pstar is None) and isinstance(thresholds, dict):
        tbs = thresholds.get("thresholds_by_scope")
        if isinstance(tbs, dict) and tbs:
            # Prefer thresholds.json selected_scope if valid
            sc = thresholds.get("selected_scope") or thresholds.get("scope")
            if isinstance(sc, str) and sc.strip() and sc.strip() in tbs:
                p = _thr_from_scope_entry(tbs.get(sc.strip()))
                if p is not None:
                    pstar = p
                    pstar_scope = sc.strip()

    # 3) deployment_config.json decision (tie-breaker / fallback)
    if pstar is None:
        dep_path = meta_dir_p / "deployment_config.json"
        dep = _read_json(dep_path) if dep_path.exists() else None
        if isinstance(dep, dict):
            dec = dep.get("decision")
            if isinstance(dec, dict):
                sc = dec.get("scope")
                th = dec.get("threshold")
                if isinstance(sc, str) and sc.strip():
                    pstar_scope = sc.strip()

                # If thresholds_by_scope exists, prefer its per-scope "best.threshold"
                if isinstance(thresholds, dict):
                    tbs = thresholds.get("thresholds_by_scope")
                    if isinstance(tbs, dict) and pstar_scope and (pstar_scope in tbs):
                        p = _thr_from_scope_entry(tbs.get(pstar_scope))
                        if p is not None:
                            pstar = p

                # else fall back to deployment decision threshold itself
                if pstar is None:
                    pstar = _safe_float01(th)

    # Fail-closed if still None (caller will enforce no-trade)


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
        feature_names=feature_names,
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
