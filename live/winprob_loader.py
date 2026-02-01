# live/winprob_loader.py
from __future__ import annotations

import hashlib
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings

import numpy as np
import pandas as pd

from .artifact_bundle import ArtifactBundle, BundleError, SchemaError, load_bundle

LOG = logging.getLogger("winprob")


def _infer_model_input_features(model: Any) -> Optional[List[str]]:
    """
    Best-effort: for sklearn estimators/pipelines, prefer feature_names_in_ captured at fit time.
    Returns list[str] or None.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        try:
            return [str(x) for x in list(names)]
        except Exception:
            pass

    # Try pipeline steps (some pipelines attach feature_names_in_ to the final estimator or a step)
    try:
        steps = getattr(model, "named_steps", None)
        if isinstance(steps, dict):
            for step in steps.values():
                names = getattr(step, "feature_names_in_", None)
                if names is not None:
                    try:
                        return [str(x) for x in list(names)]
                    except Exception:
                        continue
    except Exception:
        pass

    return None


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str  # "numeric" | "categorical"
    dtype: str
    categories: Optional[List[Any]] = None
    codes: Optional[Dict[str, Any]] = None


def _is_nan(x: Any) -> bool:
    try:
        return bool(np.isnan(x))
    except Exception:
        return False


def _parse_manifest(manifest_obj: Any) -> List[FeatureSpec]:
    """
    Supports multiple manifest formats.

    Canonical (offline team):
      {
        "features": {
          "numeric_cols": [...],
          "cat_cols": [...]
        },
        ...
      }

    Also supports:
      {"schema": {...}} wrapper
      schema-container at top-level: {"cat_cols": [...], "num_cols"/"numeric_cols": [...]}
      older feature-spec formats (fallback)
    """

    # 1) Unwrap {"schema": {...}} if present
    if isinstance(manifest_obj, dict) and isinstance(manifest_obj.get("schema"), dict):
        inner = manifest_obj["schema"]
        if any(k in inner for k in ("cat_cols", "num_cols", "numeric_cols")):
            manifest_obj = inner

    # 2) Unwrap canonical {"features": {...}} if it looks like a schema container
    if isinstance(manifest_obj, dict) and isinstance(manifest_obj.get("features"), dict):
        inner = manifest_obj["features"]
        if any(k in inner for k in ("cat_cols", "num_cols", "numeric_cols")):
            manifest_obj = inner

    # 3) Primary schema-container format
    if isinstance(manifest_obj, dict) and any(k in manifest_obj for k in ("cat_cols", "num_cols", "numeric_cols")):
        cat_cols = manifest_obj.get("cat_cols") or []

        # canonical key is numeric_cols; accept aliases
        num_cols = manifest_obj.get("numeric_cols")
        if num_cols is None:
            num_cols = manifest_obj.get("num_cols")
        num_cols = num_cols or []

        if not isinstance(cat_cols, list) or not all(isinstance(x, str) for x in cat_cols):
            raise BundleError(f"feature_manifest: cat_cols must be list[str], got {type(cat_cols)}")
        if not isinstance(num_cols, list) or not all(isinstance(x, str) for x in num_cols):
            raise BundleError(f"feature_manifest: numeric_cols/num_cols must be list[str], got {type(num_cols)}")

        # Optional maps (not present in your current bundle, but supported)
        dtypes = manifest_obj.get("dtypes") or manifest_obj.get("dtype_map") or manifest_obj.get("raw_dtypes") or {}
        if dtypes is None:
            dtypes = {}
        if not isinstance(dtypes, dict):
            raise BundleError(f"feature_manifest: dtypes must be dict, got {type(dtypes)}")

        categories_map = manifest_obj.get("categories") or manifest_obj.get("cats") or {}
        if categories_map is None:
            categories_map = {}
        if not isinstance(categories_map, dict):
            raise BundleError(f"feature_manifest: categories must be dict, got {type(categories_map)}")

        codes_map = manifest_obj.get("codes") or manifest_obj.get("codebook") or manifest_obj.get("codebooks") or {}
        if codes_map is None:
            codes_map = {}
        if not isinstance(codes_map, dict):
            raise BundleError(f"feature_manifest: codes must be dict, got {type(codes_map)}")

        specs: List[FeatureSpec] = []

        # IMPORTANT: offline team says authoritative order is numeric_cols + cat_cols
        for name in num_cols:
            dt = str(dtypes.get(name, "float64"))
            specs.append(FeatureSpec(name=name, kind="numeric", dtype=dt))

        for name in cat_cols:
            dt = str(dtypes.get(name, "category"))
            cats = categories_map.get(name)
            codes = codes_map.get(name)
            specs.append(
                FeatureSpec(
                    name=name,
                    kind="categorical",
                    dtype=dt,
                    categories=list(cats) if isinstance(cats, (list, tuple)) else None,
                    codes=dict(codes) if isinstance(codes, dict) else None,
                )
            )

        names = [s.name for s in specs]
        if len(names) != len(set(names)):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise BundleError(f"feature_manifest has duplicate feature names: {dup}")

        return specs

    # 4) Fallback formats (older exports)
    items: List[Tuple[str, Any]] = []

    if isinstance(manifest_obj, dict):
        if "features" in manifest_obj and isinstance(manifest_obj["features"], dict):
            items = list(manifest_obj["features"].items())
        else:
            meta_keys = {
                "cat_cols", "num_cols", "numeric_cols", "dtypes", "dtype_map", "raw_dtypes",
                "categories", "cats", "codes", "codebook", "codebooks",
                "version", "created_at", "notes", "schema",
                "include_regimes_as_features", "target",
            }
            items = [(k, v) for k, v in manifest_obj.items() if isinstance(k, str) and k not in meta_keys]

    elif isinstance(manifest_obj, list):
        for i, it in enumerate(manifest_obj):
            if isinstance(it, dict) and "name" in it:
                items.append((str(it["name"]), it))
            elif isinstance(it, dict) and "feature" in it:
                items.append((str(it["feature"]), it))
            else:
                raise BundleError(f"Unsupported manifest entry at idx={i}: {it}")
    else:
        raise BundleError(f"Unsupported feature_manifest.json format: {type(manifest_obj)}")

    specs: List[FeatureSpec] = []
    for name, desc in items:
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(desc, dict):
            raise BundleError(f"Invalid feature spec for {name}: {desc}")

        dtype = str(desc.get("dtype", desc.get("type", "float64")))
        raw_kind = desc.get("kind", desc.get("role", None))
        cats = desc.get("categories", desc.get("cats", None))
        codes = desc.get("codes", desc.get("codebook", None))

        if raw_kind is None:
            kind = "categorical" if (cats is not None or "category" in dtype or dtype in ("object", "str", "string")) else "numeric"
        else:
            kind = "categorical" if str(raw_kind).lower() in ("cat", "categorical", "category") else "numeric"

        specs.append(
            FeatureSpec(
                name=name,
                kind=kind,
                dtype=dtype,
                categories=list(cats) if isinstance(cats, (list, tuple)) else None,
                codes=dict(codes) if isinstance(codes, dict) else None,
            )
        )

    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise BundleError(f"feature_manifest has duplicate feature names: {dup}")

    return specs

def _iter_json(obj: Any, path: str = ""):
    """
    Yield (path, key, value) for all dict entries in a nested JSON-like object.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield (p, k, v)
            yield from _iter_json(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            yield from _iter_json(v, p)


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


def _threshold_from_scope_entry(entry: Any) -> Any:
    """
    thresholds_by_scope[scope] can be:
      - float
      - {"threshold": 0.xx, ...}
      - {"best": {..., "threshold": 0.xx}, "top5": [...]}  (your current export)
    """
    if isinstance(entry, dict):
        if "threshold" in entry:
            return entry.get("threshold")
        best = entry.get("best")
        if isinstance(best, dict) and ("threshold" in best):
            return best.get("threshold")
        top5 = entry.get("top5")
        if isinstance(top5, list) and top5:
            t0 = top5[0]
            if isinstance(t0, dict) and ("threshold" in t0):
                return t0.get("threshold")
        return None
    return entry


def _extract_pstar_from_thresholds(
    thresholds_obj: Any,
    *,
    meta_dir: Path,
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Derive (pstar, pstar_scope, src) from exported artifacts.

    Supported:
      - legacy direct keys in thresholds.json: pstar / p* / p_star (+ optional scope)
      - structured thresholds.json:
          {
            "thresholds_by_scope": {
              "<SCOPE>": {"threshold": 0.xx, ...} | 0.xx,
              ...
            },
            ...
          }
        plus deployment_config.json to select the active scope/threshold deterministically.

    Fail-closed if ambiguous or missing.
    """
    if not isinstance(thresholds_obj, dict):
        return (None, None, None)

    # 1) Legacy direct keys
    direct = thresholds_obj.get("pstar") or thresholds_obj.get("p*") or thresholds_obj.get("p_star")
    p_direct = _safe_float01(direct)
    if p_direct is not None:
        sc = thresholds_obj.get("scope") or thresholds_obj.get("pstar_scope") or thresholds_obj.get("selected_scope")
        sc = str(sc).strip() if isinstance(sc, str) and sc.strip() else None
        return (p_direct, sc, "artifact:thresholds.json:direct")

    # 2) Structured thresholds_by_scope
    tbs = thresholds_obj.get("thresholds_by_scope")
    if not isinstance(tbs, dict) or not tbs:
        return (None, None, None)

    scopes = [str(k) for k in tbs.keys() if isinstance(k, str) and k.strip()]
    scope_set = set(scopes)

    # 2a) Try explicit scope hints inside thresholds.json itself
    for k in ("scope", "selected_scope", "active_scope", "default_scope", "best_scope", "chosen_scope"):
        v = thresholds_obj.get(k)
        if isinstance(v, str) and v.strip() in scope_set:
            chosen_scope = v.strip()
            entry = tbs.get(chosen_scope)

            thr = _threshold_from_scope_entry(entry)

            p = _safe_float01(thr)
            if p is not None:
                return (p, chosen_scope, f"artifact:thresholds.json:{k}")

    # 2b) Use deployment_config.json to choose scope/threshold
    deploy_path = meta_dir / "deployment_config.json"
    deploy_obj = None
    if deploy_path.exists():
        try:
            deploy_obj = json.loads(deploy_path.read_text())
        except Exception:
            deploy_obj = None

    deploy_scope: Optional[str] = None
    deploy_thr: Optional[float] = None

    if isinstance(deploy_obj, dict):
        # Find any scope-like string that matches thresholds_by_scope keys
        scope_candidates: list[Tuple[str, str]] = []
        thr_candidates: list[Tuple[str, float]] = []

        for pth, key, val in _iter_json(deploy_obj):
            if isinstance(key, str):
                kl = key.lower()
                if "scope" in kl and isinstance(val, str) and val.strip() in scope_set:
                    scope_candidates.append((pth, val.strip()))
                if key == "threshold":
                    vv = _safe_float01(val)
                    if vv is not None:
                        thr_candidates.append((pth, vv))

        # Prefer the first matching scope candidate (deterministic traversal order)
        if scope_candidates:
            deploy_scope = scope_candidates[0][1]

        # Prefer the first threshold candidate
        if thr_candidates:
            deploy_thr = thr_candidates[0][1]

    # If we got a scope from deployment config, use the scope's threshold from thresholds.json
    if deploy_scope is not None:
        entry = tbs.get(deploy_scope)

        thr = _threshold_from_scope_entry(entry)

        p = _safe_float01(thr)
        if p is not None:
            return (p, deploy_scope, "artifact:deployment_config.json->thresholds.json:scope")

    # Otherwise, if we only got a deploy threshold, match it uniquely to a scope
    if deploy_thr is not None:
        matches: list[str] = []
        for sc in scopes:
            entry = tbs.get(sc)

            thr = _threshold_from_scope_entry(entry)

            p = _safe_float01(thr)
            if p is None:
                continue
            if abs(p - deploy_thr) <= 1e-12:
                matches.append(sc)

        if len(matches) == 1:
            return (deploy_thr, matches[0], "artifact:deployment_config.json:threshold->thresholds.json:unique_match")

    # Ambiguous / missing => fail-closed
    return (None, None, None)


class WinProbScorer:
    """
    Strict, deterministic scorer.
    For your current export, model.joblib is assumed to be a sklearn pipeline that accepts a
    one-row DataFrame of raw features.
    """

    def __init__(
        self,
        artifact_dir: str | Path | None = None,
        *,
        bundle: ArtifactBundle | None = None,
        strict_schema: bool = True,
    ):
        if bundle is None:
            if artifact_dir is None:
                artifact_dir = "results/meta_export"
            bundle = load_bundle(artifact_dir, strict=True)

        self.bundle = bundle
        self.bundle_id = bundle.bundle_id
        self.model = bundle.model
        self.model_kind = bundle.model_kind
        self.calibrator = bundle.calibrator

        # p* / scope thresholding (strict-parity):
        # Prefer bundle-provided pstar, else derive from thresholds.json + deployment_config.json.
        self.pstar = getattr(bundle, "pstar", None)
        self.pstar_scope = getattr(bundle, "pstar_scope", None)
        self.pstar_src: Optional[str] = None

        if self.pstar is None or (isinstance(self.pstar, (float, int)) and (not np.isfinite(float(self.pstar)))):
            thr_obj = getattr(bundle, "thresholds", None)
            meta_dir = Path(getattr(bundle, "meta_dir", artifact_dir if artifact_dir is not None else "."))
            p, sc, src = _extract_pstar_from_thresholds(thr_obj, meta_dir=meta_dir)
            self.pstar = p
            self.pstar_scope = sc
            self.pstar_src = src
        else:
            self.pstar_src = "artifact:thresholds.json:direct_or_bundle"

        self.strict_schema = bool(strict_schema)


        # Parse manifest → raw feature specs
        self._raw_specs = _parse_manifest(bundle.feature_manifest)
        self._raw_spec_by_name = {s.name: s for s in self._raw_specs}
        self.raw_features = [s.name for s in self._raw_specs]
        self.raw_cat_cols = [s.name for s in self._raw_specs if s.kind == "categorical"]
        self.raw_num_cols = [s.name for s in self._raw_specs if s.kind == "numeric"]

        # Infer model input feature names (Option A: sklearn pipeline uses DataFrame column names).
        inferred = _infer_model_input_features(self.model)
        if inferred:
            self.model_features = inferred
        else:
            # Fallback: assume model was trained with the manifest ordering.
            self.model_features = list(self.raw_features)

        # Safety: enforce that manifest == model expected inputs (set equality).
        # If you ever intentionally allow model to consume a strict subset, relax this explicitly.
        raw_set = set(self.raw_features)
        model_set = set(self.model_features)
        if raw_set != model_set:
            missing_in_raw = sorted(model_set - raw_set)
            extra_in_raw = sorted(raw_set - model_set)
            raise BundleError(
                "Model/manifest feature mismatch. "
                f"missing_in_manifest={missing_in_raw} extra_in_manifest={extra_in_raw}"
            )

        self._diag_once = False
        self._last_hash = None
        self._same_vec_count = 0

    @property
    def required_keys(self) -> list[str]:
        # Backwards-compatible alias used by live_trader integration.
        return list(self.raw_features)


    def score(self, raw_row: Dict[str, Any]) -> float:
        p_raw, p_cal = self.score_with_details(raw_row)
        return p_cal

    def score_with_details(self, raw_row: Dict[str, Any]) -> Tuple[float, float]:
        df, vec_hash = self._build_df(raw_row)
        p_raw = self._predict_proba(df)
        p_cal = self._calibrate(p_raw)
        self._diag(vec_hash)
        return p_raw, p_cal

    def _validate_raw_schema(self, raw_row: Dict[str, Any]) -> None:
        if not isinstance(raw_row, dict):
            raise SchemaError(f"raw_row must be dict, got {type(raw_row)}")

        keys = set(raw_row.keys())
        required = set(self.raw_features)

        missing = sorted(required - keys)
        extra = sorted(keys - required)

        if missing:
            raise SchemaError(f"Missing required raw features: {missing[:50]}" + (" ..." if len(missing) > 50 else ""))
        if self.strict_schema and extra:
            raise SchemaError(f"Extra raw features not in manifest: {extra[:50]}" + (" ..." if len(extra) > 50 else ""))

        for name in self.raw_features:
            spec = self._raw_spec_by_name[name]
            v = raw_row.get(name)

            # live/winprob_loader.py (inside _validate_raw_schema)

            if spec.kind == "numeric":
                if v is None:
                    raise SchemaError(f"Missing required numeric feature: {name}")

                # Reject bools (they cast to float but are almost always a bug)
                if isinstance(v, (bool, np.bool_)):
                    raise SchemaError(f"Invalid numeric value for {name}: {v}")

                # Allow NaN, but reject +/-inf and non-castable values
                try:
                    fv = float(v)
                except Exception:
                    raise SchemaError(f"Invalid numeric value for {name}: {v}")

                if np.isinf(fv):
                    raise SchemaError(f"Invalid numeric value for {name}: {v}")

                # NOTE: NaN is allowed here on purpose; downstream pipeline/model should handle it.

            else:
                if v is None:
                    raise SchemaError(f"Invalid categorical value for {name}: {v}")

                allowed: Optional[set] = None
                if spec.categories is not None:
                    allowed = set(str(x) for x in spec.categories)
                if spec.codes is not None:
                    allowed_codes = set(str(k) for k in spec.codes.keys()) | set(str(x) for x in spec.codes.values())
                    allowed = allowed_codes if allowed is None else (allowed | allowed_codes)

                if allowed is not None and str(v) not in allowed:
                    raise SchemaError(f"Categorical {name} value '{v}' not in allowed set")

    def _build_df(self, raw_row: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        self._validate_raw_schema(raw_row)

        data: Dict[str, Any] = {}
        for name in self.model_features:
            s = self._raw_spec_by_name[name]
            v = raw_row[name]
            if s.kind == "numeric":
                data[name] = float(v)
            else:
                data[name] = str(v)

        df = pd.DataFrame([data], columns=list(self.model_features))
        vec_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
        return df, vec_hash


    def _predict_proba(self, df: pd.DataFrame) -> float:
        try:
            with warnings.catch_warnings():
                # LightGBM sklearn wrapper warns if it was fit with feature names but receives numpy
                # (common when a sklearn Pipeline/transformer converts DataFrame->ndarray).
                warnings.filterwarnings(
                    "ignore",
                    message=r"X does not have valid feature names, but .* was fitted with feature names",
                    category=UserWarning,
                )

                if hasattr(self.model, "predict_proba"):
                    p = float(self.model.predict_proba(df)[:, 1][0])
                else:
                    p = float(self.model.predict(df)[0])

        except Exception as e:
            raise BundleError(f"Model predict failed (kind={self.model_kind}): {e}") from e

        if not np.isfinite(p):
            raise BundleError(f"Model returned non-finite probability: {p}")
        return float(min(max(p, 0.0), 1.0))



    def _calibrate(self, p_raw: float) -> float:
        cal = self.calibrator
        if cal is None:
            return p_raw

        # If the calibrator was fit on a pandas DataFrame, it may carry feature names
        # (feature_names_in_) and emit sklearn's "X does not have valid feature names"
        # warning when passed a bare ndarray/list. Prefer a 1-col DataFrame when possible.
        try:
            if hasattr(cal, "predict_proba") or hasattr(cal, "predict"):
                in_feats = _infer_model_input_features(cal)
                if in_feats and len(in_feats) == 1:
                    df_in = pd.DataFrame([{in_feats[0]: float(p_raw)}], columns=list(in_feats))
                    if hasattr(cal, "predict_proba"):
                        return float(cal.predict_proba(df_in)[:, 1][0])
                    return float(cal.predict(df_in)[0])

                x = np.array([[float(p_raw)]], dtype=float)
                if hasattr(cal, "predict_proba"):
                    return float(cal.predict_proba(x)[:, 1][0])
                if hasattr(cal, "predict"):
                    return float(cal.predict(x)[0])
        except Exception:
            pass

        if isinstance(cal, dict):
            ctype = str(cal.get("type", cal.get("kind", ""))).lower()
            if ctype == "platt":
                a = float(cal.get("a", 1.0))
                b = float(cal.get("b", 0.0))
                z = a * float(p_raw) + b
                return float(1.0 / (1.0 + np.exp(-z)))
            if ctype == "isotonic":
                xs = cal.get("xs") or cal.get("x")
                ys = cal.get("ys") or cal.get("y")
                if isinstance(xs, list) and isinstance(ys, list) and len(xs) == len(ys) and len(xs) >= 2:
                    return float(np.interp(float(p_raw), np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))

        return p_raw


    def _diag(self, vec_hash: str) -> None:
        if not self._diag_once:
            # Compute diagnostics locally (avoid NameError; deterministic)
            raw_feats = len(
                (getattr(self, "required_keys", None) or getattr(self, "raw_features", None) or [])  # type: ignore[arg-type]
            )
            model_cols = len(getattr(self, "model_features", []) or [])
            pstar = getattr(self, "pstar", None)
            pstar_s = (
                f"{float(pstar):.6f}"
                if isinstance(pstar, (int, float, np.floating))
                else "None"
            )

            LOG.info(
                "WinProb ready (bundle=%s, raw_feats=%d, model_cols=%d, p*=%s)",
                self.bundle_id,
                raw_feats,
                model_cols,
                pstar_s,
            )

            self._diag_once = True


        if self._last_hash is None:
            self._last_hash = vec_hash
            return

        if vec_hash == self._last_hash:
            self._same_vec_count += 1
            if self._same_vec_count in (5, 25, 100):
                LOG.warning(
                    "[WINPROB DIAG] %d consecutive identical feature vectors (hash=%s, bundle=%s)",
                    self._same_vec_count,
                    vec_hash,
                    self.bundle_id,
                )
        else:
            self._last_hash = vec_hash
            self._same_vec_count = 0
