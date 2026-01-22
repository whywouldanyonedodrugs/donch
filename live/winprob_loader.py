# live/winprob_loader.py
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .artifact_bundle import ArtifactBundle, BundleError, SchemaError, load_bundle

LOG = logging.getLogger("winprob")


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
    Supported manifest formats (best-effort):

    1) {"features": {"f1": {"dtype": "float64", "kind": "numeric", ...}, ...}}
    2) {"f1": {"dtype": "float64", ...}, "f2": {...}, ...}
    3) [{"name": "f1", "dtype": "float64", ...}, ...]
    """
    items: List[Tuple[str, Any]] = []

    if isinstance(manifest_obj, dict):
        if "features" in manifest_obj and isinstance(manifest_obj["features"], dict):
            items = list(manifest_obj["features"].items())
        else:
            items = [(k, v) for k, v in manifest_obj.items() if isinstance(k, str)]
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

        if isinstance(desc, str):
            dtype = desc
            kind = "categorical" if ("cat" in dtype or "str" in dtype or "object" in dtype) else "numeric"
            specs.append(FeatureSpec(name=name, kind=kind, dtype=dtype))
            continue

        if not isinstance(desc, dict):
            raise BundleError(f"Invalid feature spec for {name}: {desc}")

        dtype = str(desc.get("dtype", desc.get("type", "float64")))
        raw_kind = desc.get("kind", desc.get("role", None))
        cats = desc.get("categories", desc.get("cats", None))
        codes = desc.get("codes", desc.get("codebook", None))

        if raw_kind is None:
            if cats is not None or "category" in dtype or dtype in ("object", "str", "string"):
                kind = "categorical"
            else:
                kind = "numeric"
        else:
            rk = str(raw_kind).lower()
            kind = "categorical" if rk in ("cat", "categorical", "category") else "numeric"

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


class WinProbScorer:
    """Strict, deterministic scorer for the exported meta-model bundle."""

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
        self.dir = bundle.meta_dir
        self.bundle_id = bundle.bundle_id
        self.model = bundle.model
        self.model_kind = bundle.model_kind
        self.ohe = bundle.ohe
        self.calibrator = bundle.calibrator
        self.pstar = bundle.pstar
        self.feature_names: List[str] = list(bundle.feature_names)

        self.strict_schema = bool(strict_schema)

        # Raw schema
        self._raw_specs = _parse_manifest(bundle.feature_manifest)
        self._raw_spec_by_name = {s.name: s for s in self._raw_specs}
        self.raw_features: List[str] = [s.name for s in self._raw_specs]

        # Cat columns as used during training (order matters)
        try:
            self.raw_cat_cols: List[str] = list(getattr(self.ohe, "feature_names_in_", []))
        except Exception:
            self.raw_cat_cols = []

        raw_cat_set = set(self.raw_cat_cols)
        self.raw_num_cols: List[str] = []
        for s in self._raw_specs:
            if s.kind == "categorical" or s.name in raw_cat_set:
                continue
            self.raw_num_cols.append(s.name)

        self.is_loaded = True

        # Diag / identical-vector detection
        self._diag_once = False
        self._last_hash = None
        self._same_vec_count = 0

        self._validate_bundle_consistency()

    def score(self, raw_row: Dict[str, Any]) -> float:
        """Return calibrated win-probability in [0,1]. Raises SchemaError on invalid raw_row."""
        p_raw, p_cal = self.score_with_details(raw_row)
        return p_cal

    def score_with_details(self, raw_row: Dict[str, Any]) -> Tuple[float, float]:
        X, vec_hash = self._build_X(raw_row)
        p_raw = self._predict_proba(X)
        p_cal = self._calibrate(p_raw)
        self._diag(vec_hash)
        return p_raw, p_cal

    def _validate_bundle_consistency(self) -> None:
        n = len(self.feature_names)
        if n <= 0:
            raise BundleError("feature_names.json is empty")

        try:
            if self.model_kind == "lgb_booster":
                model_names = list(getattr(self.model, "feature_name", lambda: [])())
                if model_names and len(model_names) != n:
                    raise BundleError(f"model.txt feature count {len(model_names)} != feature_names.json {n}")
            else:
                n_model = int(getattr(self.model, "n_features_in_", n))
                if n_model != n:
                    raise BundleError(f"joblib model expects {n_model} features != feature_names.json {n}")
        except BundleError:
            raise
        except Exception:
            pass

    def _validate_raw_schema(self, raw_row: Dict[str, Any]) -> None:
        if not isinstance(raw_row, dict):
            raise SchemaError(f"raw_row must be dict, got {type(raw_row)}")

        keys = set(raw_row.keys())
        required = set(self.raw_features)

        missing = sorted(required - keys)
        extra = sorted(keys - required)

        if missing:
            raise SchemaError(
                f"Missing required raw features: {missing[:40]}" + (" ..." if len(missing) > 40 else "")
            )
        if self.strict_schema and extra:
            raise SchemaError(
                f"Extra raw features not in manifest: {extra[:40]}" + (" ..." if len(extra) > 40 else "")
            )

        for name in self.raw_features:
            spec = self._raw_spec_by_name[name]
            v = raw_row.get(name)

            if spec.kind == "numeric":
                if v is None or _is_nan(v) or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
                    raise SchemaError(f"Invalid numeric value for {name}: {v}")
                if isinstance(v, bool):
                    raise SchemaError(f"Invalid numeric value for {name} (bool not allowed): {v}")
                try:
                    float(v)
                except Exception as e:
                    raise SchemaError(f"Numeric feature {name} not castable to float: {v} ({e})") from e

            else:
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    raise SchemaError(f"Invalid categorical value for {name}: {v}")

                allowed: Optional[set] = None
                if spec.categories is not None:
                    allowed = set(str(x) for x in spec.categories)
                if spec.codes is not None:
                    allowed_codes = set(str(k) for k in spec.codes.keys()) | set(str(x) for x in spec.codes.values())
                    allowed = allowed_codes if allowed is None else (allowed | allowed_codes)

                if allowed is not None and str(v) not in allowed:
                    raise SchemaError(f"Categorical {name} value '{v}' not in allowed set")

    def _build_X(self, raw_row: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        self._validate_raw_schema(raw_row)

        # Build cat frame for OHE
        cat_vals: Dict[str, Any] = {}
        for c in self.raw_cat_cols:
            if c not in raw_row:
                raise SchemaError(f"OHE expects raw categorical '{c}' but it's missing from raw_row/manifest")
            cat_vals[c] = raw_row[c]
        cat_df = pd.DataFrame([cat_vals]) if self.raw_cat_cols else pd.DataFrame(index=[0])

        # OHE outputs
        ohe_cols_out: List[str] = []
        ohe_vals: Optional[np.ndarray] = None
        if self.raw_cat_cols:
            try:
                Xo = self.ohe.transform(cat_df[self.raw_cat_cols].astype(str))
                if hasattr(Xo, "toarray"):
                    Xo = Xo.toarray()
                ohe_vals = np.asarray(Xo, dtype=np.float64)
                ohe_cols_out = list(self.ohe.get_feature_names_out(self.raw_cat_cols))
            except Exception as e:
                raise SchemaError(f"OHE transform failed: {e}") from e

        # Final model matrix in fixed column order
        X_df = pd.DataFrame(np.zeros((1, len(self.feature_names)), dtype=np.float64), columns=self.feature_names)

        # Fill numeric raw features
        for c in self.raw_num_cols:
            if c in X_df.columns:
                X_df.at[0, c] = float(raw_row[c])

        # Fill OHE outputs (only those present in feature_names)
        if ohe_vals is not None and ohe_cols_out:
            common = [c for c in ohe_cols_out if c in X_df.columns]
            if common:
                idx_in_ohe = [ohe_cols_out.index(c) for c in common]
                X_df.loc[0, common] = ohe_vals[0, idx_in_ohe]

        vec = X_df.to_numpy(dtype=np.float64, copy=False)
        vec_hash = hashlib.md5(vec.tobytes()).hexdigest()
        return vec, vec_hash

    def _predict_proba(self, X: np.ndarray) -> float:
        if X.ndim != 2 or X.shape[0] != 1:
            raise ValueError(f"X must be shape (1, n_features), got {X.shape}")

        if self.model_kind == "lgb_booster":
            try:
                p = float(self.model.predict(X)[0])
            except Exception as e:
                raise BundleError(f"LightGBM Booster predict failed: {e}") from e
        else:
            try:
                p = float(self.model.predict_proba(X)[:, 1][0])
            except Exception as e:
                raise BundleError(f"Sklearn model predict_proba failed: {e}") from e

        if not np.isfinite(p):
            raise BundleError(f"Model returned non-finite probability: {p}")
        return float(min(max(p, 0.0), 1.0))

    def _calibrate(self, p_raw: float) -> float:
        cal = self.calibrator
        if cal is None:
            return p_raw

        if isinstance(cal, dict):
            ctype = str(cal.get("type", cal.get("kind", ""))).lower()
            if ctype == "platt":
                a = float(cal.get("a", 1.0))
                b = float(cal.get("b", 0.0))
                z = a * p_raw + b
                return float(1.0 / (1.0 + np.exp(-z)))
            if ctype == "isotonic":
                xs = cal.get("xs") or cal.get("x")
                ys = cal.get("ys") or cal.get("y")
                if isinstance(xs, list) and isinstance(ys, list) and len(xs) == len(ys) and len(xs) >= 2:
                    return float(np.interp(p_raw, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)))
            return p_raw

        try:
            if hasattr(cal, "predict") and not hasattr(cal, "predict_proba"):
                return float(cal.predict([p_raw])[0])
            if hasattr(cal, "predict_proba"):
                return float(cal.predict_proba(np.array([[p_raw]], dtype=float))[:, 1][0])
        except Exception:
            return p_raw

        return p_raw

    def _diag(self, vec_hash: str) -> None:
        if not self._diag_once:
            LOG.info(
                "WinProb bundle=%s model_kind=%s raw_feats=%d model_cols=%d p*=%s",
                self.bundle_id,
                self.model_kind,
                len(self.raw_features),
                len(self.feature_names),
                (f"{self.pstar:.4f}" if isinstance(self.pstar, (float, int)) else "None"),
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
