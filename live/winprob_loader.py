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
    """
    Strict, deterministic scorer.

    Modes:
      - external_ohe: uses bundle.ohe + bundle.feature_names to build model matrix
      - pipeline_raw: uses model.joblib pipeline to transform raw DataFrame internally
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
        self.pstar = bundle.pstar
        self.strict_schema = bool(strict_schema)

        self._raw_specs = _parse_manifest(bundle.feature_manifest)
        self._raw_spec_by_name = {s.name: s for s in self._raw_specs}
        self.raw_features = [s.name for s in self._raw_specs]

        # Decide mode
        if bundle.model_kind in ("lgb_booster", "lgbm_sklearn"):
            if bundle.ohe is None or not bundle.feature_names:
                raise BundleError("external_ohe mode requires ohe.joblib and feature_names.json")
            self.mode = "external_ohe"
            self.ohe = bundle.ohe
            self.feature_names = list(bundle.feature_names)

            # determine which raw cols are categorical for OHE
            try:
                self.raw_cat_cols = list(getattr(self.ohe, "feature_names_in_", []))
            except Exception:
                self.raw_cat_cols = []
            cat_set = set(self.raw_cat_cols)
            self.raw_num_cols = [s.name for s in self._raw_specs if (s.name not in cat_set and s.kind != "categorical")]

        else:
            # pipeline_raw (model.joblib)
            self.mode = "pipeline_raw"
            self.ohe = None
            self.feature_names = list(bundle.feature_names) if bundle.feature_names else []

            # For pipeline_raw, we still want stable dtype casting:
            self.raw_cat_cols = [s.name for s in self._raw_specs if s.kind == "categorical"]
            self.raw_num_cols = [s.name for s in self._raw_specs if s.kind == "numeric"]

        # diag
        self._diag_once = False
        self._last_hash = None
        self._same_vec_count = 0

    def score(self, raw_row: Dict[str, Any]) -> float:
        p_raw, p_cal = self.score_with_details(raw_row)
        return p_cal

    def score_with_details(self, raw_row: Dict[str, Any]) -> Tuple[float, float]:
        X_obj, vec_hash = self._build_X(raw_row)
        p_raw = self._predict_proba(X_obj)
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

    def _build_X(self, raw_row: Dict[str, Any]) -> Tuple[Any, str]:
        self._validate_raw_schema(raw_row)

        if self.mode == "pipeline_raw":
            # Build DataFrame with correct dtypes (no lookahead / strict schema already enforced)
            data = {}
            for s in self._raw_specs:
                v = raw_row[s.name]
                if s.kind == "numeric":
                    data[s.name] = float(v)
                else:
                    data[s.name] = str(v)
            df = pd.DataFrame([data], columns=[s.name for s in self._raw_specs])

            vec_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
            return df, vec_hash

        # external_ohe mode: manual OHE + fixed feature_names matrix
        cat_vals = {c: raw_row[c] for c in self.raw_cat_cols}
        cat_df = pd.DataFrame([cat_vals]) if self.raw_cat_cols else pd.DataFrame(index=[0])

        try:
            Xo = self.ohe.transform(cat_df[self.raw_cat_cols].astype(str))
            if hasattr(Xo, "toarray"):
                Xo = Xo.toarray()
            ohe_vals = np.asarray(Xo, dtype=np.float64)
            ohe_cols_out = list(self.ohe.get_feature_names_out(self.raw_cat_cols))
        except Exception as e:
            raise SchemaError(f"OHE transform failed: {e}") from e

        X_df = pd.DataFrame(np.zeros((1, len(self.feature_names)), dtype=np.float64), columns=self.feature_names)

        for c in self.raw_num_cols:
            if c in X_df.columns:
                X_df.at[0, c] = float(raw_row[c])

        if ohe_vals is not None and ohe_cols_out:
            common = [c for c in ohe_cols_out if c in X_df.columns]
            if common:
                idx = [ohe_cols_out.index(c) for c in common]
                X_df.loc[0, common] = ohe_vals[0, idx]

        vec = X_df.to_numpy(dtype=np.float64, copy=False)
        vec_hash = hashlib.md5(vec.tobytes()).hexdigest()
        return vec, vec_hash

    def _predict_proba(self, X_obj: Any) -> float:
        try:
            if hasattr(self.model, "predict_proba"):
                p = float(self.model.predict_proba(X_obj)[:, 1][0])
            else:
                # LightGBM Booster case not expected here; but keep as guard.
                p = float(self.model.predict(X_obj)[0])
        except Exception as e:
            raise BundleError(f"Model predict failed (mode={self.mode}, kind={self.model_kind}): {e}") from e

        if not np.isfinite(p):
            raise BundleError(f"Model returned non-finite probability: {p}")
        return float(min(max(p, 0.0), 1.0))

    def _calibrate(self, p_raw: float) -> float:
        cal = self.calibrator
        if cal is None:
            return p_raw

        # joblib calibrator (isotonic/logistic etc.)
        try:
            if hasattr(cal, "predict_proba"):
                return float(cal.predict_proba(np.array([[p_raw]], dtype=float))[:, 1][0])
            if hasattr(cal, "predict"):
                return float(cal.predict([p_raw])[0])
        except Exception:
            pass

        # json calibrator (platt or isotonic points)
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

    def _diag(self, vec_hash: str) -> None:
        if not self._diag_once:
            LOG.info(
                "WinProb ready bundle=%s mode=%s model_kind=%s raw_feats=%d p*=%s",
                self.bundle_id,
                self.mode,
                self.model_kind,
                len(self.raw_features),
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
