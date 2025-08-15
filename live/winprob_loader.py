# live/winprob_loader.py
from __future__ import annotations

import joblib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("winprob")

@dataclass
class _Loaded:
    kind: str                          # "statsmodels" | "sklearn_bundle" | "sklearn_dict" | "sklearn_estimator"
    model: Any
    features: Sequence[str]
    path: Optional[Path] = None        # <— new: remember which file we loaded

class WinProbScorer:
    """
    Loads a win-probability model from disk and scores a single feature dict.
    Supports:
      1) statsmodels results (expects .model.exog_names and .predict)
      2) research ModelBundle (with .feature_names and .calibrator.predict_proba)
      3) legacy sklearn dict {"pipeline": estimator, "features": [...]}
    """
    def __init__(self, paths: Optional[Sequence[str]] = None):
        # Default search order: joblib (statsmodels), then research pkl, then any leftover pkl
        self.paths = [Path(p) for p in (paths or [
            "win_probability_model.joblib",
            "win_probability_model.pkl",
        ])]
        self._loaded: Optional[_Loaded] = None
        self._try_load_any()

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def expected_features(self) -> Sequence[str]:
        return self._loaded.features if self._loaded else []

    @property
    def kind(self) -> str:             # <— new: expose kind for logging in live_trader
        return self._loaded.kind if self._loaded else "none"

    def _try_load_any(self):
        for p in self.paths:
            if not p.exists():
                continue
            try:
                obj = joblib.load(p)
                loaded = self._coerce(obj)
                if loaded:
                    self._loaded = loaded
                    LOG.info(                       # <— one-line, clear startup message
                        "WinProb model loaded: kind=%s file=%s features=%d",
                        loaded.kind, p.name, len(loaded.features)
                    )
                    return
            except Exception as e:
                LOG.warning("Failed to load model %s: %s", p, e)
        LOG.warning("No compatible win-probability model found. Sizing will use wp=0.0.")

    def _coerce(self, obj: Any) -> Optional[_Loaded]:
        # 1) statsmodels results
        try:
            import statsmodels.api as sm  # noqa: F401 (import to ensure dep exists)
            if hasattr(obj, "model") and hasattr(obj, "predict"):
                exog = getattr(obj.model, "exog_names", None)
                if isinstance(exog, (list, tuple)) and "const" in exog:
                    feats = [c for c in exog if c != "const"]
                    return _Loaded(kind="statsmodels", model=obj, features=feats)
        except Exception:
            pass

        # 2) research ModelBundle (dataclass with .feature_names and .calibrator)
        if hasattr(obj, "feature_names") and hasattr(obj, "calibrator"):
            cal = getattr(obj, "calibrator")
            if hasattr(cal, "predict_proba"):
                feats = list(getattr(obj, "feature_names"))
                return _Loaded(kind="sklearn_bundle", model=cal, features=feats)

        # 3) legacy sklearn dict {"pipeline": estimator, "features": [...]}
        if isinstance(obj, dict) and "pipeline" in obj and "features" in obj:
            est = obj["pipeline"]
            feats = list(obj["features"])
            if hasattr(est, "predict_proba"):
                return _Loaded(kind="sklearn_dict", model=est, features=feats)

        # 4) plain sklearn estimator with feature names
        if hasattr(obj, "predict_proba"):
            feats = getattr(obj, "feature_names_in_", None)
            if feats is not None:
                return _Loaded(kind="sklearn_estimator", model=obj, features=list(feats))

        return None

    def score(self, features: dict) -> float:
        """Return calibrated P(win) in [0,1] for a single example."""
        if not self._loaded:
            return 0.0

        kind = self._loaded.kind
        model = self._loaded.model
        cols = list(self._loaded.features)

        X = pd.DataFrame([features], index=[0])
        X = X.reindex(columns=cols, fill_value=0.0)

        try:
            if kind == "statsmodels":
                import statsmodels.api as sm
                Xc = sm.add_constant(X.astype(float), prepend=True, has_constant="add")
                p = float(model.predict(Xc)[0])

            else:
                # sklearn flavors
                p = float(model.predict_proba(X)[:, 1][0])

        except Exception as e:
            LOG.warning("WinProb scoring failed (%s); returning 0.0", e)
            return 0.0

        # clamp
        return max(0.0, min(1.0, p))
