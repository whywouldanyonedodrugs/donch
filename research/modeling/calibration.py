# research/modeling/calibration.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

def _safe_prefit(base: BaseEstimator, X: pd.DataFrame, y: np.ndarray) -> BaseEstimator:
    """
    Prefit guard: if a CV split ends up single-class, CalibratedClassifierCV can error.
    We prefit a cloned estimator; if y is single-class overall, use a constant-prob dummy.
    """
    if np.unique(y).size < 2:
        class _ConstProb(BaseEstimator):
            def fit(self, X, y): return self
            def predict_proba(self, X):
                p = float(np.mean(y)) if len(y) else 0.5
                return np.column_stack([1 - np.full(len(X), p), np.full(len(X), p)])
        return _ConstProb().fit(X, y)
    return clone(base).fit(X, y)

def fit_with_best_calibration(
    base_pipeline: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int | None = 3,
) -> Tuple[CalibratedClassifierCV, Dict[str, float]]:
    """
    Fit base model and wrap with probability calibration, trying 'isotonic' and 'sigmoid'.
    Pick the method with the lower Brier loss on the training data.
    Returns (fitted_calibrator, {'isotonic': brier, 'sigmoid': brier}).
    """
    scores: Dict[str, float] = {}
    fitted: Dict[str, CalibratedClassifierCV] = {}

    for method in ("isotonic", "sigmoid"):
        prefit = _safe_prefit(base_pipeline, X, y)
        cal = CalibratedClassifierCV(prefit, cv=cv, method=method)
        cal = cal.fit(X, y)
        proba = cal.predict_proba(X)[:, 1]
        scores[method] = float(brier_score_loss(y, proba))
        fitted[method] = cal

    best = min(scores, key=scores.get)
    return fitted[best], scores
