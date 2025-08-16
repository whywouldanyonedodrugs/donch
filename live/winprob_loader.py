# live/winprob_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


class WinProbScorer:
    """
    Robust loader + scorer for the exported meta-model.

    Looks for (preferred names):
      - donch_meta_lgbm.joblib       (sklearn LightGBM wrapper)
      - ohe.joblib                   (sklearn OneHotEncoder fitted on training cats)
      - feature_names.json           (FINAL input column order used for training; numeric + OHE outputs)
      - calibrator.joblib            (optional Platt/isotonic calibration)
      - pstar.txt                    (optional threshold for live gating)

    If feature_names.json is missing or stale, falls back to model feature
    names from the booster. Always constructs a DataFrame with EXACT column order
    expected by the model to avoid shape/name mismatches.
    """

    def __init__(self, artifact_dir: str | Path = "results/meta_export"):
        self.dir = Path(artifact_dir)
        self.model = None
        self.ohe = None
        self.calibrator = None
        self.pstar: Optional[float] = None
        self.expected_features: List[str] = []
        self.raw_cat_cols: List[str] = []

        self.is_loaded = False
        try:
            self._load()
            self.is_loaded = True
        except Exception as e:
            print(f"[WinProbScorer] load failed from {self.dir}: {e}")
            self.is_loaded = False

    # -------------------- public API --------------------

    def score(self, row: Dict[str, Any]) -> float:
        """
        Score a single row (dict of features). 'row' may contain raw cats
        (e.g., 'entry_rule', 'pullback_type', 'regime_1d'); this scorer
        will OHE them using ohe.joblib and assemble the exact feature order.
        Returns probability in [0,1]. If not loaded, returns 0.0.
        """
        if not self.is_loaded or self.model is None:
            return 0.0

        # Build a single-row DataFrame X with columns in self.expected_features
        X = self._build_X(row)

        # LightGBM sklearn wrapper prefers DataFrame with names; this avoids warnings
        proba = float(self.model.predict_proba(X)[:, 1][0])

        # optional calibrator (both IsotonicRegression and Platt LR expose .predict)
        if self.calibrator is not None:
            try:
                proba = float(self.calibrator.predict([proba])[0])
            except Exception:
                # Some calibrators expect 2D; try that as fallback
                proba = float(self.calibrator.predict(np.array([[proba]]))[0])

        # bound defensively
        if not np.isfinite(proba):
            proba = 0.0
        return float(max(0.0, min(1.0, proba)))

    # -------------------- internals --------------------

    def _load(self):
        if not self.dir.exists():
            raise FileNotFoundError(self.dir)

        # 1) core model
        self.model = joblib.load(self.dir / "donch_meta_lgbm.joblib")

        # 2) ohe (optional but strongly recommended)
        ohe_path = self.dir / "ohe.joblib"
        if ohe_path.exists():
            self.ohe = joblib.load(ohe_path)
            # raw categorical columns used at train time
            try:
                self.raw_cat_cols = list(getattr(self.ohe, "feature_names_in_", []))
            except Exception:
                self.raw_cat_cols = []
        else:
            self.ohe = None
            self.raw_cat_cols = []

        # 3) calibrator (optional)
        cal_path = self.dir / "calibrator.joblib"
        if cal_path.exists():
            try:
                self.calibrator = joblib.load(cal_path)
            except Exception:
                self.calibrator = None

        # 4) threshold (optional)
        pstar_path = self.dir / "pstar.txt"
        if pstar_path.exists():
            try:
                self.pstar = float(pstar_path.read_text().strip())
            except Exception:
                self.pstar = None

        # 5) expected (input) feature names — FIRST try JSON, then model's booster
        feat_json = self.dir / "feature_names.json"
        names_from_json: List[str] = []
        if feat_json.exists():
            try:
                names_from_json = list(json.loads(feat_json.read_text()))
            except Exception:
                names_from_json = []

        names_from_model = self._model_feature_names()

        # sanity reconcile: prefer JSON if it matches model.n_features_in_
        n_model = self._model_n_features()
        if names_from_json and len(names_from_json) == n_model:
            self.expected_features = names_from_json
        else:
            self.expected_features = names_from_model or names_from_json
            if self.expected_features and len(self.expected_features) != n_model:
                print(
                    f"[WinProbScorer] feature_names mismatch: JSON={len(names_from_json)} "
                    f"model={n_model}. Using model feature names."
                )
                self.expected_features = names_from_model
            if not self.expected_features:
                # last resort: generate generic names
                self.expected_features = [f"f{i}" for i in range(n_model)]

    def _model_feature_names(self) -> List[str]:
        # LightGBM sklearn wrapper stores names in booster_ or sometimes .feature_name_
        try:
            if hasattr(self.model, "feature_name_") and self.model.feature_name_:
                return list(self.model.feature_name_)
        except Exception:
            pass
        try:
            if hasattr(self.model, "booster_"):
                f = self.model.booster_.feature_name()
                if f:
                    return list(f)
        except Exception:
            pass
        # If the model was trained on a numpy array, names may be absent
        return []

    def _model_n_features(self) -> int:
        try:
            return int(getattr(self.model, "n_features_in_", len(self._model_feature_names())))
        except Exception:
            return len(self._model_feature_names())

    def _build_X(self, row: Dict[str, Any]) -> pd.DataFrame:
        """
        Build a 1xN DataFrame with columns in EXACT order self.expected_features.
        Steps:
          1) Make a one-row DataFrame from 'row'
          2) Expand known categoricals via ohe.joblib -> dense array with OHE names
          3) Fill numeric features -> 0.0 if missing
          4) Combine into target DataFrame with expected column order; fill absent cols with 0.0
        """
        # base one-row frame
        base = pd.DataFrame([row])

        # OHE expansion (if provided)
        ohe_cols_out: List[str] = []
        ohe_vals: Optional[np.ndarray] = None
        if self.ohe is not None and self.raw_cat_cols:
            # ensure all raw cat columns exist as strings
            for c in self.raw_cat_cols:
                if c not in base.columns:
                    base[c] = ""
            try:
                Xo = self.ohe.transform(base[self.raw_cat_cols].astype(str))
                if hasattr(Xo, "toarray"):
                    Xo = Xo.toarray()
                ohe_vals = np.asarray(Xo, dtype=float)  # shape (1, n_ohe)
                ohe_cols_out = list(self.ohe.get_feature_names_out(self.raw_cat_cols))
            except Exception:
                # if transform fails, fall back to empty OHE
                ohe_cols_out = []
                ohe_vals = None

        # Prepare empty X with expected columns
        cols = list(self.expected_features)
        X = pd.DataFrame(data=np.zeros((1, len(cols)), dtype=float), columns=cols)

        # Fill OHE outputs where names match expected columns
        if ohe_vals is not None and ohe_cols_out:
            # intersect in order of expected cols
            common = [c for c in ohe_cols_out if c in X.columns]
            if common:
                # map positions from ohe to expected
                idx_in_ohe = [ohe_cols_out.index(c) for c in common]
                X.loc[0, common] = ohe_vals[0, idx_in_ohe]

        # Fill numeric columns: any expected column not in ohe_cols_out
        for c in X.columns:
            if c in ohe_cols_out:
                continue
            # treat as numeric; pick from 'row' if present, else 0
            v = row.get(c, 0.0)
            try:
                X.at[0, c] = float(v) if v is not None and np.isfinite(v) else 0.0
            except Exception:
                X.at[0, c] = 0.0

        # Final defensive cast
        return X.astype(float)
