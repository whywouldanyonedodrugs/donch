# live/winprob_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

import hashlib
import logging
LOG = logging.getLogger("winprob")

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

        self._diag_once = False
        self._last_hash = None
        self._same_vec_count = 0

        def _calibrate(self, p_raw: float) -> float:
            if self.calibrator is None:
                return p_raw
            # Try isotonic first (has .predict), else Platt (.predict_proba)
            if hasattr(self.calibrator, "predict") and not hasattr(self.calibrator, "predict_proba"):
                # isotonic regression returns calibrated probas
                return float(self.calibrator.predict([p_raw])[0])
            elif hasattr(self.calibrator, "predict_proba"):
                # logistic regression: need predict_proba on [[p_raw]]
                import numpy as _np
                return float(self.calibrator.predict_proba(_np.array([[p_raw]]))[:,1][0])
            else:
                return p_raw  # unknown calibrator type

    # -------------------- public API --------------------

    def score(self, row: dict) -> float:
        # 1) Build X exactly as used today
        X, used_cols = self._build_X(row)   # your existing method that returns ndarray (1, n_features)

        # 2) Model raw proba
        p_raw = float(self.model.predict_proba(X)[:, 1][0])
        p = self._calibrate(p_raw)

        # 3) One-shot deep diag
        if not self._diag_once:
            # What did we *think* the categorical inputs are?
            cat_in = list(getattr(self.ohe, "feature_names_in_", [])) if getattr(self, "ohe", None) is not None else []
            ohe_out = list(self.ohe.get_feature_names_out(cat_in)) if getattr(self, "ohe", None) is not None else []

            # Feature coverage:
            missing = [f for f in self.feature_order if f not in used_cols]
            present = [f for f in self.feature_order if f in used_cols]

            # Nonzero snapshot (which columns are actually influencing)
            nz_idx = list(np.where(X[0] != 0.0)[0])
            nz_names = [self.feature_order[i] for i in nz_idx][:20]  # first 20 for brevity

            LOG.warning(
                "[WINPROB DIAG]\n"
                "  n_features: %d\n"
                "  raw_proba: %.6f  calibrated: %.6f\n"
                "  cat_inputs_expected: %s\n"
                "  ohe_outputs_seen: %d\n"
                "  present_cols: %d  missing_cols: %d\n"
                "  first_20_nonzero: %s\n",
                len(self.feature_order), p_raw, p,
                cat_in, len(ohe_out),
                len(present), len(missing),
                nz_names
            )
            if missing:
                LOG.warning("[WINPROB DIAG] Missing features (first 30): %s", missing[:30])
            self._diag_once = True

        # 4) Vector hash: are many calls identical?
        h = hashlib.md5(X.tobytes()).hexdigest()
        if self._last_hash is None:
            self._last_hash = h
        else:
            if h == self._last_hash:
                self._same_vec_count += 1
                if self._same_vec_count in (5, 25, 100):
                    LOG.warning("[WINPROB DIAG] Detected %d consecutive identical feature vectors (hash=%s).",
                                self._same_vec_count, h)
            else:
                self._last_hash = h
                self._same_vec_count = 0

        return p

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
