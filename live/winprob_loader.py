# winprob_loader.py
from __future__ import annotations
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

class WinProbScorer:
    """
    Loads:
      - donch_meta_lgbm.joblib  (sklearn LGBMClassifier)
      - ohe.joblib               (sklearn OneHotEncoder for raw string cats)
      - feature_names.json       (FINAL column order: numerics + one-hot names)
      - calibrator.joblib        (optional Platt/Isotonic; .predict(prob) → prob)
      - pstar.txt                (optional threshold, read by the bot separately)
    Usage:
      s = WinProbScorer("models/donch_meta"); s.score(raw_feature_dict)
    """
    def __init__(self, model_dir: str = "models/donch_meta"):
        self.dir = Path(model_dir)
        self.is_loaded = False
        self.kind = "lgbm+ohe"
        self.model = None
        self.ohe = None
        self.calibrator = None
        self.expected_features: list[str] = []  # final training order (numeric + one-hot)

        try:
            self.model = joblib.load(self.dir / "donch_meta_lgbm.joblib")
            # optional encoder
            ohe_path = self.dir / "ohe.joblib"
            if ohe_path.exists():
                self.ohe = joblib.load(ohe_path)
            # expected final columns
            self.expected_features = json.loads((self.dir / "feature_names.json").read_text())
            # optional calibrator
            cal = self.dir / "calibrator.joblib"
            if cal.exists():
                self.calibrator = joblib.load(cal)
            self.is_loaded = True
        except Exception as e:
            # leave is_loaded False; caller will degrade gracefully
            print(f"[WinProbScorer] load failed from {self.dir}: {e}")

    def _build_vector(self, raw: Dict[str, Any]) -> np.ndarray:
        """
        Accepts a dict of raw features. It can include:
          - numerics directly named as in training
          - raw categorical strings matching the OHE's feature_names_in_
        We assemble a single 1×N row in EXACT training order self.expected_features.
        Missing values → 0.0; missing/unknown categories → all-zeros one-hot.
        """
        # 1) Prepare a one-row DataFrame with everything we were given
        df = pd.DataFrame([raw])
        # 2) If we have an OHE, transform the raw categorical inputs
        ohe_out = None
        ohe_cols = []
        if self.ohe is not None:
            cat_in = list(getattr(self.ohe, "feature_names_in_", []))
            # ensure the categorical columns exist (string type)
            for c in cat_in:
                if c not in df.columns:
                    df[c] = ""  # safe fallback cat
            Xo = self.ohe.transform(df[cat_in].astype(str))  # (1, k)
            ohe_cols = list(self.ohe.get_feature_names_out(cat_in))
            # convert sparse to dense if needed
            ohe_out = np.asarray(Xo.todense() if hasattr(Xo, "todense") else Xo, dtype=np.float64)

        # 3) Build a dict of all columns we can populate now
        row_vals: Dict[str, float] = {}
        # numeric features are "expected_features minus OHE-output names"
        numeric_expected = [c for c in self.expected_features if c not in set(ohe_cols)]
        for c in numeric_expected:
            v = df[c].iloc[0] if c in df.columns else 0.0
            try:
                row_vals[c] = float(v) if v is not None and np.isfinite(v) else 0.0
            except Exception:
                row_vals[c] = 0.0

        # 4) If we have OHE output columns, scatter them by name; else leave zeros
        ohe_map: Dict[str, float] = {}
        if ohe_out is not None:
            assert len(ohe_cols) == ohe_out.shape[1], "OHE shape mismatch"
            for j, name in enumerate(ohe_cols):
                ohe_map[name] = float(ohe_out[0, j])
        # 5) Assemble in the exact expected order
        out = []
        for name in self.expected_features:
            if name in ohe_map:
                out.append(ohe_map[name])
            else:
                out.append(row_vals.get(name, 0.0))
        X = np.asarray([out], dtype=np.float64)
        return X

    def score(self, raw: Dict[str, Any]) -> float:
        if not self.is_loaded:
            return 0.0
        X = self._build_vector(raw)
        try:
            p = float(self.model.predict_proba(X)[:, 1][0])
        except Exception:
            # try decision_function (rare)
            p = float(self.model.predict_proba(X)[:, 1][0])
        if self.calibrator is not None:
            try:
                # isotonic & Platt both expose predict()
                p = float(self.calibrator.predict([p])[0])
            except Exception:
                pass
        # clamp
        return max(0.0, min(1.0, p))
