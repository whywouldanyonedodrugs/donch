# live/winprob_loader.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import hashlib
import logging

LOG = logging.getLogger("winprob")


def _read_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_first(dir: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = dir / n
        if p.exists():
            return p
    return None


class WinProbScorer:
    """
    Robust loader + scorer for the research win-probability model.

    Artifacts directory is expected to contain at least:
      - expected_features.json  (list[str] in final model order)
      - model.pkl or model.joblib  (sklearn estimator with predict_proba)

    Optionally:
      - calibrator.pkl / calib.pkl / calibration.pkl (isotonic or Platt)
      - ohe.joblib / ohe.pkl / onehot.joblib (sklearn OneHotEncoder)
    """
    def __init__(self, artifacts_dir: Optional[str | Path] = None) -> None:
        self.dir: Optional[Path] = None
        self.model = None
        self.calibrator = None
        self.ohe = None
        self.expected_features: List[str] = []
        self._diag_once = False
        self._last_hash = None
        self._same_vec_count = 0

        # inferred schema
        self._ohe_cols: List[str] = []
        self._cat_raw_keys: List[str] = []
        self._num_cols: List[str] = []

        self.KEY_ALIASES = {
            "symbol": "sym",
            "pair": "sym",
            "ticker": "sym",
            "base": "sym",
            "market": "sym",
            "instrument": "sym",
        }

        if artifacts_dir is not None:
            self.load(artifacts_dir)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and bool(self.expected_features)

    # ── Loading ───────────────────────────────────────────────────────
    def load(self, artifacts_dir: str | Path) -> None:
        self.dir = Path(artifacts_dir)
        try:
            ef_path = _find_first(self.dir, ["expected_features.json", "feature_order.json", "columns.json"])
            if ef_path is None:
                raise FileNotFoundError("expected_features.json not found")
            self.expected_features = list(_read_json_any(ef_path))

            model_path = _find_first(self.dir, ["model.pkl", "model.joblib", "clf.pkl", "estimator.pkl"])
            if model_path is None:
                raise FileNotFoundError("model.pkl/joblib not found")
            self.model = joblib.load(model_path)

            calib_path = _find_first(self.dir, ["calibrator.pkl", "calib.pkl", "calibration.pkl"])
            self.calibrator = joblib.load(calib_path) if calib_path else None

            ohe_path = _find_first(self.dir, ["ohe.joblib", "ohe.pkl", "onehot.joblib"])
            self.ohe = joblib.load(ohe_path) if ohe_path else None

            self._infer_schema_from_expected()

            LOG.info("[WINPROB] loaded model=%s  calibrator=%s  ohe=%s  features=%d",
                     model_path.name,
                     calib_path.name if calib_path else "none",
                     getattr(ohe_path, "name", "none") if ohe_path else "none",
                     len(self.expected_features))
        except Exception as e:
            LOG.exception("[WinProbScorer] load failed from %s: %s", self.dir, e)
            self.model = None
            self.expected_features = []

    def _infer_schema_from_expected(self) -> None:
        """Infer OHE output cols, raw categorical keys, and numeric cols from expected feature names."""
        ohe_cols, cat_keys, num_cols = [], set(), []
        for col in self.expected_features:
            if "=" in col or col.startswith(("cat__", "ohe__")):
                ohe_cols.append(col)
                left = col.split("=")[0]
                rk = left.split("__")[-1] if "__" in left else left
                cat_keys.add(rk)
            else:
                num_cols.append(col)
        self._ohe_cols = list(ohe_cols)
        self._cat_raw_keys = list(cat_keys)
        self._num_cols = list(num_cols)

    # ── Builders ──────────────────────────────────────────────────────
    def _normalize_row_keys(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in row.items():
            lk = str(k).strip().lower()
            lk = self.KEY_ALIASES.get(lk, lk)
            out[lk] = v
        return out

    def _cat_value_from_row(self, row: Dict[str, Any], raw_key: str) -> Any:
        v = row.get(raw_key, None)
        if v is None and raw_key == "sym":
            for alt in ("symbol", "pair", "ticker", "base", "market", "instrument"):
                vv = row.get(alt)
                if vv is not None:
                    v = vv
                    break
        return v

    def _parse_ohe_target(self, colname: str) -> Optional[Tuple[str, str]]:
        # "cat__sym=ETHUSDT" | "ohe__sym=BTCUSDT" | "sym=SOLUSDT"
        if "=" in colname:
            left, right = colname.split("=", 1)
            raw_key = left.split("__")[-1] if "__" in left else left
            return raw_key, right
        # Fallback: "cat__sym_ETHUSDT"
        m = re.match(r"^(?:cat__|ohe__)?([^_]+)_(.+)$", colname)
        if m:
            return m.group(1), m.group(2)
        return None

    def _manual_ohe(self, row_norm: Dict[str, Any]) -> pd.DataFrame:
        data = {}
        for col in self._ohe_cols:
            tgt = self._parse_ohe_target(col)
            if not tgt:
                data[col] = 0.0
                continue
            raw_key, want_val = tgt
            have_val = self._cat_value_from_row(row_norm, raw_key)
            data[col] = 1.0 if have_val is not None and str(have_val) == str(want_val) else 0.0
        return pd.DataFrame([data], columns=self._ohe_cols)

    def _ohe_transform(self, row_norm: Dict[str, Any]) -> pd.DataFrame:
        if self.ohe is None:
            return self._manual_ohe(row_norm)

        cat_in = list(getattr(self.ohe, "feature_names_in_", [])) or self._cat_raw_keys
        raw = {k: self._cat_value_from_row(row_norm, k) for k in cat_in}
        X_cat = pd.DataFrame([raw], columns=cat_in)
        try:
            arr = self.ohe.transform(X_cat)
            arr = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
            names = (list(self.ohe.get_feature_names_out(cat_in))
                     if hasattr(self.ohe, "get_feature_names_out") else self._ohe_cols)
            df = pd.DataFrame(arr, columns=names)
        except Exception as e:
            LOG.debug("[WINPROB] ohe.transform failed (%s); falling back to manual OHE.", e)
            df = self._manual_ohe(row_norm)

        # align to expected
        if self._ohe_cols:
            drop = [c for c in df.columns if c not in self._ohe_cols]
            if drop:
                df.drop(columns=drop, inplace=True)
            for c in self._ohe_cols:
                if c not in df.columns:
                    df[c] = 0.0
            df = df[self._ohe_cols]
        return df

    def _build_X(self, row: Dict[str, Any]) -> pd.DataFrame:
        if not self.expected_features:
            raise RuntimeError("WinProbScorer not loaded")

        row_norm = self._normalize_row_keys(row)

        # 1) Categorical block
        df_cat = self._ohe_transform(row_norm) if self._ohe_cols else pd.DataFrame([{}])

        # 2) Numeric block
        num_vals: Dict[str, float] = {}
        for c in self._num_cols:
            raw_key = c.split("__")[-1] if "__" in c else c
            v = row_norm.get(c, None)
            if v is None:
                v = row_norm.get(raw_key, None)
            try:
                num_vals[c] = float(v) if v is not None and np.isfinite(v) else 0.0
            except Exception:
                num_vals[c] = 0.0
        df_num = pd.DataFrame([num_vals], columns=self._num_cols) if self._num_cols else pd.DataFrame([{}])

        # 3) Concatenate and enforce final order
        X = pd.concat([df_cat, df_num], axis=1)
        for c in self.expected_features:
            if c not in X.columns:
                X[c] = 0.0
        X = X[self.expected_features].astype(float)
        return X

    # ── Scoring ───────────────────────────────────────────────────────
    def _calibrate(self, p_raw: float) -> float:
        if self.calibrator is None:
            return p_raw
        try:
            if hasattr(self.calibrator, "predict") and not hasattr(self.calibrator, "predict_proba"):
                out = self.calibrator.predict(np.array([[p_raw]]))
                return float(np.clip(out[0], 0.0, 1.0))
            elif hasattr(self.calibrator, "predict_proba"):
                out = self.calibrator.predict_proba(np.array([[p_raw]]))[:, 1]
                return float(np.clip(out[0], 0.0, 1.0))
        except Exception as e:
            LOG.debug("[WINPROB] calibrator failed: %s", e)
        return float(np.clip(p_raw, 0.0, 1.0))

    def score(self, row: Dict[str, Any]) -> float:
        if not self.is_loaded:
            return 0.0

        X = self._build_X(row)

        # one-shot diag
        if not self._diag_once:
            present = [c for c in self.expected_features if c in X.columns]
            missing = [c for c in self.expected_features if c not in X.columns]
            nz = int((np.abs(X.to_numpy()) > 0).sum())
            LOG.info("[WINPROB DIAG] features=%d  present=%d  missing=%d  nonzero=%d",
                     len(self.expected_features), len(present), len(missing), nz)
            if missing[:30]:
                LOG.warning("[WINPROB DIAG] Missing (first 30): %s", missing[:30])
            nz_idx = np.where(np.abs(X.to_numpy())[0] > 0)[0].tolist()
            nz_names = [self.expected_features[i] for i in nz_idx][:15]
            LOG.info("[WINPROB DIAG] first_nonzero: %s", nz_names)
            self._diag_once = True

        p_raw = float(self.model.predict_proba(X)[:, 1][0])
        p = self._calibrate(p_raw)

        # identical vector detector
        try:
            vec_bytes = X.to_numpy().tobytes()
            h = hashlib.md5(vec_bytes).hexdigest()
            if self._last_hash == h:
                self._same_vec_count += 1
                if self._same_vec_count in (2, 5, 25, 100):
                    LOG.warning("[WINPROB DIAG] %d identical vectors in a row (hash=%s).", self._same_vec_count, h)
            else:
                self._last_hash = h
                self._same_vec_count = 0
        except Exception:
            pass

        return float(max(0.0, min(1.0, p)))

    def score_df(self, X: pd.DataFrame) -> float:
        if not self.is_loaded:
            return 0.0
        if list(X.columns) != list(self.expected_features):
            for c in self.expected_features:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[self.expected_features]
        p_raw = float(self.model.predict_proba(X)[:, 1][0])
        return self._calibrate(p_raw)
