# live/winprob_loader.py
from __future__ import annotations

import json, os, re, hashlib, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

LOG = logging.getLogger("winprob")


def _load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find(dir_: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = dir_ / n
        if p.exists():
            return p
    return None


class WinProbScorer:
    """
    Loads win-probability artifacts and scores a single meta-row (dict).
    Supports your export names:
      - donch_meta_lgbm.joblib
      - feature_names.json
      - ohe.joblib (optional)
      - calibrator.joblib (optional)
      - pstar.txt (optional; convenience only)
    Also tolerates older names (model.pkl, expected_features.json, etc.).
    """

    KEY_ALIASES = {
        "symbol": "sym",
        "pair": "sym",
        "ticker": "sym",
        "base": "sym",
        "market": "sym",
        "instrument": "sym",
    }

    def __init__(self, artifacts_dir: Optional[str | Path] = None) -> None:
        self.dir: Optional[Path] = None
        self.model = None
        self.calibrator = None
        self.ohe = None
        self.expected_features: List[str] = []
        self.pstar: Optional[float] = None

        # diagnostics
        self._diag_once = False
        self._last_hash = None
        self._same_vec_count = 0

        # schema inferred from expected_features
        self._ohe_cols: List[str] = []
        self._cat_raw_keys: List[str] = []
        self._num_cols: List[str] = []

        # Resolve directory: ENV → common defaults
        if artifacts_dir is None:
            env_dir = os.getenv("DONCH_WINPROB_DIR") or os.getenv("WINPROB_DIR")
            if env_dir:
                artifacts_dir = Path(env_dir)
            else:
                for cand in ("results/meta_export", "results/meta", "results/meta-model"):
                    if Path(cand).exists():
                        artifacts_dir = Path(cand)
                        break

        if artifacts_dir is not None:
            self.load(artifacts_dir)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and bool(self.expected_features)

    # ───────────────────────── load ─────────────────────────
    def load(self, artifacts_dir: str | Path) -> None:
        self.dir = Path(artifacts_dir)
        try:
            # 1) Model
            model_path = _find(self.dir, [
                "donch_meta_lgbm.joblib", "model.joblib", "model.pkl", "clf.pkl", "estimator.pkl"
            ])
            if model_path is None:
                raise FileNotFoundError("model artifact not found (donch_meta_lgbm.joblib/model.pkl)")
            self.model = joblib.load(model_path)

            # 2) Features
            ef_path = _find(self.dir, [
                "feature_names.json", "expected_features.json", "feature_order.json", "columns.json"
            ])
            if ef_path is not None:
                self.expected_features = list(_load_json(ef_path))
            else:
                # last-ditch: try introspecting
                if hasattr(self.model, "feature_names_in_"):
                    self.expected_features = list(self.model.feature_names_in_)
                elif hasattr(self.model, "feature_names_"):
                    self.expected_features = list(self.model.feature_names_)
                else:
                    raise FileNotFoundError("feature_names.json / expected_features.json not found")

            # 3) Optional artifacts
            ohe_path = _find(self.dir, ["ohe.joblib", "ohe.pkl", "onehot.joblib"])
            self.ohe = joblib.load(ohe_path) if ohe_path else None

            calib_path = _find(self.dir, ["calibrator.joblib", "calibrator.pkl", "calib.pkl", "calibration.pkl"])
            self.calibrator = joblib.load(calib_path) if calib_path else None

            pstar_path = self.dir / "pstar.txt"
            if pstar_path.exists():
                try:
                    self.pstar = float(pstar_path.read_text().strip())
                except Exception:
                    self.pstar = None

            self._infer_schema_from_expected()

            LOG.info("[WINPROB] loaded dir=%s  model=%s  features=%d  ohe=%s  calibrator=%s  pstar=%s",
                     str(self.dir),
                     model_path.name,
                     len(self.expected_features),
                     getattr(ohe_path, "name", "none") if ohe_path else "none",
                     getattr(calib_path, "name", "none") if calib_path else "none",
                     f"{self.pstar:.2f}" if isinstance(self.pstar, float) else "none")

        except Exception as e:
            LOG.exception("[WinProbScorer] load failed from %s: %s", self.dir, e)
            self.model = None
            self.expected_features = []
            self.ohe = None
            self.calibrator = None

    def _infer_schema_from_expected(self) -> None:
        ohe_cols, cat_keys, num_cols = [], set(), []
        for col in self.expected_features:
            if "=" in col or col.startswith(("cat__", "ohe__")):
                ohe_cols.append(col)
                left = col.split("=")[0]
                rk = left.split("__")[-1] if "__" in left else left
                cat_keys.add(rk)
            else:
                num_cols.append(col)
        self._ohe_cols = ohe_cols
        self._cat_raw_keys = list(cat_keys)
        self._num_cols = num_cols

    # ───────────────────────── build X ─────────────────────────
    def _normalize_keys(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in row.items():
            k2 = self.KEY_ALIASES.get(str(k).lower(), str(k).lower())
            out[k2] = v
        return out

    def _parse_ohe_target(self, colname: str) -> Optional[Tuple[str, str]]:
        # "cat__sym=ETHUSDT" | "ohe__sym=BTCUSDT" | "sym=SOLUSDT" | "cat__sym_ETHUSDT"
        if "=" in colname:
            left, right = colname.split("=", 1)
            raw_key = left.split("__")[-1] if "__" in left else left
            return raw_key, right
        m = re.match(r"^(?:cat__|ohe__)?([^_]+)_(.+)$", colname)
        if m:
            return m.group(1), m.group(2)
        return None

    def _cat_value(self, row: Dict[str, Any], raw_key: str) -> Any:
        if raw_key in row:
            return row[raw_key]
        if raw_key == "sym":
            for alt in ("symbol", "pair", "ticker", "base", "market", "instrument"):
                if alt in row:
                    return row[alt]
        return None

    def _manual_ohe(self, row_norm: Dict[str, Any]) -> pd.DataFrame:
        data = {}
        for col in self._ohe_cols:
            tgt = self._parse_ohe_target(col)
            if not tgt:
                data[col] = 0.0
                continue
            rk, want = tgt
            have = self._cat_value(row_norm, rk)
            data[col] = 1.0 if have is not None and str(have) == str(want) else 0.0
        return pd.DataFrame([data], columns=self._ohe_cols)

    def _ohe_transform(self, row_norm: Dict[str, Any]) -> pd.DataFrame:
        if not self._ohe_cols:
            return pd.DataFrame([{}])
        if self.ohe is None:
            return self._manual_ohe(row_norm)

        cat_in = list(getattr(self.ohe, "feature_names_in_", [])) or self._cat_raw_keys
        raw = {k: self._cat_value(row_norm, k) for k in cat_in}
        Xcat = pd.DataFrame([raw], columns=cat_in)
        try:
            arr = self.ohe.transform(Xcat)
            arr = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
            names = (list(self.ohe.get_feature_names_out(cat_in))
                     if hasattr(self.ohe, "get_feature_names_out") else self._ohe_cols)
            df = pd.DataFrame(arr, columns=names)
        except Exception:
            df = self._manual_ohe(row_norm)

        # align to expected OHE outputs
        if self._ohe_cols:
            for c in self._ohe_cols:
                if c not in df.columns:
                    df[c] = 0.0
            drop = [c for c in df.columns if c not in self._ohe_cols]
            if drop:
                df.drop(columns=drop, inplace=True)
            df = df[self._ohe_cols]
        return df

    def _build_X(self, row: Dict[str, Any]) -> pd.DataFrame:
        if not self.expected_features:
            raise RuntimeError("WinProbScorer not loaded")
        r = self._normalize_keys(row)
        df_cat = self._ohe_transform(r)

        num_vals: Dict[str, float] = {}
        for c in self._num_cols:
            raw_key = c.split("__")[-1] if "__" in c else c
            v = r.get(c, r.get(raw_key, 0.0))
            try:
                num_vals[c] = float(v) if v is not None and np.isfinite(v) else 0.0
            except Exception:
                num_vals[c] = 0.0
        df_num = pd.DataFrame([num_vals], columns=self._num_cols) if self._num_cols else pd.DataFrame([{}])

        X = pd.concat([df_cat, df_num], axis=1)
        for c in self.expected_features:
            if c not in X.columns:
                X[c] = 0.0
        return X[self.expected_features].astype(float)

    # ───────────────────────── scoring ─────────────────────────
    def _calibrate(self, p_raw: float) -> float:
        if self.calibrator is None:
            return float(np.clip(p_raw, 0.0, 1.0))
        try:
            if hasattr(self.calibrator, "predict") and not hasattr(self.calibrator, "predict_proba"):
                out = self.calibrator.predict(np.array([[p_raw]]))
                return float(np.clip(out[0], 0.0, 1.0))
            if hasattr(self.calibrator, "predict_proba"):
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
            nz = int((np.abs(X.to_numpy()) > 0).sum())
            LOG.info("[WINPROB DIAG] features=%d  nonzero=%d  first_nonzero=%s",
                     len(self.expected_features), nz,
                     [c for c, v in zip(X.columns, X.to_numpy()[0]) if v != 0][:15])
            self._diag_once = True

        p_raw = float(self.model.predict_proba(X)[:, 1][0])
        p = self._calibrate(p_raw)

        # identical vector detector (helps catch mapping bugs)
        try:
            h = hashlib.md5(X.to_numpy().tobytes()).hexdigest()
            if self._last_hash == h:
                self._same_vec_count += 1
                if self._same_vec_count in (2, 5, 25, 100):
                    LOG.warning("[WINPROB DIAG] %d identical vectors in a row (hash=%s).", self._same_vec_count, h)
            else:
                self._last_hash = h
                self._same_vec_count = 0
        except Exception:
            pass

        return float(np.clip(p, 0.0, 1.0))

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
