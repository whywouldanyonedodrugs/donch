# live/winprob_loader.py  — WinProbScorer patch (drop-in)
from __future__ import annotations
import json, os, re, hashlib, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib, numpy as np, pandas as pd

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
    """

    # key normalization (lower-cased)
    KEY_ALIASES = {
        "symbol": "sym", "pair": "sym", "ticker": "sym", "market": "sym", "instrument": "sym",
    }

    # categorical groups that were one-hot encoded at train time
    GROUP_PREFIXES = ("pullback_type_", "entry_rule_", "regime_1d_")  # e.g., pullback_type_retest

    # live->train numeric aliases (all lower-case)
    NUMERIC_ALIASES: Dict[str, List[str]] = {
        "rsi_1h": ["rsi1h", "rsi_1h"],
        "adx_1h": ["adx1h", "adx_1h"],
        "eth_macd_hist_4h": [
            "eth_macd_hist_4h", "eth_macd_hist4h", "eth_macd4h_hist",
            "eth_macd_hist", "eth macd4h hist", "eth macd(4h) hist"
        ],
        "vol_mult": ["vol_mult", "vol mult (median 30d)", "vol_mult_median_30d"],
        "don_break_level": ["don_break_level"],
        "don_break_len": ["don_break_len"],
        "don_dist_atr": ["don_dist_atr", "dist_atr"],
        "atr_pct": ["atr_pct", "atr%"],
        "atr_1h": ["atr_1h"],
        "atr": ["atr"],
        "entry": ["entry", "price"],  # your live “Price” is entry at signal time
        "rs_pct": ["rs_pct", "rs pct"],
        "hour_sin": ["hour_sin"], "hour_cos": ["hour_cos"], "dow": ["dow"],
        "vol_spike_i": ["vol_spike_i", "vol_spike"],
    }

    def __init__(self, artifacts_dir: Optional[str | Path] = None) -> None:
        self.dir: Optional[Path] = None
        self.model = None
        self.calibrator = None
        self.ohe = None
        self.expected_features: List[str] = []
        self.pstar: Optional[float] = None

        self._diag_once = False
        self._last_hash = None
        self._same_vec_count = 0

        self._ohe_cols: List[str] = []
        self._num_cols: List[str] = []

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

    def load(self, artifacts_dir: str | Path) -> None:
        self.dir = Path(artifacts_dir)
        try:
            model_path = _find(self.dir, [
                "donch_meta_lgbm.joblib", "model.joblib", "model.pkl", "clf.pkl", "estimator.pkl"
            ])
            if model_path is None:
                raise FileNotFoundError("model artifact not found (donch_meta_lgbm.joblib/model.pkl)")
            self.model = joblib.load(model_path)

            ef_path = _find(self.dir, [
                "feature_names.json", "expected_features.json", "feature_order.json", "columns.json"
            ])
            if ef_path is not None:
                self.expected_features = list(_load_json(ef_path))
            elif hasattr(self.model, "feature_names_in_"):
                self.expected_features = list(self.model.feature_names_in_)
            else:
                raise FileNotFoundError("feature_names.json / expected_features.json not found")

            ohe_path = _find(self.dir, ["ohe.joblib", "ohe.pkl", "onehot.joblib"])
            self.ohe = joblib.load(ohe_path) if ohe_path else None

            calib_path = _find(self.dir, ["calibrator.joblib", "calibrator.pkl", "calib.pkl", "calibration.pkl"])
            self.calibrator = joblib.load(calib_path) if calib_path else None

            pstar_path = self.dir / "pstar.txt"
            if pstar_path.exists():
                try: self.pstar = float(pstar_path.read_text().strip())
                except Exception: self.pstar = None

            self._infer_schema_from_expected()

            LOG.info("[WINPROB] loaded dir=%s model=%s features=%d ohe=%s calibrator=%s p*=%s",
                     str(self.dir), model_path.name, len(self.expected_features),
                     getattr(ohe_path, "name", "none") if ohe_path else "none",
                     getattr(calib_path, "name", "none") if calib_path else "none",
                     f"{self.pstar:.2f}" if isinstance(self.pstar, float) else "none")
        except Exception as e:
            LOG.exception("[WinProbScorer] load failed from %s: %s", self.dir, e)
            self.model = None; self.expected_features = []; self.ohe = None; self.calibrator = None

    def _infer_schema_from_expected(self) -> None:
        ohe_cols, num_cols = [], []
        for col in self.expected_features:
            if (
                "=" in col or col.startswith(("cat__", "ohe__")) or
                any(col.startswith(pref) for pref in self.GROUP_PREFIXES)
            ):
                ohe_cols.append(col)
            else:
                num_cols.append(col)
        self._ohe_cols = ohe_cols
        self._num_cols = num_cols

    # ----- helpers -----
    def _norm_key(self, k: str) -> str:
        k2 = self.KEY_ALIASES.get(str(k).lower(), str(k).lower())
        k2 = k2.replace("%", "pct")
        k2 = re.sub(r"[^\w]+", "_", k2).strip("_")
        return k2

    def _normalize_keys(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return { self._norm_key(k): v for k, v in row.items() }

    def _lookup_num(self, row: Dict[str, Any], col: str) -> float:
        # exact
        if col in row: return row[col]
        # aliases
        for al in self.NUMERIC_ALIASES.get(col, []):
            if al in row: return row[al]
        # derived fallbacks
        if col == "atr_pct" and "atr" in row and "entry" in row and row["entry"]:
            try: return float(row["atr"]) / float(row["entry"])
            except Exception: return 0.0
        if col == "don_dist_atr" and "entry" in row and "don_break_level" in row:
            atr_scale = row.get("atr_1h") or row.get("atr") or 0.0
            try:
                if float(atr_scale) != 0.0:
                    return (float(row["entry"]) - float(row["don_break_level"])) / float(atr_scale)
            except Exception:
                return 0.0
        return 0.0

    def _manual_ohe(self, row: Dict[str, Any]) -> pd.DataFrame:
        """Produce columns in self._ohe_cols, including group-style one-hots like 'pullback_type_retest'."""
        data: Dict[str, float] = {}
        for col in self._ohe_cols:
            v = 0.0
            # group-style: prefix_value
            for pref in self.GROUP_PREFIXES:
                if col.startswith(pref):
                    base = pref[:-1] if pref.endswith("_") else pref  # e.g., 'pullback_type'
                    want = col[len(pref):]
                    have = str(row.get(base, "")).strip()
                    v = 1.0 if have == want else 0.0
                    break
            data[col] = float(v)
        return pd.DataFrame([data], columns=self._ohe_cols) if self._ohe_cols else pd.DataFrame([{}])

    def _ohe_transform(self, row_norm: Dict[str, Any]) -> pd.DataFrame:
        # If an sklearn OneHotEncoder was exported, use it; otherwise do manual OHE.
        # (sklearn naming via get_feature_names_out is the train-time source of columns). :contentReference[oaicite:1]{index=1}
        if self.ohe is None:
            return self._manual_ohe(row_norm)

        cat_in = list(getattr(self.ohe, "feature_names_in_", []))
        if not cat_in:
            return self._manual_ohe(row_norm)

        raw = {k: row_norm.get(k) for k in cat_in}
        try:
            Xc = self.ohe.transform(pd.DataFrame([raw], columns=cat_in))
            Xc = Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc)
            names = list(self.ohe.get_feature_names_out(cat_in)) if hasattr(self.ohe, "get_feature_names_out") else []
            df = pd.DataFrame(Xc, columns=names) if names else pd.DataFrame(Xc)
        except Exception:
            df = self._manual_ohe(row_norm)

        # align to expected OHE outputs
        for c in self._ohe_cols:
            if c not in df.columns: df[c] = 0.0
        drop = [c for c in df.columns if c not in self._ohe_cols]
        if drop: df.drop(columns=drop, inplace=True)
        return df[self._ohe_cols] if self._ohe_cols else pd.DataFrame([{}])

    def _build_X(self, row: Dict[str, Any]) -> pd.DataFrame:
        if not self.expected_features:
            raise RuntimeError("WinProbScorer not loaded")
        r = self._normalize_keys(row)

        # categorical block (group OHE + optional sklearn OHE)
        df_cat = self._ohe_transform(r)

        # numeric block with alias lookup
        num_vals = { c: float(self._lookup_num(r, c)) if self._lookup_num(r, c) is not None and np.isfinite(self._lookup_num(r, c)) else 0.0
                     for c in self._num_cols }
        df_num = pd.DataFrame([num_vals], columns=self._num_cols) if self._num_cols else pd.DataFrame([{}])

        X = pd.concat([df_cat, df_num], axis=1)
        for c in self.expected_features:
            if c not in X.columns: X[c] = 0.0
        X = X[self.expected_features].astype(float)

        if not self._diag_once:
            nz = int((np.abs(X.to_numpy()) > 0).sum())
            first_nz = [c for c, v in zip(X.columns, X.to_numpy()[0]) if v != 0][:15]
            LOG.info("[WINPROB DIAG] features=%d  nonzero=%d  first_nonzero=%s", len(self.expected_features), nz, first_nz)
            self._diag_once = True
        return X

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
        if not self.is_loaded: return 0.0
        X = self._build_X(row)
        p_raw = float(self.model.predict_proba(X)[:, 1][0])
        p = self._calibrate(p_raw)

        # identical-vector guardrail
        try:
            h = hashlib.md5(X.to_numpy().tobytes()).hexdigest()
            if self._last_hash == h:
                self._same_vec_count += 1
                if self._same_vec_count in (2, 5, 25, 100):
                    LOG.warning("[WINPROB DIAG] %d identical feature vectors in a row (hash=%s).", self._same_vec_count, h)
            else:
                self._last_hash, self._same_vec_count = h, 0
        except Exception:
            pass

        # 1) log p_raw and calibrated p
        p_raw = float(self.model.predict_proba(X)[:, 1][0])
        LOG.info("[WINPROB] p_raw=%.6f  p_cal=%.6f", p_raw, self._calibrate(p_raw))

        # 2) LightGBM contributions to confirm the model is actually using features
#        try:
#            contrib = self.model.predict(X, pred_contrib=True)  # last col is base value
#            vals = contrib[0]
#            names = list(self.expected_features) + ["<base>"]
#            top = sorted(zip(names, vals), key=lambda z: abs(z[1]), reverse=True)[:6]
#            LOG.info("[WINPROB] top_contrib: %s", top)
#        except Exception as e:
#            LOG.debug("pred_contrib failed: %s", e)

        return float(np.clip(p, 0.0, 1.0))

    def score_df(self, X: pd.DataFrame) -> float:
        if not self.is_loaded: return 0.0
        for c in self.expected_features:
            if c not in X.columns: X[c] = 0.0
        X = X[self.expected_features].astype(float)
        p_raw = float(self.model.predict_proba(X)[:, 1][0])
        return self._calibrate(p_raw)
