# live/winprob_loader.py
from __future__ import annotations

import json, os, re, hashlib, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from difflib import get_close_matches

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


def _normkey(s: str) -> str:
    """Normalize key for fuzzy matching: lowercase, strip non-alnum."""
    return re.sub(r"[^0-9a-z]+", "", str(s).lower())


class WinProbScorer:
    """
    Loads win-probability artifacts and scores a single meta-row (dict).
    Supports your export names:
      - donch_meta_lgbm.joblib
      - feature_names.json
      - ohe.joblib (optional)
      - calibrator.joblib (optional)
      - pstar.txt (optional)
    Also tolerates older names (model.pkl, expected_features.json, etc.).
    """

    # Aliases for raw keys that often vary at runtime
    KEY_ALIASES = {
        "symbol": "sym",
        "pair": "sym",
        "ticker": "sym",
        "base": "sym",
        "market": "sym",
        "instrument": "sym",
    }

    # Numeric feature name -> likely live variants (case/spacing differ)
    NUMERIC_ALIASES = {
        "entry": ["entry", "entry_score", "entry_conf", "entry_strength"],
        "atr": ["atr", "ATR", "atr_1d", "ATR1d"],
        "atr_1h": ["atr1h", "ATR1h", "atr_h1", "atr60m"],
        "atr_pct": ["atr_pct", "ATR_pct", "atr_percent"],
        "don_break_len": ["don_break_len", "donch_break_len", "break_len", "break_length"],
        "don_break_level": ["don_break_level", "donch_break_level", "break_level"],
        "don_dist_atr": ["dist_atr", "donch_dist_atr", "don_upper_dist_atr", "dist_from_donch_upper_atr"],
        "rs_pct": ["rs_pct", "RS_pct", "rspercent", "rs%"],
        "hour_sin": ["hour_sin", "hour_of_day_sin", "hod_sin"],
        "hour_cos": ["hour_cos", "hour_of_day_cos", "hod_cos"],
        "dow": ["dow", "day_of_week", "weekday_idx", "wd"],
        "rsi_1h": ["rsi1h", "RSI1h", "rsi_h1"],
        "adx_1h": ["adx1h", "ADX1h", "adx_h1"],
        "vol_mult": ["vol_mult", "vol_mult_30d", "vol_mult_med_30d", "vol_mult_median_30d", "volume_mult", "Vol mult (median 30d)"],
        "eth_macd_hist_4h": ["eth_macd_hist_4h", "ETH_MACD_hist_4h", "eth_macd_hist", "ETH MACD(4h) hist"],
        "days_since_prev_break": ["days_since_prev_break", "days_since_break"],
        "consolidation_range_atr": ["consolidation_range_atr", "consol_range_atr"],
        "prior_1d_ret": ["prior_1d_ret", "ret_1d_prior", "ret_prev_1d"],
        "rv_3d": ["rv_3d", "realized_vol_3d"],
        "markov_state_4h": ["markov_state_4h", "ms_4h"],
        "markov_state_up_4h": ["markov_state_up_4h", "ms_up_4h"],
        "markov_prob_up_4h": ["markov_prob_up_4h", "mp_up_4h"],
        "vol_prob_low_1d": ["vol_prob_low_1d", "prob_vol_low_1d"],
        "regime_code_1d": ["regime_code_1d", "regime_code"],
    }

    # One-hot groups as they appear in feature_names.json
    # Columns like "pullback_type_retest", "entry_rule_close_above_break", "regime_1d_NA_LOW_VOL"
    GROUP_OHE_PREFIXES: Dict[str, List[str]] = {
        "pullback_type_": ["pullback_type"],
        "entry_rule_": ["entry_rule"],
        "regime_1d_": ["regime_1d", "regime_1d_name", "regime"],
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
        self._ohe_cols: List[str] = []          # scikit-OHE style (not used by your features, but supported)
        self._cat_raw_keys: List[str] = []
        self._num_cols: List[str] = []          # numeric features only
        self._grp_ohe_cols: Dict[str, List[str]] = {}  # prefix -> list of columns

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

            # 2) Features: prefer LightGBM's own names (training truth) over JSON
            json_feats = None
            ef_path = _find(self.dir, [
                "feature_names.json", "expected_features.json", "feature_order.json", "columns.json"
            ])
            if ef_path is not None:
                try:
                    json_feats = list(_load_json(ef_path))
                except Exception:
                    json_feats = None

            booster_feats = None
            try:
                booster = getattr(self.model, "booster_", None)
                if booster is not None and hasattr(booster, "feature_name"):
                    booster_feats = list(booster.feature_name())
            except Exception:
                booster_feats = None

            if booster_feats and len(booster_feats) > 0:
                self.expected_features = booster_feats
                if json_feats and len(json_feats) != len(booster_feats):
                    LOG.warning("[WINPROB] feature_names.json len=%d, but booster expects len=%d. "
                                "Using booster names.", len(json_feats), len(booster_feats))
            elif json_feats:
                self.expected_features = json_feats
            else:
                # last ditch: sklearn-style attributes
                if hasattr(self.model, "feature_names_in_"):
                    self.expected_features = list(self.model.feature_names_in_)
                elif hasattr(self.model, "feature_names_"):
                    self.expected_features = list(self.model.feature_names_)
                else:
                    raise FileNotFoundError("Could not determine expected features (no booster names, no JSON).")

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
        """Split expected_features into numeric vs group-OHE vs scikit-OHE."""
        ohe_cols, cat_keys, num_cols = [], set(), []
        grp_map: Dict[str, List[str]] = {pfx: [] for pfx in self.GROUP_OHE_PREFIXES.keys()}

        for col in self.expected_features:
            # scikit-OHE style (rare in your export)
            if "=" in col or col.startswith(("cat__", "ohe__")):
                ohe_cols.append(col)
                left = col.split("=")[0]
                rk = left.split("__")[-1] if "__" in left else left
                cat_keys.add(rk)
                continue

            # our group one-hots
            matched_group = False
            for pfx in self.GROUP_OHE_PREFIXES.keys():
                if col.startswith(pfx):
                    grp_map[pfx].append(col)
                    matched_group = True
                    break
            if matched_group:
                continue

            # numeric
            num_cols.append(col)

        self._ohe_cols = ohe_cols
        self._cat_raw_keys = list(cat_keys)
        self._num_cols = num_cols
        # drop empty groups
        self._grp_ohe_cols = {pfx: cols for pfx, cols in grp_map.items() if cols}

    # ───────────────────────── helpers ─────────────────────────
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

    # --- numeric & group OHE mapping ---
    def _pick_num(self, row: Dict[str, Any], expected_key: str):
        # 1) exact
        if expected_key in row:
            return row.get(expected_key)
        # 2) alias list
        for alt in self.NUMERIC_ALIASES.get(expected_key, []):
            if alt in row:
                return row.get(alt)
        # 3) normalized exact
        nk = _normkey(expected_key)
        inv = {_normkey(k): k for k in row.keys()}
        if nk in inv:
            return row.get(inv[nk])
        # 4) fuzzy on normalized keys (conservative)
        m = get_close_matches(nk, list(inv.keys()), n=1, cutoff=0.84)
        if m:
            return row.get(inv[m[0]])
        return None

    def _apply_group_ohe(self, row: Dict[str, Any], X: pd.DataFrame) -> Tuple[int, List[str]]:
        """Set one-hot groups like pullback_type_*, entry_rule_*, regime_1d_* based on row values."""
        set_cols: List[str] = []
        count = 0
        for pfx, raw_keys in self.GROUP_OHE_PREFIXES.items():
            cols = [c for c in X.columns if c.startswith(pfx)]
            if not cols:
                continue
            # find a value from any of the candidate raw keys
            val = None
            for rk in raw_keys:
                if rk in row and row[rk] not in (None, ""):
                    val = str(row[rk]).strip()
                    break
            if val is None:
                continue
            for c in cols:
                on = c[len(pfx):]  # the category part
                bit = 1.0 if on == val else 0.0
                X.at[0, c] = bit
                if bit == 1.0:
                    set_cols.append(c)
                    count += 1
        return count, set_cols

    # ───────────────────────── build X ─────────────────────────
    def _build_X(self, row: Dict[str, Any]) -> pd.DataFrame:
        if not self.expected_features:
            raise RuntimeError("WinProbScorer not loaded")

        r = self._normalize_keys(row)

        # Start with an all-zero row with expected columns
        X = pd.DataFrame([[0.0] * len(self.expected_features)], columns=list(self.expected_features))

        # 1) scikit-OHE block (rare for your export)
        if self._ohe_cols:
            df_cat = self._ohe_transform(r)
            for c in df_cat.columns:
                if c in X.columns:
                    X.at[0, c] = float(df_cat.iloc[0][c])

        # 2) group OHE block (your export style)
        grp_set_count, grp_set_cols = self._apply_group_ohe(r, X)

        # 3) numeric block
        numeric_matched = 0
        for c in self._num_cols:
            v = self._pick_num(r, c)
            try:
                X.at[0, c] = float(v) if v is not None and np.isfinite(v) else 0.0
                if v is not None and np.isfinite(v):
                    numeric_matched += 1
            except Exception:
                X.at[0, c] = 0.0

        # one-time diagnostics
        if not self._diag_once:
            nz = int((np.abs(X.to_numpy()) > 0).sum())
            first_nz = [name for name, val in zip(X.columns, X.to_numpy()[0]) if val != 0.0][:15]
            LOG.info("[WINPROB DIAG] features=%d  nonzero=%d  numeric_matched=%d  group_ohe_set=%d",
                     len(self.expected_features), nz, numeric_matched, grp_set_count)
            LOG.info("[WINPROB DIAG] first_nonzero: %s", first_nz)
            self._diag_once = True

        return X.astype(float)

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

        p_raw = float(self.model.predict_proba(X)[:, 1][0])
        p = self._calibrate(p_raw)

        # identical vector detector (to catch mapping issues early)
        try:
            vec = X.to_numpy()
            h = hashlib.md5(vec.tobytes()).hexdigest()
            if self._last_hash == h:
                self._same_vec_count += 1
                if self._same_vec_count in (2, 5, 10):
                    nz = int((np.abs(vec) > 0).sum())
                    LOG.info("[WINPROB DIAG] %d identical feature vectors in a row (hash=%s, nonzero=%d).",
                             self._same_vec_count, h, nz)
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
