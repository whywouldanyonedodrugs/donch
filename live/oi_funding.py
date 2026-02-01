from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

import inspect
import os


from .regimes_report import RegimeThresholds


WIN_1H = 12
WIN_4H = 48
WIN_1D = 288
WIN_3D = 3 * WIN_1D
WIN_7D = 7 * WIN_1D


class StaleDerivativesDataError(RuntimeError):
    pass

def _coerce_thresholds(thresholds: Any) -> RegimeThresholds:
    """
    LiveTrader historically passes a plain dict loaded from regimes_report.json.
    The OI/Funding feature code expects a RegimeThresholds object.
    Coerce dict -> RegimeThresholds (ignoring any extra keys).
    """
    if isinstance(thresholds, RegimeThresholds):
        return thresholds

    if isinstance(thresholds, dict):
        t = thresholds
        # These keys are required by add_oi_funding_features below.
        return RegimeThresholds(
            funding_neutral_eps=float(t["funding_neutral_eps"]),
            oi_source=str(t["oi_source"]),
            oi_q33=float(t["oi_q33"]),
            oi_q66=float(t["oi_q66"]),
            btc_vol_hi=float(t["btc_vol_hi"]),
        )

    raise TypeError(f"thresholds must be RegimeThresholds or dict, got {type(thresholds)!r}")



def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out = out.sort_index()
        return out

    if "timestamp" not in df.columns:
        raise KeyError("df must have DatetimeIndex or a 'timestamp' column")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.set_index("timestamp", drop=True).sort_index()
    return out


def _zscore_rolling(x: pd.Series, win: int, *, min_periods: int) -> pd.Series:
    mu = x.rolling(win, min_periods=min_periods).mean()
    sd = x.rolling(win, min_periods=min_periods).std()
    return (x - mu) / (sd + 1e-12)


def _bucket_3way(x: pd.Series, q33: float, q66: float) -> pd.Series:
    # returns float series with values -1, 0, +1 or NaN
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    out = out.where(~x.notna(), out)  # keep NaN where x is NaN
    out.loc[x <= q33] = -1.0
    out.loc[(x > q33) & (x < q66)] = 0.0
    out.loc[x >= q66] = 1.0
    return out


def _funding_regime_code(funding_rate: pd.Series, eps: float) -> pd.Series:
    # Ensure numeric; keep index intact
    fr = pd.to_numeric(funding_rate, errors="coerce")

    out = pd.Series(np.nan, index=fr.index, dtype="float64")
    out.loc[fr <= -eps] = -1.0
    out.loc[(fr > -eps) & (fr < eps)] = 0.0
    out.loc[fr >= eps] = 1.0
    return out



def add_oi_funding_features(
    df5: pd.DataFrame,
    *,
    thresholds: RegimeThresholds,
    decision_ts: Optional[pd.Timestamp] = None,
    staleness_max_age: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """
    Computes OI/Funding feature block on a 5m grid, using as-of semantics via ffill.

    Required raw columns on df5:
      - close
      - open_interest
      - funding_rate

    Adds columns:
      - oi_level, oi_notional_est, oi_pct_1h/4h/1d, oi_z_7d, oi_chg_norm_vol_1h, oi_price_div_1h
      - funding_rate (ffill), funding_abs, funding_z_7d, funding_rollsum_3d, funding_oi_div, est_leverage
      - funding_regime_code, oi_regime_code, S3_funding_x_oi
    """
    df = _ensure_dt_index(df5)
    thresholds = _coerce_thresholds(thresholds)

    if df.empty:
        raise ValueError("df5 is empty")

    for col in ("close", "open_interest", "funding_rate"):
        if col not in df.columns:
            raise KeyError(f"Missing required raw column: {col}")

    # If staleness enforcement is requested but decision_ts isn't provided,
    # infer decision_ts as the last available 5m timestamp (as-of semantics).
    if staleness_max_age is not None and decision_ts is None:
        decision_ts = df.index[-1]

    # as-of mapping on 5m grid: forward-fill raw derivatives
    close = df["close"].astype("float64")
    oi = df["open_interest"].astype("float64").ffill()
    fr = df["funding_rate"].astype("float64").ffill()

    if decision_ts is not None:
        decision_ts = pd.to_datetime(decision_ts, utc=True)
        if decision_ts < df.index.min():
            raise ValueError(f"decision_ts {decision_ts} is before df start {df.index.min()}")

        # staleness: require last known non-NaN raw to be recent enough
        if staleness_max_age is not None:
            staleness_max_age = pd.Timedelta(staleness_max_age)

            last_oi = df["open_interest"].dropna().index.max() if df["open_interest"].notna().any() else None
            last_fr = df["funding_rate"].dropna().index.max() if df["funding_rate"].notna().any() else None

            if last_oi is None or (decision_ts - last_oi) > staleness_max_age:
                raise StaleDerivativesDataError(
                    f"open_interest stale or missing: last={last_oi}, decision_ts={decision_ts}, max_age={staleness_max_age}"
                )
            if last_fr is None or (decision_ts - last_fr) > staleness_max_age:
                raise StaleDerivativesDataError(
                    f"funding_rate stale or missing: last={last_fr}, decision_ts={decision_ts}, max_age={staleness_max_age}"
                )

    out = df.copy()

    out["oi_level"] = oi
    out["oi_notional_est"] = oi * close

    out["oi_pct_1h"] = out["oi_level"].pct_change(WIN_1H)
    out["oi_pct_4h"] = out["oi_level"].pct_change(WIN_4H)
    out["oi_pct_1d"] = out["oi_level"].pct_change(WIN_1D)

    # z-score of oi_pct_1d over 7d
    out["oi_z_7d"] = _zscore_rolling(out["oi_pct_1d"], WIN_7D, min_periods=WIN_1D)

    # normalized by abs 1h price move
    price_pct_1h = close.pct_change(WIN_1H)
    out["oi_chg_norm_vol_1h"] = out["oi_pct_1h"] / (price_pct_1h.abs() + 1e-12)

    # divergence
    out["oi_price_div_1h"] = out["oi_pct_1h"] - price_pct_1h

    out["funding_rate"] = fr
    out["funding_abs"] = out["funding_rate"].abs()

    out["funding_z_7d"] = _zscore_rolling(out["funding_rate"], WIN_7D, min_periods=WIN_1D)
    out["funding_rollsum_3d"] = out["funding_rate"].rolling(WIN_3D, min_periods=WIN_1D).sum()

    out["funding_oi_div"] = out["funding_z_7d"] * out["oi_z_7d"]
    out["est_leverage"] = (out["oi_z_7d"].abs() + 0.5) * (out["funding_z_7d"].abs() + 0.5)

    # regime codes (no extra shifting; assumes as-of mapped)
    out["funding_regime_code"] = _funding_regime_code(out["funding_rate"], thresholds.funding_neutral_eps)

    if thresholds.oi_source == "oi_z_7d":
        oi_val = out["oi_z_7d"]
    elif thresholds.oi_source == "oi_pct_1d":
        oi_val = out["oi_pct_1d"]
    else:
        raise ValueError(f"Unsupported oi_source={thresholds.oi_source!r}")

    out["oi_regime_code"] = _bucket_3way(oi_val, thresholds.oi_q33, thresholds.oi_q66)

    # S3 = (funding_regime_code + 1)*3 + (oi_regime_code + 1)
    frc = out["funding_regime_code"]
    oic = out["oi_regime_code"]
    s3 = (frc + 1.0) * 3.0 + (oic + 1.0)
    s3 = s3.where(frc.notna() & oic.notna(), np.nan)
    out["S3_funding_x_oi"] = s3

    return out


def oi_funding_features_at_decision(
    df5: pd.DataFrame,
    decision_ts: pd.Timestamp,
    *,
    thresholds: RegimeThresholds,
    staleness_max_age: Optional[pd.Timedelta] = None,
) -> Dict[str, float]:
    """
    Convenience: compute the block and return the feature values at the as-of decision timestamp.
    Uses pure as-of semantics: last row at or before decision_ts.
    """
    df = _ensure_dt_index(df5)
    decision_ts = pd.to_datetime(decision_ts, utc=True)
    df = df.loc[:decision_ts]
    if df.empty:
        raise ValueError(f"No rows at or before decision_ts={decision_ts}")

    df2 = add_oi_funding_features(df, thresholds=thresholds, decision_ts=decision_ts, staleness_max_age=staleness_max_age)
    row = df2.iloc[-1]

    keys = [
        "oi_level",
        "oi_notional_est",
        "oi_pct_1h",
        "oi_pct_4h",
        "oi_pct_1d",
        "oi_z_7d",
        "oi_chg_norm_vol_1h",
        "oi_price_div_1h",
        "funding_rate",
        "funding_abs",
        "funding_z_7d",
        "funding_rollsum_3d",
        "funding_oi_div",
        "est_leverage",
        "funding_regime_code",
        "oi_regime_code",
        "S3_funding_x_oi",
    ]
    return {k: (float(row[k]) if pd.notna(row[k]) else np.nan) for k in keys}

# ---------------------------------------------------------------------
# Legacy API shims (used by live_trader.py)
# ---------------------------------------------------------------------

from typing import Any, Tuple  # noqa: E402


def _extract_series(obj: Any, *, preferred_cols: list[str]) -> pd.Series:
    """
    Normalize a Series/DataFrame-like into a numeric Series.
    - If obj is Series: returns it.
    - If obj is DataFrame: returns the first matching preferred col, else the first numeric col.
    Raises KeyError if it cannot find a usable column.
    """
    if obj is None:
        raise KeyError("Expected a Series/DataFrame, got None")

    if isinstance(obj, pd.Series):
        s = obj
    elif isinstance(obj, pd.DataFrame):
        for c in preferred_cols:
            if c in obj.columns:
                s = obj[c]
                break
        else:
            num_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
            if not num_cols:
                raise KeyError(f"No numeric columns found. cols={list(obj.columns)}")
            s = obj[num_cols[0]]
    else:
        raise TypeError(f"Unsupported type: {type(obj)!r}")

    s = pd.to_numeric(s, errors="coerce")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex on derivatives series")

    # ensure UTC + sorted
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    else:
        s.index = s.index.tz_convert("UTC")
    s = s.sort_index()
    return s


async def fetch_series_5m(
    exchange: Any,
    symbol: str,
    *,
    lookback_oi_days: int = 7,
    lookback_fr_days: int = 7,
    end_ts: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Legacy hook for LiveTrader.

    Contract:
      returns (oi5, fr5) where:
        - oi5 has DatetimeIndex and column "open_interest"
        - fr5 has DatetimeIndex and column "funding_rate"

    This is intentionally exchange-proxy specific. If your ExchangeProxy implements
    `fetch_oi_funding_series_5m(...)`, we delegate to it.

    If not implemented, we raise NotImplementedError (LiveTrader catches and fails-closed).
    """
    if hasattr(exchange, "fetch_oi_funding_series_5m"):
        res = exchange.fetch_oi_funding_series_5m(
            symbol,
            lookback_oi_days=lookback_oi_days,
            lookback_fr_days=lookback_fr_days,
            end_ts=end_ts,
        )
        if inspect.isawaitable(res):
            oi5, fr5 = await res
        else:
            oi5, fr5 = res
        return oi5, fr5

    raise NotImplementedError(
        "ExchangeProxy is missing fetch_oi_funding_series_5m(). "
        "Implement it (preferred) or replace LiveTrader OI/Funding ingestion with a DerivativesCache."
    )


def compute_oi_funding_features(
    *,
    df5: pd.DataFrame,
    oi5: Any,
    fr5: Any,
    thresholds: RegimeThresholds,
    allow_nans: bool = True,
    staleness_max_age: Optional[pd.Timedelta] = None,
) -> Dict[str, float]:
    """
    Legacy helper used by LiveTrader._build_oi_funding_features().

    Semantics:
      - As-of mapping: reindex(df5.index, method='ffill') (no wall-clock leakage).
      - decision_ts: last timestamp in df5 index.
      - Fail-closed on missing required raw at decision_ts (open_interest or funding_rate).
      - Staleness check uses LAST RAW observation timestamps (not ffilled timestamps).
      - If allow_nans is False: fail-closed if any returned feature is NaN.
    """
    df = _ensure_dt_index(df5)
    if len(df.index) == 0:
        raise ValueError("df5 is empty")

    decision_ts = df.index[-1]
    out = df.copy()

    oi_s = _extract_series(oi5, preferred_cols=["open_interest", "openInterest", "oi", "value"])
    fr_s = _extract_series(fr5, preferred_cols=["funding_rate", "fundingRate", "fr", "value"])

    # Staleness check must be based on raw series, not on ffilled grids.
    #
    # Funding cadence on Bybit perps is typically 8h. Offline pipelines often forward-fill the
    # last printed funding rate across 5m decision grids. To preserve strict "as-of" semantics
    # while avoiding unnecessary vetoes, allow configuring OI vs Funding staleness separately
    # via environment variables:
    #   - DONCH_OI_MAX_AGE_MIN
    #   - DONCH_FUNDING_MAX_AGE_MIN
    #
    # If env vars are not set, both fall back to `staleness_max_age` (legacy behavior).
    if staleness_max_age is not None:
        oi_max_age = staleness_max_age
        fr_max_age = staleness_max_age

        try:
            v = os.getenv('DONCH_OI_MAX_AGE_MIN')
            if v is not None and str(v).strip() != '':
                oi_max_age = pd.Timedelta(minutes=int(v))
        except Exception:
            pass

        try:
            v = os.getenv('DONCH_FUNDING_MAX_AGE_MIN')
            if v is not None and str(v).strip() != '':
                fr_max_age = pd.Timedelta(minutes=int(v))
        except Exception:
            pass

        last_oi = oi_s.dropna().index.max() if len(oi_s.dropna()) else None
        last_fr = fr_s.dropna().index.max() if len(fr_s.dropna()) else None
        if last_oi is None or last_fr is None:
            raise StaleDerivativesDataError('Derivatives series missing raw observations (OI/FR)')
        if (decision_ts - last_oi) > oi_max_age:
            raise StaleDerivativesDataError(
                f"OI stale: last={last_oi} decision_ts={decision_ts} max_age={oi_max_age}"
            )
        if (decision_ts - last_fr) > fr_max_age:
            raise StaleDerivativesDataError(
                f"Funding stale: last={last_fr} decision_ts={decision_ts} max_age={fr_max_age}"
            )

    # As-of mapping onto decision grid
    out["open_interest"] = oi_s.reindex(out.index, method="ffill")
    out["funding_rate"] = fr_s.reindex(out.index, method="ffill")

    # Required raw must exist at decision_ts after as-of mapping
    if pd.isna(out.at[decision_ts, "open_interest"]) or pd.isna(out.at[decision_ts, "funding_rate"]):
        raise KeyError("Missing required derivatives raw at decision_ts (open_interest/funding_rate)")

    feats = oi_funding_features_at_decision(
        out,
        decision_ts,
        thresholds=thresholds,
        staleness_max_age=None,  # already enforced above using raw timestamps
    )

    if not allow_nans:
        bad = [k for k, v in feats.items() if v is None or (isinstance(v, float) and np.isnan(v))]
        if bad:
            raise ValueError(f"NaNs in oi/funding derived features at decision_ts: {bad}")

    return feats
