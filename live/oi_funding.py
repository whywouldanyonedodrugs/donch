from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .regimes_report import RegimeThresholds


WIN_1H = 12
WIN_4H = 48
WIN_1D = 288
WIN_3D = 3 * WIN_1D
WIN_7D = 7 * WIN_1D


class StaleDerivativesDataError(RuntimeError):
    pass


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
    out = pd.Series(np.nan, index=funding_rate.index, dtype="float64")
    ok = funding_rate.notna()
    fr = funding_rate[ok]
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

    for col in ("close", "open_interest", "funding_rate"):
        if col not in df.columns:
            raise KeyError(f"Missing required raw column: {col}")

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
