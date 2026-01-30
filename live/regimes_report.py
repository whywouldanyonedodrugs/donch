from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class RegimeThresholds:
    funding_neutral_eps: float
    oi_source: str
    oi_q33: float
    oi_q66: float
    btc_vol_hi: float


def load_regimes_report(meta_dir: Path) -> Dict[str, Any]:
    meta_dir = Path(meta_dir)
    p = meta_dir / "regimes_report.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"regimes_report.json must be a dict: {p}")
    return obj


def load_regime_thresholds(meta_dir: Path) -> RegimeThresholds:
    obj = load_regimes_report(meta_dir)
    thr = obj.get("thresholds")
    if not isinstance(thr, dict):
        raise ValueError(f"regimes_report.json missing dict 'thresholds' in {Path(meta_dir) / 'regimes_report.json'}")

    def _need(key: str) -> Any:
        if key not in thr:
            raise KeyError(f"regimes_report.json thresholds missing '{key}'")
        return thr[key]

    funding_neutral_eps = float(_need("funding_neutral_eps"))
    oi_source = str(_need("oi_source"))
    oi_q33 = float(_need("oi_q33"))
    oi_q66 = float(_need("oi_q66"))
    btc_vol_hi = float(_need("btc_vol_hi"))

    if oi_source not in ("oi_z_7d", "oi_pct_1d"):
        raise ValueError(f"Unsupported oi_source={oi_source!r} (expected 'oi_z_7d' or 'oi_pct_1d')")

    if not (oi_q33 <= oi_q66):
        raise ValueError(f"Invalid oi thresholds: oi_q33={oi_q33} > oi_q66={oi_q66}")

    if funding_neutral_eps <= 0:
        raise ValueError(f"Invalid funding_neutral_eps={funding_neutral_eps} (must be > 0)")

    return RegimeThresholds(
        funding_neutral_eps=funding_neutral_eps,
        oi_source=oi_source,
        oi_q33=oi_q33,
        oi_q66=oi_q66,
        btc_vol_hi=btc_vol_hi,
    )
