# research/modeling/multiple_tests.py
import numpy as np
from typing import Tuple

# DSR per Bailey & López de Prado (2014). :contentReference[oaicite:6]{index=6}
def deflated_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0, trials: int = 1) -> float:
    """
    Very small-sample friendly PSA/DSR approximation:
    - Compute Probabilistic Sharpe (PSR) vs. sr_benchmark
    - Then deflate by number of trials (Bonferroni-style crude upper bound)
    This is intentionally conservative; replace with full DSR later.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 10:
        return 0.0
    sr = np.mean(r) / (np.std(r, ddof=1) + 1e-12)
    n = r.size
    # Probabilistic Sharpe (PSR) ~ N approximation
    z = (sr - sr_benchmark) * np.sqrt(n - 1)
    from math import erf, sqrt
    psr = 0.5 * (1 + erf(z / np.sqrt(2)))
    # Deflate by #trials (quick-and-dirty)
    psr_deflated = max(0.0, 1.0 - (1.0 - psr) * trials)
    return float(psr_deflated)

# SPA placeholder (Hansen 2005): we’ll wire a block bootstrap later. :contentReference[oaicite:7]{index=7}
def spa_pass_fail(series_family: np.ndarray, baseline: np.ndarray, alpha: float = 0.10) -> Tuple[bool, float]:
    """
    Placeholder that returns (False, 1.0) by default.
    Integrate a stationary/block bootstrap to estimate p-values over the max loss differential.
    """
    return False, 1.0
