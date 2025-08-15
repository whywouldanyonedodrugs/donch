# research/modeling/splits.py
import numpy as np
from sklearn.model_selection import KFold

class PurgedKFold:
    """
    Time-aware CV splitter with an embargo period to prevent leakage from
    overlapping labels. Provide a strictly increasing 'order' (e.g., by opened_at).

    Parameters
    ----------
    n_splits : int
        Number of folds.
    embargo : int
        Number of index positions (bars/rows) to exclude on each side of the test
        window from the training set (e.g., for 5m bars and a 4h time-exit, use ~48).
    """
    def __init__(self, n_splits: int = 5, embargo: int = 0):
        self.n_splits = int(n_splits)
        self.embargo = int(embargo)

    def split(self, X, y=None, groups=None, order=None):
        """
        Yields (train_idx, test_idx) as integer position indices suitable for X.iloc[...].

        Notes
        -----
        - We expect 'order' to be a 1D array of length len(X) with increasing
          values that reflect time order (e.g., 0..N-1). If None, we assume
          the current row order is already time-ordered.
        - KFold is run WITHOUT shuffling to respect time ordering.
        """
        n = len(X)
        order_idx = np.arange(n) if order is None else np.asarray(order)
        if order_idx.shape[0] != n:
            raise ValueError("order length must match X length")

        # KFold yields indices into the array we pass (order_idx)
        kf = KFold(self.n_splits, shuffle=False)
        for tr_mask, te_mask in kf.split(order_idx):
            # te_mask are positions into order_idx; map to their time-index values
            test_time_idx = order_idx[te_mask]
            lo, hi = int(test_time_idx.min()), int(test_time_idx.max())

            # Purge + embargo region around test from the training pool
            lo_purge = max(0, lo - self.embargo)
            hi_purge = min(order_idx.max(), hi + self.embargo)

            # keep rows whose time index is strictly outside [lo_purge, hi_purge]
            keep_mask = (order_idx < lo_purge) | (order_idx > hi_purge)

            # Convert boolean keep_mask to position indices 0..n-1
            train_idx = np.where(keep_mask)[0]
            test_idx = te_mask  # already 0..n-1 positions
            yield train_idx, test_idx
