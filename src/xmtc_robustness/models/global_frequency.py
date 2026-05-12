from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _column_counts(y: Any) -> np.ndarray:
    if hasattr(y, "sum"):
        counts = y.sum(axis=0)
        return np.asarray(counts).ravel().astype(np.float64, copy=False)

    arr = np.asarray(y)
    if arr.ndim != 2:
        raise ValueError("y_train must be a 2D label matrix.")
    return arr.sum(axis=0).astype(np.float64, copy=False)


@dataclass
class GlobalFrequencyRanker:
    """Naive XMTC baseline that ranks labels by global training frequency.

    The model deliberately ignores document text. If propensity-scored metrics
    are robust, this ranker should lose relative ground on tail-sensitive
    evaluations even when frequent labels dominate the dataset.
    """

    label_counts_: np.ndarray | None = field(default=None, init=False)
    ranking_: np.ndarray | None = field(default=None, init=False)
    n_labels_: int | None = field(default=None, init=False)

    def fit(self, y_train: Any) -> "GlobalFrequencyRanker":
        """Compute a global label-frequency ranking from a sparse/dense matrix.

        Parameters
        ----------
        y_train:
            Binary multi-label matrix with shape `(n_train, n_labels)`. SciPy
            sparse matrices are handled without densifying.
        """

        counts = _column_counts(y_train)
        label_ids = np.arange(counts.shape[0], dtype=np.int64)

        self.label_counts_ = counts
        self.ranking_ = np.lexsort((label_ids, -counts)).astype(np.int64, copy=False)
        self.n_labels_ = counts.shape[0]
        return self

    def predict_top_k(
        self,
        x: Any | None = None,
        *,
        n_samples: int | None = None,
        k: int = 5,
    ) -> np.ndarray:
        """Return the same top-k frequent labels for every requested sample.

        The prediction is vectorized with `np.broadcast_to`, so creating
        predictions for thousands of test instances does not require a Python
        loop over instances.
        """

        if self.ranking_ is None or self.n_labels_ is None:
            raise RuntimeError("GlobalFrequencyRanker must be fitted first.")

        if n_samples is None:
            if isinstance(x, int):
                n_samples = x
            elif x is not None:
                n_samples = len(x)
            else:
                raise ValueError("Provide either `x` or `n_samples`.")

        if k <= 0:
            raise ValueError("k must be positive.")
        if k > self.n_labels_:
            raise ValueError(f"k={k} exceeds number of labels={self.n_labels_}.")

        top_labels = self.ranking_[:k]
        return np.broadcast_to(top_labels, (int(n_samples), k)).copy()
