from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


PROPENSITY_PRESETS: dict[str, tuple[float, float]] = {
    "default": (0.55, 1.5),
    "wiki10-31k": (0.55, 1.5),
    "wikilshtc": (0.5, 0.4),
    "wikilshtc-325k": (0.5, 0.4),
    "amazon-670k": (0.6, 2.6),
    "amazoncat-13k": (0.6, 2.6),
}


def _is_sparse_like(matrix: Any) -> bool:
    return hasattr(matrix, "tocsr") and not isinstance(matrix, np.ndarray)


def _as_csr(matrix: Any) -> Any:
    return matrix.tocsr() if hasattr(matrix, "tocsr") else matrix


def _validate_predictions(y_pred: Any, k: int | None = None) -> np.ndarray:
    pred = np.asarray(y_pred, dtype=np.int64)
    if pred.ndim != 2:
        raise ValueError("y_pred must have shape `(n_samples, k)`.")
    if k is not None:
        if k <= 0:
            raise ValueError("k must be positive.")
        if pred.shape[1] < k:
            raise ValueError(f"y_pred only has {pred.shape[1]} columns, cannot evaluate k={k}.")
        pred = pred[:, :k]
    if np.any(pred < 0):
        raise ValueError("y_pred contains negative label ids.")
    return pred


def _allowed_mask(allowed_labels: Any | None, n_labels: int) -> np.ndarray | None:
    if allowed_labels is None:
        return None

    arr = np.asarray(allowed_labels)
    if arr.dtype == bool:
        if arr.shape != (n_labels,):
            raise ValueError("Boolean allowed_labels mask must have shape `(n_labels,)`.")
        return arr

    mask = np.zeros(n_labels, dtype=bool)
    mask[arr.astype(np.int64)] = True
    return mask


def label_counts_from_matrix(y: Any) -> np.ndarray:
    """Return label frequencies without densifying SciPy sparse matrices."""

    if hasattr(y, "sum"):
        return np.asarray(y.sum(axis=0)).ravel().astype(np.float64, copy=False)

    arr = np.asarray(y)
    if arr.ndim != 2:
        raise ValueError("Label matrix must be 2D.")
    return arr.sum(axis=0).astype(np.float64, copy=False)


def row_positive_counts(y_true: Any, allowed_labels: Any | None = None) -> np.ndarray:
    """Count true labels per instance, optionally restricted to a label subset."""

    if _is_sparse_like(y_true):
        y_csr = _as_csr(y_true)
        n_samples, n_labels = y_csr.shape
        mask = _allowed_mask(allowed_labels, n_labels)
        if mask is None and np.all(y_csr.data != 0):
            return np.diff(y_csr.indptr).astype(np.int64, copy=False)

        counts = np.zeros(n_samples, dtype=np.int64)
        row_nnz = np.diff(y_csr.indptr)
        if y_csr.indices.size:
            row_ids = np.repeat(np.arange(n_samples), row_nnz)
            selected = y_csr.data != 0
            if mask is not None:
                selected &= mask[y_csr.indices]
            np.add.at(counts, row_ids, selected.astype(np.int64))
        return counts

    arr = np.asarray(y_true)
    if arr.ndim != 2:
        raise ValueError("y_true must be a 2D label matrix.")
    mask = _allowed_mask(allowed_labels, arr.shape[1])
    positives = arr != 0
    if mask is not None:
        positives = positives[:, mask]
    return positives.sum(axis=1).astype(np.int64, copy=False)


def relevance_at_k(
    y_true: Any,
    y_pred: Any,
    *,
    k: int | None = None,
    allowed_labels: Any | None = None,
) -> np.ndarray:
    """Return a boolean hit matrix for predicted labels.

    This is the shared primitive for all metrics. The optional `allowed_labels`
    mask lets the same predictions be evaluated on head or tail labels without
    retraining, exposing whether a model only wins by repeating frequent labels.
    """

    pred = _validate_predictions(y_pred, k=k)
    n_samples, width = pred.shape

    if _is_sparse_like(y_true):
        y_csr = _as_csr(y_true)
        if y_csr.shape[0] != n_samples:
            raise ValueError("y_true and y_pred have different sample counts.")
        if pred.max(initial=0) >= y_csr.shape[1]:
            raise ValueError("y_pred contains label ids outside y_true.")

        rows = np.repeat(np.arange(n_samples), width)
        values = y_csr[rows, pred.reshape(-1)]
        hits = np.asarray(values).reshape(n_samples, width) != 0
        mask = _allowed_mask(allowed_labels, y_csr.shape[1])
    else:
        arr = np.asarray(y_true)
        if arr.ndim != 2:
            raise ValueError("y_true must be a 2D label matrix.")
        if arr.shape[0] != n_samples:
            raise ValueError("y_true and y_pred have different sample counts.")
        if pred.max(initial=0) >= arr.shape[1]:
            raise ValueError("y_pred contains label ids outside y_true.")

        hits = arr[np.arange(n_samples)[:, None], pred] != 0
        mask = _allowed_mask(allowed_labels, arr.shape[1])

    if mask is not None:
        hits &= mask[pred]
    return hits


def precision_at_k(
    y_true: Any,
    y_pred: Any,
    k: int,
    *,
    allowed_labels: Any | None = None,
) -> float:
    """Compute mean Precision@k.

    Precision@k measures how often top-ranked predictions are exactly correct.
    In this project it acts as the conventional metric that may over-reward
    the global-frequency baseline on imbalanced labels.
    """

    pred = _validate_predictions(y_pred, k=k)
    hits = relevance_at_k(y_true, pred, allowed_labels=allowed_labels)
    return float((hits.sum(axis=1) / pred.shape[1]).mean())


def ndcg_at_k(
    y_true: Any,
    y_pred: Any,
    k: int,
    *,
    allowed_labels: Any | None = None,
) -> float:
    """Compute mean nDCG@k with binary relevance."""

    pred = _validate_predictions(y_pred, k=k)
    hits = relevance_at_k(y_true, pred, allowed_labels=allowed_labels).astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, pred.shape[1] + 2, dtype=np.float64))
    dcg = hits @ discounts

    counts = np.minimum(row_positive_counts(y_true, allowed_labels), pred.shape[1])
    discount_cumsum = np.concatenate(([0.0], np.cumsum(discounts)))
    idcg = discount_cumsum[counts]
    scores = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)
    return float(scores.mean())


def propensity_scores(
    label_counts: Any,
    n_samples: int,
    *,
    a: float = 0.55,
    b: float = 1.5,
    min_propensity: float = 1e-12,
) -> np.ndarray:
    """Compute Jain et al. label propensities from training frequencies.

    The formula gives rare labels smaller propensity values, so their inverse
    propensity weights are larger in PS-Precision and PS-nDCG. This is the
    correction whose robustness against the frequency baseline is being tested.
    """

    counts = np.asarray(label_counts, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError("label_counts must be a 1D array.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if b <= 0:
        raise ValueError("b must be positive.")

    c = (np.log(float(n_samples)) - 1.0) * ((b + 1.0) ** a)
    c = max(c, 0.0)
    prop = 1.0 / (1.0 + c * np.exp(-a * np.log(counts + b)))
    return np.clip(prop, min_propensity, 1.0)


@dataclass(frozen=True)
class _InversePropensityFactory:
    def __call__(
        self,
        label_counts: Any,
        n_samples: int,
        *,
        a: float = 0.55,
        b: float = 1.5,
        min_propensity: float = 1e-12,
    ) -> np.ndarray:
        prop = propensity_scores(
            label_counts,
            n_samples,
            a=a,
            b=b,
            min_propensity=min_propensity,
        )
        return 1.0 / prop

    def from_label_matrix(
        self,
        y_train: Any,
        *,
        dataset: str = "default",
        a: float | None = None,
        b: float | None = None,
        min_propensity: float = 1e-12,
    ) -> np.ndarray:
        """Build inverse propensity weights from the training label matrix."""

        counts = label_counts_from_matrix(y_train)
        n_samples = y_train.shape[0]
        if a is None or b is None:
            preset_a, preset_b = PROPENSITY_PRESETS.get(dataset.lower(), PROPENSITY_PRESETS["default"])
            a = preset_a if a is None else a
            b = preset_b if b is None else b
        return self(counts, n_samples, a=a, b=b, min_propensity=min_propensity)


inverse_propensity_scores = _InversePropensityFactory()


def ps_precision_at_k(
    y_true: Any,
    y_pred: Any,
    k: int,
    inverse_propensity: Any,
    *,
    allowed_labels: Any | None = None,
) -> float:
    """Compute mean Propensity-Scored Precision@k."""

    pred = _validate_predictions(y_pred, k=k)
    inv = np.asarray(inverse_propensity, dtype=np.float64)
    if pred.max(initial=0) >= inv.shape[0]:
        raise ValueError("inverse_propensity is shorter than the largest predicted label id.")

    hits = relevance_at_k(y_true, pred, allowed_labels=allowed_labels).astype(np.float64)
    gains = hits * inv[pred]
    return float((gains.sum(axis=1) / pred.shape[1]).mean())


def _ideal_weighted_dcg(
    y_true: Any,
    inverse_propensity: np.ndarray,
    k: int,
    allowed_labels: Any | None,
) -> np.ndarray:
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))

    if _is_sparse_like(y_true):
        y_csr = _as_csr(y_true)
        n_samples, n_labels = y_csr.shape
        mask = _allowed_mask(allowed_labels, n_labels)
        out = np.zeros(n_samples, dtype=np.float64)
        for row in range(n_samples):
            start, end = y_csr.indptr[row], y_csr.indptr[row + 1]
            labels = y_csr.indices[start:end]
            data = y_csr.data[start:end]
            labels = labels[data != 0]
            if mask is not None:
                labels = labels[mask[labels]]
            if labels.size == 0:
                continue
            weights = np.sort(inverse_propensity[labels])[-k:][::-1]
            out[row] = float(weights @ discounts[: weights.shape[0]])
        return out

    arr = np.asarray(y_true)
    n_samples, n_labels = arr.shape
    mask = _allowed_mask(allowed_labels, n_labels)
    out = np.zeros(n_samples, dtype=np.float64)
    for row in range(n_samples):
        labels = np.flatnonzero(arr[row])
        if mask is not None:
            labels = labels[mask[labels]]
        if labels.size == 0:
            continue
        weights = np.sort(inverse_propensity[labels])[-k:][::-1]
        out[row] = float(weights @ discounts[: weights.shape[0]])
    return out


def ps_ndcg_at_k(
    y_true: Any,
    y_pred: Any,
    k: int,
    inverse_propensity: Any,
    *,
    allowed_labels: Any | None = None,
) -> float:
    """Compute mean Propensity-Scored nDCG@k.

    The ideal ranking is formed by the true labels with the largest inverse
    propensity weights, making the denominator tail-aware as in propensity
    scored ranking evaluations.
    """

    pred = _validate_predictions(y_pred, k=k)
    inv = np.asarray(inverse_propensity, dtype=np.float64)
    if pred.max(initial=0) >= inv.shape[0]:
        raise ValueError("inverse_propensity is shorter than the largest predicted label id.")

    hits = relevance_at_k(y_true, pred, allowed_labels=allowed_labels).astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, pred.shape[1] + 2, dtype=np.float64))
    dcg = (hits * inv[pred]) @ discounts
    idcg = _ideal_weighted_dcg(y_true, inv, pred.shape[1], allowed_labels)
    scores = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)
    return float(scores.mean())


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    *,
    ks: Iterable[int] = (1, 3, 5),
    inverse_propensity: Any | None = None,
    allowed_labels: Any | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """Evaluate a prediction matrix with traditional and PS metrics."""

    results: dict[str, float] = {}
    for k in ks:
        results[f"{prefix}precision@{k}"] = precision_at_k(
            y_true,
            y_pred,
            k,
            allowed_labels=allowed_labels,
        )
        results[f"{prefix}ndcg@{k}"] = ndcg_at_k(
            y_true,
            y_pred,
            k,
            allowed_labels=allowed_labels,
        )
        if inverse_propensity is not None:
            results[f"{prefix}ps_precision@{k}"] = ps_precision_at_k(
                y_true,
                y_pred,
                k,
                inverse_propensity,
                allowed_labels=allowed_labels,
            )
            results[f"{prefix}ps_ndcg@{k}"] = ps_ndcg_at_k(
                y_true,
                y_pred,
                k,
                inverse_propensity,
                allowed_labels=allowed_labels,
            )
    return results
