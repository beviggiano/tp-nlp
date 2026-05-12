from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .metrics import evaluate_predictions, label_counts_from_matrix


def label_counts(y_train: Any) -> np.ndarray:
    """Return training label counts used to define head/tail partitions."""

    return label_counts_from_matrix(y_train)


def labels_to_mask(labels: Any, n_labels: int) -> np.ndarray:
    mask = np.zeros(n_labels, dtype=bool)
    mask[np.asarray(labels, dtype=np.int64)] = True
    return mask


def frequency_partitions(
    counts: Any,
    *,
    head_fraction: float = 0.2,
    tail_fraction: float = 0.2,
    include_zero_labels: bool = False,
) -> dict[str, np.ndarray]:
    """Split labels into head, middle, and tail by global frequency.

    These subsets support sensitivity analysis: a metric is more robust when it
    exposes that a frequency-only baseline performs poorly on the tail.
    """

    counts_arr = np.asarray(counts, dtype=np.float64)
    if counts_arr.ndim != 1:
        raise ValueError("counts must be a 1D array.")
    if not 0 < head_fraction <= 1:
        raise ValueError("head_fraction must be in (0, 1].")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be in (0, 1].")

    eligible = np.arange(counts_arr.shape[0], dtype=np.int64)
    if not include_zero_labels:
        eligible = eligible[counts_arr > 0]
    if eligible.size == 0:
        return {"head": np.array([], dtype=np.int64), "middle": np.array([], dtype=np.int64), "tail": np.array([], dtype=np.int64)}

    desc = eligible[np.lexsort((eligible, -counts_arr[eligible]))]
    asc = eligible[np.lexsort((eligible, counts_arr[eligible]))]

    n_head = max(1, int(np.ceil(eligible.size * head_fraction)))
    n_tail = max(1, int(np.ceil(eligible.size * tail_fraction)))
    head = desc[:n_head]
    tail = asc[:n_tail]

    used = labels_to_mask(np.concatenate((head, tail)), counts_arr.shape[0])
    middle = eligible[~used[eligible]]
    return {"head": head, "middle": middle, "tail": tail}


def evaluate_by_frequency_partition(
    y_true: Any,
    y_pred: Any,
    counts: Any,
    *,
    ks: tuple[int, ...] = (1, 3, 5),
    inverse_propensity: Any | None = None,
    head_fraction: float = 0.2,
    tail_fraction: float = 0.2,
) -> dict[str, float]:
    """Evaluate one prediction matrix separately on head/middle/tail labels."""

    partitions = frequency_partitions(
        counts,
        head_fraction=head_fraction,
        tail_fraction=tail_fraction,
    )
    n_labels = np.asarray(counts).shape[0]
    results: dict[str, float] = {}

    for name, labels in partitions.items():
        if labels.size == 0:
            continue
        mask = labels_to_mask(labels, n_labels)
        results.update(
            evaluate_predictions(
                y_true,
                y_pred,
                ks=ks,
                inverse_propensity=inverse_propensity,
                allowed_labels=mask,
                prefix=f"{name}/",
            )
        )
    return results


def compare_predictions(
    y_true: Any,
    predictions: Mapping[str, Any],
    *,
    ks: tuple[int, ...] = (1, 3, 5),
    inverse_propensity: Any | None = None,
    counts: Any | None = None,
    include_frequency_partitions: bool = True,
) -> dict[str, dict[str, float]]:
    """Evaluate multiple rankers with the same metric engine.

    The returned dictionary is ready for tables comparing the global-frequency
    baseline against retrieval models across traditional and propensity-scored
    metrics.
    """

    results: dict[str, dict[str, float]] = {}
    for name, y_pred in predictions.items():
        model_results = evaluate_predictions(
            y_true,
            y_pred,
            ks=ks,
            inverse_propensity=inverse_propensity,
        )
        if include_frequency_partitions:
            if counts is None:
                raise ValueError("counts is required for frequency partition analysis.")
            model_results.update(
                evaluate_by_frequency_partition(
                    y_true,
                    y_pred,
                    counts,
                    ks=ks,
                    inverse_propensity=inverse_propensity,
                )
            )
        results[name] = model_results
    return results
