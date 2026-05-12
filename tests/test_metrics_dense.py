from __future__ import annotations

import unittest

import numpy as np

from xmtc_robustness.metrics import (
    inverse_propensity_scores,
    ndcg_at_k,
    precision_at_k,
    ps_ndcg_at_k,
    ps_precision_at_k,
)


class MetricsDenseTest(unittest.TestCase):
    def test_precision_and_ndcg_dense(self) -> None:
        y_true = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ]
        )
        y_pred = np.array(
            [
                [0, 1],
                [3, 0],
            ]
        )

        self.assertAlmostEqual(precision_at_k(y_true, y_pred, 2), 0.5)
        self.assertGreater(ndcg_at_k(y_true, y_pred, 2), 0.0)

    def test_inverse_propensity_weights_rare_labels_more(self) -> None:
        counts = np.array([100, 10, 1], dtype=float)
        inv = inverse_propensity_scores(counts, n_samples=200)

        self.assertLess(inv[0], inv[1])
        self.assertLess(inv[1], inv[2])

    def test_ps_metrics_accept_dense_inputs(self) -> None:
        y_true = np.array([[1, 0, 1]])
        y_pred = np.array([[2, 0]])
        inv = np.array([1.0, 2.0, 4.0])

        self.assertAlmostEqual(ps_precision_at_k(y_true, y_pred, 2, inv), 2.5)
        self.assertAlmostEqual(ps_ndcg_at_k(y_true, y_pred, 2, inv), 1.0)


if __name__ == "__main__":
    unittest.main()
