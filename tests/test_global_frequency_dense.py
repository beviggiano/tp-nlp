from __future__ import annotations

import unittest

import numpy as np

from xmtc_robustness import GlobalFrequencyRanker


class GlobalFrequencyRankerDenseTest(unittest.TestCase):
    def test_predicts_same_most_frequent_labels_for_all_rows(self) -> None:
        y_train = np.array(
            [
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 1],
            ]
        )

        ranker = GlobalFrequencyRanker().fit(y_train)
        pred = ranker.predict_top_k(n_samples=2, k=3)

        np.testing.assert_array_equal(pred, np.array([[0, 1, 2], [0, 1, 2]]))


if __name__ == "__main__":
    unittest.main()
