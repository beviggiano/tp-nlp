from __future__ import annotations

from typing import Protocol

import numpy as np


class TopKRanker(Protocol):
    """Common protocol for rankers that emit top-k label ids.

    Keeping a small interface makes the experimental pipeline replaceable:
    a text-aware model and the naive frequency baseline can be evaluated by the
    same metric code, which is the core comparison in the robustness study.
    """

    def predict_top_k(self, *args: object, k: int, **kwargs: object) -> np.ndarray:
        """Return an integer array with shape `(n_samples, k)`."""
