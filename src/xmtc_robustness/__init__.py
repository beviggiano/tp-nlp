"""Experimental tools for XMTC propensity-scoring robustness studies."""

from .analysis import (
    compare_predictions,
    evaluate_by_frequency_partition,
    frequency_partitions,
    label_counts,
)
from .metrics import (
    evaluate_predictions,
    inverse_propensity_scores,
    ndcg_at_k,
    precision_at_k,
    propensity_scores,
    ps_ndcg_at_k,
    ps_precision_at_k,
)
from .models.global_frequency import GlobalFrequencyRanker
from .models.retrieval import InstanceKNNRetrievalRanker

__all__ = [
    "GlobalFrequencyRanker",
    "InstanceKNNRetrievalRanker",
    "compare_predictions",
    "evaluate_by_frequency_partition",
    "evaluate_predictions",
    "frequency_partitions",
    "inverse_propensity_scores",
    "label_counts",
    "ndcg_at_k",
    "precision_at_k",
    "propensity_scores",
    "ps_ndcg_at_k",
    "ps_precision_at_k",
]
