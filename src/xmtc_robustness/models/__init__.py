"""Rankers used in the XMTC experiments."""

from .global_frequency import GlobalFrequencyRanker
from .retrieval import InstanceKNNRetrievalRanker

__all__ = ["GlobalFrequencyRanker", "InstanceKNNRetrievalRanker"]
