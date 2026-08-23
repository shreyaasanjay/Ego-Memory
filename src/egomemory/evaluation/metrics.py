"""Evaluation metrics reported in the project results."""

from egomemory.schema import RetrievalResult


def recall_at_k(results: list[RetrievalResult], target_event_id: str, k: int = 5) -> float:
    return float(any(result.event.event_id == target_event_id for result in results[:k]))


def temporal_localization_error(results: list[RetrievalResult], target_time: float) -> float:
    if not results:
        raise ValueError("Cannot evaluate an empty result list.")
    return abs(results[0].event.start_time - target_time)
