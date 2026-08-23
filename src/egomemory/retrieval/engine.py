"""Rerank visual candidates with time, audio, motion, and location evidence."""

from dataclasses import dataclass

from egomemory.retrieval.index import MemoryIndex
from egomemory.schema import RetrievalResult


@dataclass
class QueryContext:
    embedding: list[float]
    text: str = ""
    reference_time: float | None = None
    relation: str | None = None  # "before", "after", or "near"
    location: str | None = None


class RetrievalEngine:
    def __init__(self, index: MemoryIndex, weights: dict[str, float]):
        self.index = index
        self.weights = weights

    def retrieve(self, query: QueryContext, k: int = 5, candidate_k: int = 25) -> list[RetrievalResult]:
        candidates = self.index.search(query.embedding, candidate_k)
        results = []
        for event, vision_score in candidates:
            temporal = self._temporal_score(event.start_time, query.reference_time, query.relation)
            spatial = 1.0 if query.location and event.location == query.location else 0.0
            text = query.text.lower()
            asks_about_audio = any(token in text for token in ("say", "said", "tell", "talk", "hear", "sound"))
            asks_about_motion = any(token in text for token in ("walk", "move", "reach", "stir"))
            audio = float(event.audio_features.get("speech_activity", 0.0)) if asks_about_audio else 0.0
            motion = float(event.motion_features.get("motion_energy", 0.0)) if asks_about_motion else 0.0
            score = (self.weights.get("vision", 1.0) * vision_score + self.weights.get("temporal", 0.0) * temporal + self.weights.get("spatial", 0.0) * spatial + self.weights.get("audio", 0.0) * audio + self.weights.get("motion", 0.0) * motion)
            results.append(RetrievalResult(event, score, {"vision": vision_score, "temporal": temporal, "spatial": spatial, "audio": audio, "motion": motion}))
        return sorted(results, key=lambda item: item.score, reverse=True)[:k]

    @staticmethod
    def _temporal_score(event_time: float, reference_time: float | None, relation: str | None) -> float:
        if reference_time is None or relation is None:
            return 0.0
        delta = event_time - reference_time
        if relation == "before" and delta < 0:
            return 1.0 / (1.0 + abs(delta))
        if relation == "after" and delta > 0:
            return 1.0 / (1.0 + abs(delta))
        if relation == "near":
            return 1.0 / (1.0 + abs(delta))
        return 0.0
