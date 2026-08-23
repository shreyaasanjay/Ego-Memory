"""Multimodal reranking of FAISS visual candidates."""

import numpy as np

from egomemory.retrieval.index import MemoryIndex
from egomemory.schema import MemoryEvent, RetrievalResult


MOTION_TERMS = {"walk", "move", "moving", "shake", "shaking", "stir", "stirring", "reach", "reaching"}
SPEECH_TERMS = {"say", "said", "tell", "talk", "explain", "describe", "hear", "sound"}


class MultimodalRetrievalEngine:
    """Retrieve visual candidates with FAISS, then fuse aligned signals."""

    def __init__(self, events: list[MemoryEvent]):
        self.events = events
        self.visual_index = MemoryIndex(events)
        self.text_matrix = np.asarray([event.text_embedding for event in events], dtype=np.float32)
        self.text_matrix /= np.maximum(np.linalg.norm(self.text_matrix, axis=1, keepdims=True), 1e-12)
        self.event_positions = {event.event_id: index for index, event in enumerate(events)}
        motion = np.asarray([event.motion_features.get("motion_energy", 0.0) for event in events], dtype=float)
        low, high = np.percentile(motion, [5, 95])
        self.motion = np.clip((motion - low) / max(high - low, 1e-9), 0.0, 1.0)

    def retrieve(self, query_embedding: list[float], query_text: str, k: int = 5, candidate_k: int = 30) -> list[RetrievalResult]:
        text = query_text.lower()
        asks_about_speech = any(term in text for term in SPEECH_TERMS)
        asks_about_motion = any(term in text for term in MOTION_TERMS)
        # Audio transcript is semantically most valuable for spoken questions;
        # vision remains the primary signal for object/action questions.
        visual_weight = 0.45 if asks_about_speech else 0.65
        transcript_weight = 0.45 if asks_about_speech else 0.25
        motion_weight = 0.10 if asks_about_motion else 0.0
        visual_weight += 0.10 - motion_weight

        query = np.asarray(query_embedding, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        results = []
        for event, visual_score in self.visual_index.search(query.tolist(), candidate_k):
            position = self.event_positions[event.event_id]
            transcript_score = float(self.text_matrix[position] @ query)
            motion_score = float(self.motion[position])
            score = visual_weight * visual_score + transcript_weight * transcript_score + motion_weight * motion_score
            results.append(RetrievalResult(event, score, {"vision": visual_score, "transcript_audio": transcript_score, "motion": motion_score}))
        return sorted(results, key=lambda item: item.score, reverse=True)[:k]
