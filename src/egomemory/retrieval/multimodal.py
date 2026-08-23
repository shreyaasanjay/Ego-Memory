"""Multimodal reranking of FAISS visual candidates."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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
        self.transcript_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.transcript_tfidf = self.transcript_vectorizer.fit_transform(
            [event.narration or "no speech" for event in events]
        )
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
        # Candidate generation must cover both modalities. Restricting candidates
        # to visual FAISS search alone can hide a correct speech-only moment.
        visual_candidates = self.visual_index.search(query.tolist(), candidate_k)
        semantic_transcript_scores = self.text_matrix @ query
        lexical_transcript_scores = (self.transcript_tfidf @ self.transcript_vectorizer.transform([query_text]).T).toarray().ravel()
        # CLIP supplies paraphrase tolerance; TF-IDF preserves precise words in
        # the often-noisy automatic transcript.
        transcript_scores = 0.4 * self._normalize(semantic_transcript_scores) + 0.6 * lexical_transcript_scores
        transcript_positions = np.argsort(-transcript_scores)[:candidate_k]
        candidate_positions = {self.event_positions[event.event_id] for event, _score in visual_candidates}
        candidate_positions.update(int(position) for position in transcript_positions)

        results = []
        for position in candidate_positions:
            event = self.events[position]
            visual_score = float(self.visual_index.matrix[position] @ query)
            transcript_score = float(transcript_scores[position])
            motion_score = float(self.motion[position])
            score = visual_weight * visual_score + transcript_weight * transcript_score + motion_weight * motion_score
            results.append(RetrievalResult(event, score, {"vision": visual_score, "transcript_audio": transcript_score, "motion": motion_score}))
        return sorted(results, key=lambda item: item.score, reverse=True)[:k]

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        low, high = values.min(), values.max()
        return (values - low) / max(high - low, 1e-12)
