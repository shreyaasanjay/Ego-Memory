"""A FAISS-backed visual index with a NumPy fallback for small local runs."""

import numpy as np

from egomemory.schema import MemoryEvent


class MemoryIndex:
    def __init__(self, events: list[MemoryEvent]):
        if not events:
            raise ValueError("Cannot build an index with no memory events.")
        self.events = events
        self.matrix = np.asarray([event.visual_embedding for event in events], dtype=np.float32)
        if self.matrix.ndim != 2 or self.matrix.shape[1] == 0:
            raise ValueError("Every event needs a non-empty visual embedding.")
        self.matrix /= np.maximum(np.linalg.norm(self.matrix, axis=1, keepdims=True), 1e-12)
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.matrix.shape[1])
            self._index.add(self.matrix)
        except ImportError:
            self._index = None

    def search(self, query_embedding: list[float], k: int = 5) -> list[tuple[MemoryEvent, float]]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.matrix.shape[1]:
            raise ValueError("Query and event embeddings have different dimensions.")
        query /= max(float(np.linalg.norm(query)), 1e-12)
        k = min(k, len(self.events))
        if self._index is not None:
            scores, indices = self._index.search(query, k)
            return [(self.events[i], float(score)) for i, score in zip(indices[0], scores[0])]
        scores = (self.matrix @ query.T).ravel()
        indices = np.argsort(-scores)[:k]
        return [(self.events[i], float(scores[i])) for i in indices]
