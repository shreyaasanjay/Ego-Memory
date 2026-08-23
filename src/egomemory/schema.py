"""Stable data structures shared by indexing, retrieval, and evaluation."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MemoryEvent:
    event_id: str
    start_time: float
    end_time: float
    narration: str = ""
    visual_embedding: list[float] = field(default_factory=list)
    audio_features: dict[str, float] = field(default_factory=dict)
    motion_features: dict[str, float] = field(default_factory=dict)
    location: str | None = None
    source_video: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    event: MemoryEvent
    score: float
    component_scores: dict[str, float]
