from egomemory.evaluation.metrics import recall_at_k, temporal_localization_error
from egomemory.preprocessing.take_loader import make_windows
from egomemory.retrieval.engine import QueryContext, RetrievalEngine
from egomemory.retrieval.index import MemoryIndex
from egomemory.schema import MemoryEvent


def test_windows_cover_duration():
    assert make_windows(10, window_seconds=5, stride_seconds=5) == [(0.0, 5.0), (5.0, 10.0)]


def test_temporal_retrieval_and_metrics():
    events = [
        MemoryEvent("a", 10, 14, visual_embedding=[1, 0]),
        MemoryEvent("b", 20, 24, visual_embedding=[0.9, 0.1]),
    ]
    engine = RetrievalEngine(MemoryIndex(events), {"vision": 0.5, "temporal": 0.5})
    results = engine.retrieve(QueryContext([1, 0], reference_time=20, relation="before"), k=2)
    assert results[0].event.event_id == "a"
    assert recall_at_k(results, "a") == 1.0
    assert temporal_localization_error(results, 10) == 0
