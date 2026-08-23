"""Run a data-free retrieval check before processing a real take."""

from egomemory.retrieval.engine import QueryContext, RetrievalEngine
from egomemory.retrieval.index import MemoryIndex
from egomemory.schema import MemoryEvent


events = [
    MemoryEvent("fridge", 6.0, 14.0, "Opened refrigerator and picked up milk", [1.0, 0.0]),
    MemoryEvent("counter", 14.0, 22.0, "Poured milk at counter", [0.7, 0.3]),
]
engine = RetrievalEngine(MemoryIndex(events), {"vision": 1.0})
for result in engine.retrieve(QueryContext([1.0, 0.0], text="When did I get milk?")):
    print(f"{result.event.start_time:05.1f}s  {result.event.narration}  score={result.score:.3f}")
