"""Search a built EgoMemory take index with a natural-language query."""

import argparse
import json
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

from egomemory.retrieval.multimodal import MultimodalRetrievalEngine
from egomemory.schema import MemoryEvent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=Path("results/fair_cooking_05_4/events.json"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    events = [MemoryEvent(**event) for event in json.loads(args.index.read_text(encoding="utf-8"))]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        inputs = processor(text=[args.query], return_tensors="pt", padding=True).to(device)
        # Pool and project explicitly to the same CLIP space used for images.
        text_output = model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        embedding = model.text_projection(text_output.pooler_output)[0].cpu().numpy().tolist()

    engine = MultimodalRetrievalEngine(events)
    for rank, result in enumerate(engine.retrieve(embedding, args.query, k=args.top_k), start=1):
        event = result.event
        scores = result.component_scores
        print(f"{rank}. {event.start_time:06.1f}–{event.end_time:06.1f}s  score={result.score:.3f} (vision={scores['vision']:.3f}, transcript={scores['transcript_audio']:.3f}, motion={scores['motion']:.3f})")
        print(f"   {event.narration or '[no overlapping transcript]'}")


if __name__ == "__main__":
    main()
