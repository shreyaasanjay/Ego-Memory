"""Run fixed-weight modality ablations over the labeled EgoMemory queries."""

import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from egomemory.schema import MemoryEvent


ROOT = Path(__file__).parents[1]
EVENTS_PATH = ROOT / "results/fair_cooking_05_4/events.json"
QUERIES_PATH = ROOT / "configs/evaluation_queries.json"
OUT_DIR = ROOT / "results/ablation"

# Chosen before running the experiment and unchanged for every query.
CONFIGURATIONS = {
    "vision": {"vision": 1.0, "audio": 0.0, "motion": 0.0},
    "audio_transcript": {"vision": 0.0, "audio": 1.0, "motion": 0.0},
    "motion": {"vision": 0.0, "audio": 0.0, "motion": 1.0},
    "vision_audio": {"vision": 0.5, "audio": 0.5, "motion": 0.0},
    "vision_motion": {"vision": 0.5, "audio": 0.0, "motion": 0.5},
    "audio_motion": {"vision": 0.0, "audio": 0.5, "motion": 0.5},
    "full_multimodal": {"vision": 0.4, "audio": 0.4, "motion": 0.2},
}


def normalize(values: np.ndarray) -> np.ndarray:
    low, high = values.min(), values.max()
    return (values - low) / max(high - low, 1e-12)


def overlaps(event: MemoryEvent, start: float, end: float) -> bool:
    return event.end_time >= start and event.start_time <= end


def query_embedding(query: str, model, processor, device: str) -> np.ndarray:
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_output = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        vector = model.text_projection(text_output.pooler_output)[0].cpu().numpy()
    return vector / np.linalg.norm(vector)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    events = [MemoryEvent(**item) for item in json.loads(EVENTS_PATH.read_text(encoding="utf-8"))]
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    visual = np.asarray([event.visual_embedding for event in events], dtype=float)
    text = np.asarray([event.text_embedding for event in events], dtype=float)
    visual /= np.maximum(np.linalg.norm(visual, axis=1, keepdims=True), 1e-12)
    text /= np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    motion = normalize(np.asarray([event.motion_features.get("motion_energy", 0.0) for event in events], dtype=float))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    rows = []
    for query in queries:
        vector = query_embedding(query["query"], model, processor, device)
        signals = {"vision": normalize(visual @ vector), "audio": normalize(text @ vector), "motion": motion}
        target_start, target_end = float(query["start_time"]), float(query["end_time"])
        target_center = (target_start + target_end) / 2
        for configuration, weights in CONFIGURATIONS.items():
            scores = sum(weights[name] * signals[name] for name in weights)
            order = np.argsort(-scores)
            top_five = [events[index] for index in order[:5]]
            top_one = events[order[0]]
            rows.append({
                "query_id": query["query_id"], "query": query["query"], "query_modality": query["modality"],
                "validation_status": query.get("validation_status", "unspecified"),
                "configuration": configuration, "recall_at_5": int(any(overlaps(event, target_start, target_end) for event in top_five)),
                "temporal_error_seconds": round(abs((top_one.start_time + top_one.end_time) / 2 - target_center), 3),
                "top1_start": top_one.start_time, "top1_end": top_one.end_time,
                "top5_windows": "; ".join(f"{event.start_time:.1f}-{event.end_time:.1f}" for event in top_five),
            })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for configuration in CONFIGURATIONS:
        selected = [row for row in rows if row["configuration"] == configuration]
        summary.append({
            "configuration": configuration,
            "queries": len(selected),
            "recall_at_5": round(sum(row["recall_at_5"] for row in selected) / len(selected), 3),
            "mean_temporal_error_seconds": round(sum(row["temporal_error_seconds"] for row in selected) / len(selected), 3),
        })
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "summary.json").write_text(json.dumps({"fixed_weights": CONFIGURATIONS, "queries_file": str(args.queries), "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
