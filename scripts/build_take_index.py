"""Create timestamped multimodal memory windows for one Ego-Exo4D take.

Uses CLIP for aligned text/image retrieval, simple per-window audio activity,
and average wearable motion energy from the closed-loop trajectory.  The output
contains no raw dataset media and is safe to regenerate locally.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import CLIPModel, CLIPProcessor

from egomemory.preprocessing.take_loader import make_windows
from egomemory.schema import MemoryEvent


def transcript_for_window(segments: list[dict], start: float, end: float) -> str:
    return " ".join(
        segment.get("text", "").strip()
        for segment in segments
        if float(segment.get("end", 0)) >= start and float(segment.get("start", 0)) <= end
    )


def frame_at(video: cv2.VideoCapture, time_seconds: float) -> np.ndarray:
    video.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000)
    ok, frame = video.read()
    if not ok:
        raise RuntimeError(f"Could not read a frame at {time_seconds:.2f}s")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def audio_rms(audio: sf.SoundFile, start: float, end: float) -> float:
    audio.seek(int(start * audio.samplerate))
    samples = audio.read(int((end - start) * audio.samplerate), dtype="float32", always_2d=True)
    return float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0


def motion_by_window(trajectory_path: Path, windows: list[tuple[float, float]], duration: float) -> list[float]:
    columns = ["tracking_timestamp_us", "device_linear_velocity_x_device", "device_linear_velocity_y_device", "device_linear_velocity_z_device"]
    trajectory = pd.read_csv(trajectory_path, usecols=columns)
    timestamps = trajectory["tracking_timestamp_us"].to_numpy(dtype=float)
    relative_seconds = (timestamps - timestamps.min()) / 1_000_000
    # Align the trajectory span to the frame-aligned take duration; exact sync is
    # refined later using the official take timesync metadata.
    relative_seconds *= duration / max(relative_seconds.max(), 1e-9)
    velocity = trajectory[columns[1:]].to_numpy(dtype=float)
    energy = np.linalg.norm(velocity, axis=1)
    return [float(energy[(relative_seconds >= start) & (relative_seconds < end)].mean()) if np.any((relative_seconds >= start) & (relative_seconds < end)) else 0.0 for start, end in windows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("take_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/fair_cooking_05_4"))
    parser.add_argument("--window", type=float, default=8.0)
    parser.add_argument("--stride", type=float, default=4.0)
    args = parser.parse_args()

    video_path = args.take_dir / "frame_aligned_videos/downscaled/448/aria02_214-1.mp4"
    audio_path = args.take_dir / "audio/aria02.wav"
    transcript_path = args.take_dir / "audio/aria02_transcriptions.json"
    trajectory_path = args.take_dir / "trajectory/closed_loop_trajectory.csv"
    for path in (video_path, audio_path, transcript_path, trajectory_path):
        if not path.exists():
            raise FileNotFoundError(path)

    video = cv2.VideoCapture(str(video_path))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    windows = make_windows(duration, args.window, args.stride)
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))["segments"]
    motion = motion_by_window(trajectory_path, windows, duration)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    args.output.mkdir(parents=True, exist_ok=True)

    events: list[MemoryEvent] = []
    with sf.SoundFile(audio_path) as audio, torch.no_grad():
        for index, (start, end) in enumerate(windows):
            image = frame_at(video, (start + end) / 2)
            model_inputs = processor(images=image, return_tensors="pt").to(device)
            # Transformers v5 returns vision-token states from get_image_features;
            # pool and project explicitly to CLIP's shared 512-D embedding space.
            vision_output = model.vision_model(pixel_values=model_inputs["pixel_values"])
            embedding = model.visual_projection(vision_output.pooler_output)[0].cpu().numpy()
            embedding = (embedding / np.linalg.norm(embedding)).tolist()
            events.append(MemoryEvent(
                event_id=f"window_{index:03d}", start_time=round(start, 3), end_time=round(end, 3),
                narration=transcript_for_window(transcript, start, end), visual_embedding=embedding,
                audio_features={"rms": audio_rms(audio, start, end)},
                motion_features={"motion_energy": motion[index]}, source_video=str(video_path),
            ))
            print(f"Indexed {index + 1}/{len(windows)} windows", end="\r")
    video.release()
    # The transcription is the audio-language representation for each aligned
    # memory window. Batch encoding makes it practical for the whole take.
    for offset in range(0, len(events), 16):
        batch = events[offset:offset + 16]
        inputs = processor(text=[event.narration or "[no speech]" for event in batch], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_output = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            text_vectors = model.text_projection(text_output.pooler_output).cpu().numpy()
        for event, vector in zip(batch, text_vectors):
            vector /= np.linalg.norm(vector)
            event.text_embedding = vector.tolist()
    (args.output / "events.json").write_text(json.dumps([event.to_dict() for event in events], indent=2), encoding="utf-8")
    np.save(args.output / "visual_embeddings.npy", np.asarray([event.visual_embedding for event in events], dtype=np.float32))
    (args.output / "manifest.json").write_text(json.dumps({"take": args.take_dir.name, "video": str(video_path), "duration_seconds": duration, "window_seconds": args.window, "stride_seconds": args.stride, "embedding_model": model_name, "windows": len(events)}, indent=2), encoding="utf-8")
    print(f"\nWrote {len(events)} memory windows to {args.output}")


if __name__ == "__main__":
    main()
