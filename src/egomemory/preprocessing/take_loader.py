"""Locate downloaded Ego-Exo4D take assets without assuming a fixed release layout."""

from pathlib import Path


MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".vrs"}


def inspect_take(take_dir: str | Path) -> dict[str, list[str]]:
    """Return the available modalities for a single downloaded take directory."""
    root = Path(take_dir)
    if not root.exists():
        raise FileNotFoundError(f"Take directory does not exist: {root}")

    assets: dict[str, list[str]] = {"video": [], "audio": [], "imu": [], "trajectory": [], "annotations": []}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        relative = str(path.relative_to(root))
        if path.suffix.lower() in MEDIA_EXTENSIONS:
            assets["video"].append(relative)
        elif path.suffix.lower() in {".wav", ".flac", ".mp3", ".aac"}:
            assets["audio"].append(relative)
        elif "imu" in name:
            assets["imu"].append(relative)
        elif any(token in name for token in ("trajectory", "point_cloud", "pose", "gps")):
            assets["trajectory"].append(relative)
        elif path.suffix.lower() in {".json", ".jsonl", ".csv"}:
            assets["annotations"].append(relative)
    return assets


def make_windows(duration_seconds: float, window_seconds: float = 8.0, stride_seconds: float = 4.0) -> list[tuple[float, float]]:
    """Generate overlapping, timestamped memory windows."""
    if duration_seconds <= 0 or window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("Duration, window size, and stride must be positive.")
    windows = []
    start = 0.0
    while start < duration_seconds:
        windows.append((start, min(start + window_seconds, duration_seconds)))
        start += stride_seconds
    return windows
