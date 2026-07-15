from __future__ import annotations

from pathlib import Path

import cv2

from bike_ai.common.config import ensure_dir


def probe_video(path: str | Path) -> dict[str, float | int | str]:
    """Read basic video metadata with OpenCV."""
    video_path = Path(path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "fps": float(fps),
        "frame_count": frames,
        "width": width,
        "height": height,
        "duration_sec": float(frames / fps) if fps else 0.0,
        "resolution": f"{width}x{height}",
    }


def extract_frames(video_path: str | Path, out_dir: str | Path, fps: float, image_ext: str = "jpg") -> int:
    """Extract frames at a fixed target fps."""
    video_path = Path(video_path)
    out = ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps))
    saved = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            cv2.imwrite(str(out / f"frame_{saved:06d}.{image_ext}"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved

