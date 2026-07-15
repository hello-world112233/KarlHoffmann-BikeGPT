from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bike_ai.ingestion.events import EVENT_TYPES


@dataclass(frozen=True)
class TimelineSample:
    t: float
    motion: float
    blur: float
    brightness: float


@dataclass(frozen=True)
class ProposedEvent:
    source_video_id: str
    source_path: str
    clip_id: str
    start_time: str
    end_time: str
    event_type: str
    camera_view: str
    quality: str
    priority: int
    note: str


def seconds_to_time(value: float) -> str:
    value = max(0.0, float(value))
    minutes, seconds = divmod(value, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def sample_timeline(
    video_path: str | Path,
    *,
    sample_fps: float = 2.0,
    resize_width: int = 320,
) -> list[TimelineSample]:
    import cv2

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return []

    duration = frame_count / fps
    times = np.arange(0.0, duration, 1.0 / sample_fps)
    prev_gray: np.ndarray | None = None
    samples: list[TimelineSample] = []

    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000.0))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            gray_small = cv2.resize(gray, (resize_width, max(1, int(h * scale))))
        else:
            gray_small = gray

        motion = 0.0
        if prev_gray is not None and prev_gray.shape == gray_small.shape:
            motion = float(np.mean(cv2.absdiff(prev_gray, gray_small)))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        samples.append(TimelineSample(t=float(t), motion=motion, blur=blur, brightness=brightness))
        prev_gray = gray_small

    cap.release()
    return samples


def _window_score(samples: list[TimelineSample], start: float, end: float) -> dict[str, float]:
    win = [s for s in samples if start <= s.t < end]
    if not win:
        return {"motion": 0.0, "blur": 0.0, "brightness": 0.0}
    return {
        "motion": float(np.mean([s.motion for s in win])),
        "blur": float(np.mean([s.blur for s in win])),
        "brightness": float(np.mean([s.brightness for s in win])),
    }


def _overlaps(a: tuple[float, float], b: tuple[float, float], min_overlap: float = 0.35) -> bool:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    inter = max(0.0, right - left)
    denom = max(1e-6, min(a[1] - a[0], b[1] - b[0]))
    return inter / denom >= min_overlap


def propose_events_for_video(
    source_video_id: str,
    source_path: str | Path,
    *,
    clip_seconds: float = 12.0,
    stride_seconds: float = 4.0,
    max_events: int = 4,
    sample_fps: float = 2.0,
) -> list[ProposedEvent]:
    path = Path(source_path)
    samples = sample_timeline(path, sample_fps=sample_fps)
    if not samples:
        return []

    duration = samples[-1].t
    candidates: list[tuple[float, float, str, float, str]] = []

    # Almost every competition clip contains a useful early start/launch segment.
    candidates.append((0.0, min(clip_seconds, duration), "start", 9999.0, "auto_start_window"))

    starts = np.arange(0.0, max(0.0, duration - clip_seconds), stride_seconds)
    window_stats = []
    for start in starts:
        end = min(duration, float(start + clip_seconds))
        stat = _window_score(samples, float(start), end)
        window_stats.append((float(start), end, stat))

    if window_stats:
        motions = np.array([w[2]["motion"] for w in window_stats])
        blurs = np.array([w[2]["blur"] for w in window_stats])
        motion_hi = float(np.percentile(motions, 80))
        blur_lo = float(np.percentile(blurs, 25))

        for start, end, stat in window_stats:
            if stat["motion"] >= motion_hi:
                candidates.append(
                    (
                        start,
                        end,
                        "sprint",
                        stat["motion"],
                        f"auto_high_motion motion={stat['motion']:.2f}",
                    )
                )
            if stat["blur"] <= blur_lo and stat["motion"] > 0:
                candidates.append(
                    (
                        start,
                        end,
                        "hard_case",
                        1000.0 - stat["blur"] + stat["motion"],
                        f"auto_possible_blur blur={stat['blur']:.1f} motion={stat['motion']:.2f}",
                    )
                )

    candidates = sorted(candidates, key=lambda x: x[3], reverse=True)
    selected: list[tuple[float, float, str, float, str]] = []
    for cand in candidates:
        interval = (cand[0], cand[1])
        if any(_overlaps(interval, (s[0], s[1])) for s in selected):
            continue
        selected.append(cand)
        if len(selected) >= max_events:
            break

    events: list[ProposedEvent] = []
    for idx, (start, end, event_type, score, note) in enumerate(selected, start=1):
        if event_type not in EVENT_TYPES:
            event_type = "other"
        clip_id = f"{source_video_id}_{event_type}_{idx:03d}"
        quality = "auto_candidate"
        priority = 1 if event_type == "start" else 2
        events.append(
            ProposedEvent(
                source_video_id=source_video_id,
                source_path=str(path),
                clip_id=clip_id,
                start_time=seconds_to_time(start),
                end_time=seconds_to_time(end),
                event_type=event_type,
                camera_view="unknown",
                quality=quality,
                priority=priority,
                note=f"{note}; score={score:.2f}",
            )
        )
    return events


def propose_events_from_inventory(
    inventory_csv: str | Path,
    out_csv: str | Path,
    *,
    clip_seconds: float = 12.0,
    stride_seconds: float = 4.0,
    max_events_per_video: int = 4,
    sample_fps: float = 2.0,
) -> Path:
    with Path(inventory_csv).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    all_events: list[ProposedEvent] = []
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] proposing events for {row['video_id']}")
        all_events.extend(
            propose_events_for_video(
                row["video_id"],
                row["path"],
                clip_seconds=clip_seconds,
                stride_seconds=stride_seconds,
                max_events=max_events_per_video,
                sample_fps=sample_fps,
            )
        )

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_video_id",
        "source_path",
        "clip_id",
        "start_time",
        "end_time",
        "event_type",
        "camera_view",
        "quality",
        "priority",
        "note",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in all_events:
            writer.writerow(event.__dict__)
    return out
