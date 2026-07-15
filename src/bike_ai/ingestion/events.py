from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path


EVENT_TYPES = [
    "start",
    "first_lap_acceleration",
    "straight",
    "curve",
    "sprint",
    "occlusion",
    "multi_rider",
    "far_view",
    "hard_case",
    "other",
]


@dataclass(frozen=True)
class EventClip:
    source_video_id: str
    source_path: str
    clip_id: str
    start_time: str
    end_time: str
    event_type: str
    camera_view: str
    quality: str
    priority: int
    note: str = ""


def write_event_template(inventory_csv: str | Path, out_csv: str | Path) -> Path:
    """Create a manual event-index template from a video inventory CSV."""
    inventory_csv = Path(inventory_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with inventory_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

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
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            video_id = row["video_id"]
            writer.writerow(
                {
                    "source_video_id": video_id,
                    "source_path": row["path"],
                    "clip_id": f"{video_id}_start_001",
                    "start_time": "00:00:00",
                    "end_time": "00:00:12",
                    "event_type": "start",
                    "camera_view": "unknown",
                    "quality": "unknown",
                    "priority": "1",
                    "note": "Edit this row after watching the full video.",
                }
            )
    return out_csv


def read_event_csv(path: str | Path) -> list[EventClip]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    clips: list[EventClip] = []
    for row in rows:
        if not row.get("source_path") or not row.get("clip_id"):
            continue
        if row.get("event_type") not in EVENT_TYPES:
            raise ValueError(f"Unknown event_type for {row.get('clip_id')}: {row.get('event_type')}")
        clips.append(
            EventClip(
                source_video_id=row["source_video_id"],
                source_path=row["source_path"],
                clip_id=row["clip_id"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                event_type=row["event_type"],
                camera_view=row.get("camera_view", "unknown"),
                quality=row.get("quality", "unknown"),
                priority=int(row.get("priority") or 3),
                note=row.get("note", ""),
            )
        )
    return clips


def cut_clip(
    clip: EventClip,
    out_root: str | Path,
    *,
    reencode: bool = False,
    overwrite: bool = False,
) -> Path:
    out_dir = Path(out_root) / clip.event_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip.clip_id}.mp4"
    if out_path.exists() and not overwrite:
        return out_path

    cmd = ["ffmpeg", "-y" if overwrite else "-n", "-ss", clip.start_time, "-to", clip.end_time]
    cmd += ["-i", clip.source_path]
    if reencode:
        cmd += ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-c:a", "aac"]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(out_path))

    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {clip.clip_id}\nCMD: {' '.join(cmd)}\nSTDERR:\n{proc.stderr}"
        )
    return out_path


def cut_clips_from_csv(
    event_csv: str | Path,
    out_root: str | Path,
    *,
    reencode: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    clips = read_event_csv(event_csv)
    return [
        cut_clip(clip, out_root, reencode=reencode, overwrite=overwrite)
        for clip in clips
    ]

