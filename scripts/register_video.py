from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from bike_ai.ingestion.video import probe_video
from bike_ai.registry.schema import CameraView, OcclusionLevel, SceneType, VideoRecord
from bike_ai.registry.store import JsonRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a cycling training video.")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--athlete-id", required=True)
    parser.add_argument("--scene", choices=[x.value for x in SceneType], required=True)
    parser.add_argument("--camera-view", choices=[x.value for x in CameraView], required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--location", default="Changxing")
    parser.add_argument("--occlusion", choices=[x.value for x in OcclusionLevel], default="none")
    parser.add_argument("--lighting", default=None)
    parser.add_argument("--coach-note", default=None)
    parser.add_argument("--registry-dir", default="data/registry")
    args = parser.parse_args()

    meta = probe_video(args.video_path)
    record = VideoRecord(
        video_id=args.video_id,
        file_path=args.video_path,
        athlete_id=args.athlete_id,
        session_date=date.fromisoformat(args.date),
        location=args.location,
        scene=SceneType(args.scene),
        camera_view=CameraView(args.camera_view),
        fps=float(meta["fps"]),
        resolution=str(meta["resolution"]),
        duration_sec=float(meta["duration_sec"]),
        occlusion_level=OcclusionLevel(args.occlusion),
        lighting=args.lighting,
        coach_note=args.coach_note,
    )
    out = JsonRegistry(args.registry_dir).save(record)
    print(f"registered: {out}")


if __name__ == "__main__":
    main()

