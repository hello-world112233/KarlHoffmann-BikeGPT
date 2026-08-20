#!/usr/bin/env python3
"""Sync multi-camera videos by audio cross-correlation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bike_ai.sync.audio_sync import sync_videos_by_audio, write_aligned_clips


def parse_video_args(pairs: list[str]) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"Expected name=path, got {item!r}")
        name, path = item.split("=", 1)
        videos[name] = Path(path)
    return videos


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--video",
        action="append",
        required=True,
        help="Camera video as name=path, e.g. --video side=a.mp4 --video front=b.mp4",
    )
    p.add_argument("--reference", default=None, help="Reference camera name (default: first)")
    p.add_argument("--out", required=True, help="Output sync JSON path")
    p.add_argument("--write-aligned-dir", default=None, help="Optional dir to write trimmed clips")
    p.add_argument("--duration-sec", type=float, default=None)
    p.add_argument("--max-lag-sec", type=float, default=30.0)
    args = p.parse_args()

    videos = parse_video_args(args.video)
    sync = sync_videos_by_audio(
        videos, reference=args.reference, max_lag_sec=args.max_lag_sec
    )
    out = Path(args.out)
    sync.save(out)
    print(json.dumps(sync.to_dict(), indent=2, ensure_ascii=False))

    if args.write_aligned_dir:
        paths = write_aligned_clips(
            videos, sync, args.write_aligned_dir, duration_sec=args.duration_sec
        )
        print("aligned:", {k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
