from __future__ import annotations

import argparse
from pathlib import Path

from bike_ai.ingestion.video import extract_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a registered/raw video.")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--image-ext", default="jpg")
    args = parser.parse_args()

    n = extract_frames(args.video_path, args.out_dir, args.fps, args.image_ext)
    print(f"saved {n} frames to {args.out_dir}")


if __name__ == "__main__":
    main()

