from __future__ import annotations

import argparse
from pathlib import Path

from bike_ai.ingestion.inventory import build_inventory, write_inventory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan raw cycling videos and generate an inventory report."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw_videos"),
        help="Root folder containing raw videos.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/registry/video_inventory"),
        help="Output folder for CSV/JSONL/Markdown reports.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=24,
        help="Number of frames sampled per video for basic quality metrics.",
    )
    args = parser.parse_args()

    print(f"Scanning videos: {args.root}")
    records = build_inventory(args.root, sample_frames=args.sample_frames)
    outputs = write_inventory(records, args.out_dir)
    print(f"Found {len(records)} videos.")
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
