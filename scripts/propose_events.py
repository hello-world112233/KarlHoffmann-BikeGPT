from __future__ import annotations

import argparse
from pathlib import Path

from bike_ai.ingestion.auto_events import propose_events_from_inventory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatically propose event clips from full competition videos."
    )
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--clip-seconds", type=float, default=12.0)
    parser.add_argument("--stride-seconds", type=float, default=4.0)
    parser.add_argument("--max-events-per-video", type=int, default=4)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    args = parser.parse_args()

    out = propose_events_from_inventory(
        args.inventory_csv,
        args.out_csv,
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        max_events_per_video=args.max_events_per_video,
        sample_fps=args.sample_fps,
    )
    print(f"auto event index: {out}")


if __name__ == "__main__":
    main()

