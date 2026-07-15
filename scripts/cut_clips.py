from __future__ import annotations

import argparse
from pathlib import Path

from bike_ai.ingestion.events import cut_clips_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut event clips from full competition videos.")
    parser.add_argument("--event-csv", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode clips for accurate cuts. Slower but more reliable than stream copy.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = cut_clips_from_csv(
        args.event_csv,
        args.out_root,
        reencode=args.reencode,
        overwrite=args.overwrite,
    )
    for p in paths:
        print(p)
    print(f"cut {len(paths)} clips")


if __name__ == "__main__":
    main()

