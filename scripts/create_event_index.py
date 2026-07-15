from __future__ import annotations

import argparse
from pathlib import Path

from bike_ai.ingestion.events import write_event_template


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a manual event-index CSV template from video_inventory.csv."
    )
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    out = write_event_template(args.inventory_csv, args.out_csv)
    print(f"event index template: {out}")
    print("Open this CSV, edit start_time/end_time/event_type rows, then run cut_clips.py.")


if __name__ == "__main__":
    main()

