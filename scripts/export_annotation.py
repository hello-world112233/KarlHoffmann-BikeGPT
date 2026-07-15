from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frames to an annotation task.")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--format", choices=["cvat", "label-studio"], default="cvat")
    parser.parse_args()
    raise SystemExit("Annotation export will be implemented after hard-case frame selection.")


if __name__ == "__main__":
    main()

