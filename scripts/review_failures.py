from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and tag model failure cases.")
    parser.add_argument("--inference-dir", required=True)
    parser.parse_args()
    raise SystemExit("Failure review UI/CLI will be implemented after baseline output exists.")


if __name__ == "__main__":
    main()

