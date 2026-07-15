from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline pose inference.")
    parser.add_argument("--config", default="configs/autodl.yaml")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--model", default="sapiens2")
    parser.parse_args()
    raise SystemExit(
        "Baseline runner skeleton is ready. Implement Sapiens2 inference on AutoDL first."
    )


if __name__ == "__main__":
    main()

