"""Render an athlete-only skeleton overlay for QA of Step-0 selection.

Draws the single selected athlete instance per frame on the raw frames. Can
export sampled QA images and/or an MP4.

Usage:
  python scripts/render_athlete_overlay.py \
      --athlete .../athlete/athlete_predictions.json \
      --frames-dir .../frames/scene1_dji_static_side_10fps \
      --out-dir .../athlete/overlay \
      [--sample 300 800 1400] [--video out.mp4 --fps 10] [--kpt-thr 0.3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bike_ai.tracking.keypoints import SKELETON_LINKS  # noqa: E402

GREEN = (0, 255, 0)
ORANGE = (0, 128, 255)
BLUE = (255, 128, 51)


def draw_instance(img: np.ndarray, ins: dict, kpt_thr: float) -> None:
    kpts = np.asarray(ins["keypoints"], dtype=float)
    scores = np.asarray(ins["keypoint_scores"], dtype=float)
    x1, y1, x2, y2 = (int(v) for v in ins["bbox"])
    cv2.rectangle(img, (x1, y1), (x2, y2), BLUE, 3)
    for a, b in SKELETON_LINKS:
        if scores[a] > kpt_thr and scores[b] > kpt_thr:
            pa = tuple(np.round(kpts[a]).astype(int))
            pb = tuple(np.round(kpts[b]).astype(int))
            cv2.line(img, pa, pb, GREEN, 3)
    for i in range(min(23, len(kpts))):
        if scores[i] > kpt_thr:
            p = tuple(np.round(kpts[i]).astype(int))
            cv2.circle(img, p, 4, ORANGE, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render athlete-only overlay.")
    parser.add_argument("--athlete", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample", type=int, nargs="*", default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--scale", type=float, default=1.0, help="output scale for the video")
    args = parser.parse_args()

    data = json.loads(args.athlete.read_text())
    frames = data["frames"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def render(fi: int) -> np.ndarray | None:
        frame = frames[fi]
        img_path = args.frames_dir / frame["image_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        for ins in frame["instances"]:
            draw_instance(img, ins, args.kpt_thr)
        present = bool(frame["instances"])
        label = f"frame {fi}  athlete={'YES' if present else 'MISSING'}"
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 255) if not present else (0, 255, 0), 3)
        return img

    if args.sample is not None:
        for fi in args.sample:
            img = render(fi)
            if img is not None:
                out = args.out_dir / f"qa_frame_{fi:06d}.jpg"
                cv2.imwrite(str(out), img)
                print(f"wrote {out}")

    if args.video is not None:
        h, w = data["image_size"]
        ow, oh = int(w * args.scale), int(h * args.scale)
        writer = cv2.VideoWriter(
            str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (ow, oh)
        )
        for fi in range(len(frames)):
            img = render(fi)
            if img is not None:
                if args.scale != 1.0:
                    img = cv2.resize(img, (ow, oh))
                writer.write(img)
            if fi % 200 == 0:
                print(f"  video {fi}/{len(frames)}")
        writer.release()
        print(f"wrote video {args.video}")


if __name__ == "__main__":
    main()
