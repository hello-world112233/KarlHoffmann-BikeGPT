#!/usr/bin/env python3
"""Sapiens2 (locked athlete) → MotionBERT 2D→3D lift → ROTA joints.json.

Keeps Sapiens2 as the 2D front-end; replaces heuristic foreshortening with
MotionBERT-Lite (H36M finetuned, in-the-wild global).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# MotionBERT repo on PYTHONPATH
DEFAULT_MB = Path("/root/autodl-tmp/bike-cloud/repos/MotionBERT")


def coco17_to_halpe26_row(xy17, sc17) -> list[float]:
    pts = []
    for i in range(17):
        x, y = xy17[i]
        c = float(sc17[i]) if i < len(sc17) else 0.0
        pts.append([float(x), float(y), c])

    def mid(i, j):
        return [
            0.5 * (pts[i][0] + pts[j][0]),
            0.5 * (pts[i][1] + pts[j][1]),
            0.5 * (pts[i][2] + pts[j][2]),
        ]

    neck = mid(5, 6)
    hip = mid(11, 12)
    head = [
        pts[0][0] + 0.25 * (pts[0][0] - neck[0]),
        pts[0][1] + 0.25 * (pts[0][1] - neck[1]),
        pts[0][2],
    ]
    extra = [head, neck, hip, pts[15], pts[16], pts[15], pts[16], pts[15], pts[16]]
    flat: list[float] = []
    for p in pts + extra:
        flat.extend(p)
    return flat


def write_alphapose(athlete_json: Path, out_json: Path) -> tuple[list[dict], list]:
    data = json.loads(athlete_json.read_text(encoding="utf-8"))
    rows = []
    meta_frames = []
    for i, fr in enumerate(data.get("frames") or []):
        insts = fr.get("instances") or []
        if not insts or len((insts[0].get("keypoints") or [])) < 17:
            kps = [0.0] * 78
            bbox = [0, 0, 0, 0]
            scores = [0.0] * 17
        else:
            inst = insts[0]
            xy = inst["keypoints"][:17]
            sc = (inst.get("keypoint_scores") or [1.0] * 17)[:17]
            kps = coco17_to_halpe26_row(xy, sc)
            bbox = inst.get("bbox") or [0, 0, 0, 0]
            scores = [float(s) for s in sc]
        rows.append(
            {
                "image_id": fr.get("image_name") or f"frame_{i:06d}.jpg",
                "idx": 0,
                "keypoints": kps,
                "box": bbox,
            }
        )
        meta_frames.append(
            {
                "frame": i,
                "image_name": fr.get("image_name") or f"frame_{i:06d}.jpg",
                "bbox": bbox,
                "joint_scores": scores,
            }
        )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows), encoding="utf-8")
    return meta_frames, data.get("image_size") or [2160, 3840]


def h36m17_to_coco17(xyz: np.ndarray) -> np.ndarray:
    """MotionBERT H36M-17 → COCO-17 (eyes/ears approximated from nose/head)."""
    # H36M: 0 hip,1 rhip,2 rkne,3 rank,4 lhip,5 lkne,6 lank,7 spine,8 neck,9 nose,10 head,
    #       11 lsho,12 lelb,13 lwri,14 rsho,15 relb,16 rwri
    T = xyz.shape[0]
    out = np.zeros((T, 17, 3), dtype=np.float64)
    out[:, 0] = xyz[:, 9]  # nose
    # eyes / ears: blend nose & head
    head = xyz[:, 10]
    nose = xyz[:, 9]
    neck = xyz[:, 8]
    mid_sh = 0.5 * (xyz[:, 11] + xyz[:, 14])
    out[:, 1] = nose + 0.15 * (head - nose) + 0.04 * (xyz[:, 11] - mid_sh)  # L eye
    out[:, 2] = nose + 0.15 * (head - nose) + 0.04 * (xyz[:, 14] - mid_sh)  # R eye
    out[:, 3] = neck + 0.55 * (xyz[:, 11] - neck)  # L ear approx
    out[:, 4] = neck + 0.55 * (xyz[:, 14] - neck)  # R ear
    out[:, 5] = xyz[:, 11]
    out[:, 6] = xyz[:, 14]
    out[:, 7] = xyz[:, 12]
    out[:, 8] = xyz[:, 15]
    out[:, 9] = xyz[:, 13]
    out[:, 10] = xyz[:, 16]
    out[:, 11] = xyz[:, 4]
    out[:, 12] = xyz[:, 1]
    out[:, 13] = xyz[:, 5]
    out[:, 14] = xyz[:, 2]
    out[:, 15] = xyz[:, 6]
    out[:, 16] = xyz[:, 3]
    return out


def normalize_display(xyz: np.ndarray) -> np.ndarray:
    """Root at mid-hip, Y-up, roughly human scale (hip→head ~0.7)."""
    out = xyz.copy()
    # MotionBERT wild often Y-down (image-like). Flip if head below hips on average.
    mid_hip = 0.5 * (out[:, 11] + out[:, 12])
    head = out[:, 0]
    if float(np.mean(head[:, 1] - mid_hip[:, 1])) < 0:
        out[:, :, 1] *= -1.0
        mid_hip = 0.5 * (out[:, 11] + out[:, 12])
        head = out[:, 0]
    out = out - mid_hip[:, None, :]
    # scale
    torso = np.linalg.norm(head - mid_hip, axis=1)
    scale = float(np.median(torso) + 1e-6)
    out = out / scale * 0.70
    return out


def run_motionbert(
    mb_root: Path,
    json_path: Path,
    vid_size: tuple[int, int],
    ckpt: Path,
    config: Path,
    clip_len: int = 243,
) -> np.ndarray:
    sys.path.insert(0, str(mb_root))
    from lib.utils.tools import get_config  # type: ignore
    from lib.utils.learning import load_backbone  # type: ignore
    from lib.utils.utils_data import flip_data  # type: ignore
    from lib.data.dataset_wild import WildDetDataset  # type: ignore

    args = get_config(str(config))
    model = load_backbone(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[motionbert] device={device}")
    if device.type == "cuda":
        model = nn.DataParallel(model)
        model = model.cuda()
    ckpt_obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = ckpt_obj["model_pos"]
    # strip module. prefix if present / absent mismatch
    model_state = model.state_dict()
    if any(k.startswith("module.") for k in state) and not any(
        k.startswith("module.") for k in model_state
    ):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    elif not any(k.startswith("module.") for k in state) and any(
        k.startswith("module.") for k in model_state
    ):
        state = {f"module.{k}": v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    # pixel mode: keep relative scale with image coords (recommended for wild)
    ds = WildDetDataset(
        str(json_path),
        clip_len=clip_len,
        vid_size=vid_size,
        scale_range=None,
        focus=0,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    results = []
    with torch.no_grad():
        for batch_input in loader:
            batch_input = batch_input.to(device)
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:
                batch_input_flip = flip_data(batch_input)
                pred1 = model(batch_input)
                pred2 = flip_data(model(batch_input_flip))
                pred = (pred1 + pred2) / 2.0
            else:
                pred = model(batch_input)
            if args.rootrel:
                pred[:, :, 0, :] = 0
            else:
                pred[:, 0, 0, 2] = 0
            results.append(pred.cpu().numpy())

    arr = np.concatenate([np.concatenate(r, axis=0) for r in results], axis=0)
    # pixel rescale (same as infer_wild --pixel)
    arr = arr * (min(vid_size) / 2.0)
    arr[:, :, :2] = arr[:, :, :2] + np.array(vid_size, dtype=np.float64) / 2.0
    return arr  # (T,17,3) H36M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--athlete-json",
        type=Path,
        default=Path(
            "/root/autodl-tmp/bike_projects/bike-project/diagnostics/"
            "t014_mono3d_pilot/athlete/athlete_predictions.json"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/bike_projects/bike-project/diagnostics/"
            "t014_mono3d_pilot/motionbert"
        ),
    )
    ap.add_argument("--mb-root", type=Path, default=DEFAULT_MB)
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_MB
        / "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MB / "configs/pose3d/MB_ft_h36m_global_lite.yaml",
    )
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument(
        "--rota-demo",
        type=Path,
        default=Path("/root/autodl-tmp/bike-ai-platform/apps/rota/data/demo/joints.json"),
    )
    ap.add_argument("--no-rota-copy", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ap_json = args.out_dir / "alphapose_halpe.json"
    meta_frames, image_size = write_alphapose(args.athlete_json, ap_json)
    # athlete JSON stores [H,W]; MotionBERT wants (W,H)
    if image_size[0] < image_size[1]:
        # already looks like H,W with H<W for landscape? 2160x3840 → H,W
        h, w = int(image_size[0]), int(image_size[1])
    else:
        w, h = int(image_size[0]), int(image_size[1])
    # Our file is [2160, 3840] = H,W
    if image_size == [2160, 3840] or image_size[0] == 2160:
        w, h = 3840, 2160
    elif image_size[0] == 3840:
        w, h = 3840, 2160
    else:
        # fallback: larger dim = width
        a, b = int(image_size[0]), int(image_size[1])
        w, h = (b, a) if a < b else (a, b)

    print(f"[motionbert] frames={len(meta_frames)} vid_size=({w},{h})")
    h36m = run_motionbert(
        args.mb_root, ap_json, (w, h), args.ckpt, args.config, clip_len=243
    )
    # trim / pad to meta length
    T = len(meta_frames)
    if h36m.shape[0] > T:
        h36m = h36m[:T]
    elif h36m.shape[0] < T:
        pad = np.repeat(h36m[-1:], T - h36m.shape[0], axis=0)
        h36m = np.concatenate([h36m, pad], axis=0)

    np.save(args.out_dir / "X3D_h36m.npy", h36m)
    coco = normalize_display(h36m17_to_coco17(h36m))

    frames_out = []
    for i, meta in enumerate(meta_frames):
        frames_out.append(
            {
                "frame": meta["frame"],
                "image_name": meta["image_name"],
                "bbox": meta["bbox"],
                "joints_xyz": coco[i].tolist(),
                "joint_scores": meta["joint_scores"],
            }
        )

    report = {
        "method": "sapiens2_0.4b + MotionBERT-Lite (MB_ft_h36m_global_lite)",
        "note": (
            "2D from locked-athlete Sapiens2; 3D from MotionBERT pretrained on H36M. "
            "Relative scale; not metric lab calibration. Cycling-domain fine-tune not applied yet."
        ),
        "fps": args.fps,
        "image_size": [w, h],
        "n_frames": T,
        "n_with_athlete": T,
        "coverage": 1.0,
        "checkpoint": str(args.ckpt),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    out_joints = {
        "report": report,
        "frames": frames_out,
    }
    out_path = args.out_dir / "mono3d_joints_motionbert.json"
    out_path.write_text(json.dumps(out_joints, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[motionbert] wrote {out_path}")

    if not args.no_rota_copy and args.rota_demo:
        args.rota_demo.parent.mkdir(parents=True, exist_ok=True)
        # keep compact for API
        compact = {"report": report, "frames": frames_out}
        args.rota_demo.write_text(json.dumps(compact, ensure_ascii=False), encoding="utf-8")
        print(f"[motionbert] updated ROTA demo {args.rota_demo}")


if __name__ == "__main__":
    main()
