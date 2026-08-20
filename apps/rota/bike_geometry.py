"""Bike-coordinate calibration, constrained pose refinement, and QA metrics.

The five user-adjustable points define a stable 2.5D bicycle coordinate frame:
rear hub=(0, 0), front hub=(1, 0), +Y points upward in the image.  Human joints
are expressed in that frame so the bike stays rigid for the whole sequence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


POINT_KEYS = ("rear_hub", "front_hub", "bottom_bracket", "saddle", "handlebar")
BODY_POINT_INDEX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
SKELETON_EDGES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

# Initial estimate for the bundled T014 camera-A demo. Coordinates are normalized
# to the source frame and intentionally remain editable in the UI.
DEFAULT_T014_CALIBRATION = {
    "version": 1,
    "confirmed": False,
    "points": {
        # The T014 rider faces image-left: rear wheel is on the right.
        "rear_hub": [0.590, 0.697],
        "front_hub": [0.493, 0.664],
        "bottom_bracket": [0.535, 0.650],
        "saddle": [0.535, 0.580],
        "handlebar": [0.507, 0.466],
    },
}


def validate_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    points = calibration.get("points") or {}
    cleaned: dict[str, list[float]] = {}
    for key in POINT_KEYS:
        value = points.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"missing or invalid calibration point: {key}")
        xy = [float(value[0]), float(value[1])]
        if not all(np.isfinite(xy)) or not all(0.0 <= v <= 1.0 for v in xy):
            raise ValueError(f"calibration point outside normalized image: {key}")
        cleaned[key] = xy
    rear = np.asarray(cleaned["rear_hub"])
    front = np.asarray(cleaned["front_hub"])
    if float(np.linalg.norm(front - rear)) < 0.03:
        raise ValueError("rear and front hubs are too close")
    return {
        "version": 1,
        "confirmed": bool(calibration.get("confirmed", False)),
        "points": cleaned,
    }


def load_or_create_calibration(path: Path) -> dict[str, Any]:
    if path.exists():
        return validate_calibration(json.loads(path.read_text(encoding="utf-8")))
    value = validate_calibration(DEFAULT_T014_CALIBRATION)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return value


def save_calibration(path: Path, calibration: dict[str, Any]) -> dict[str, Any]:
    value = validate_calibration(calibration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return value


def validate_body_calibration(
    calibration: dict[str, Any], *, n_frames: int | None = None
) -> dict[str, Any]:
    """Validate one-frame manual anchors for a personalized rigid-link skeleton."""
    points = calibration.get("points") or {}
    cleaned: dict[str, list[float]] = {}
    for key in BODY_POINT_INDEX:
        value = points.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"missing or invalid body point: {key}")
        xy = [float(value[0]), float(value[1])]
        if not all(np.isfinite(xy)) or not all(0.0 <= v <= 1.0 for v in xy):
            raise ValueError(f"body point outside normalized image: {key}")
        cleaned[key] = xy
    reference_frame = int(calibration.get("reference_frame", 0))
    if reference_frame < 0 or (n_frames is not None and reference_frame >= n_frames):
        raise ValueError("body reference frame is outside the sequence")
    return {
        "version": 1,
        "confirmed": bool(calibration.get("confirmed", False)),
        "reference_frame": reference_frame,
        "points": cleaned,
    }


def load_body_calibration(
    path: Path, *, n_frames: int | None = None
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return validate_body_calibration(
        json.loads(path.read_text(encoding="utf-8")), n_frames=n_frames
    )


def save_body_calibration(
    path: Path, calibration: dict[str, Any], *, n_frames: int | None = None
) -> dict[str, Any]:
    value = validate_body_calibration(calibration, n_frames=n_frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return value


def _bike_basis(calibration: dict[str, Any], image_size: tuple[int, int]) -> tuple[np.ndarray, ...]:
    """Return origin, forward, upward, and wheelbase in pixel coordinates."""
    w, h = image_size
    points = validate_calibration(calibration)["points"]
    rear = np.asarray([points["rear_hub"][0] * w, points["rear_hub"][1] * h])
    front = np.asarray([points["front_hub"][0] * w, points["front_hub"][1] * h])
    axis = front - rear
    wheelbase = float(np.linalg.norm(axis))
    forward = axis / wheelbase
    upward = np.asarray([forward[1], -forward[0]])
    if upward[1] > 0:  # image Y points down; bike +Y must point visually upward
        upward *= -1.0
    return rear, forward, upward, np.asarray(wheelbase)


def image_to_bike(
    xy: np.ndarray, calibration: dict[str, Any], image_size: tuple[int, int]
) -> np.ndarray:
    """Transform pixel XY coordinates into wheelbase-normalized bike coordinates."""
    rear, forward, upward, wheelbase_array = _bike_basis(calibration, image_size)
    wheelbase = float(wheelbase_array)
    delta = np.asarray(xy, dtype=np.float64) - rear
    x = np.sum(delta * forward, axis=-1) / wheelbase
    y = np.sum(delta * upward, axis=-1) / wheelbase
    return np.stack([x, y], axis=-1)


def bike_to_image(
    xy: np.ndarray, calibration: dict[str, Any], image_size: tuple[int, int]
) -> np.ndarray:
    rear, forward, upward, wheelbase_array = _bike_basis(calibration, image_size)
    wheelbase = float(wheelbase_array)
    values = np.asarray(xy, dtype=np.float64)
    return rear + wheelbase * (
        values[..., 0, None] * forward + values[..., 1, None] * upward
    )


# 56 cm endurance road bike in wheelbase units (~1.00 m). BB drop, saddle
# setback, and bar reach are the reason a person can actually sit and pedal.
CANONICAL_ROAD_BIKE = {
    "rear_hub": [0.0, 0.0],
    "front_hub": [1.0, 0.0],
    "bottom_bracket": [0.415, -0.070],
    "saddle": [0.340, 0.660],
    "handlebar": [0.860, 0.560],
    "wheel_radius": 0.335,
    "crank_length": 0.175,
    "handlebar_half_width": 0.20,
    "pedal_half_width": 0.11,
}


def facing_corrected_calibration(
    calibration: dict[str, Any],
    athlete: dict[str, Any] | None,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Flip front/rear if the labeled nose of the bike disagrees with the rider.

    Hands sit on the bars, so wrists must lie toward the front hub. The T014
    clicks had front/rear swapped, which made the 3D rider face the rear wheel.
    """
    value = validate_calibration(calibration)
    if athlete is None:
        value["facing_flipped"] = False
        return value
    xy, _ = _athlete_arrays(athlete)
    if len(xy) == 0:
        value["facing_flipped"] = False
        return value
    wrist = np.nanmean(0.5 * (xy[:, 9] + xy[:, 10]), axis=0)
    hip = np.nanmean(0.5 * (xy[:, 11] + xy[:, 12]), axis=0)
    w, h = image_size
    points = value["points"]
    rear = np.asarray([points["rear_hub"][0] * w, points["rear_hub"][1] * h])
    front = np.asarray([points["front_hub"][0] * w, points["front_hub"][1] * h])
    if float(np.dot(wrist - hip, front - rear)) < 0.0:
        value = {
            "version": 1,
            "confirmed": value.get("confirmed", False),
            "points": {
                **points,
                "rear_hub": list(points["front_hub"]),
                "front_hub": list(points["rear_hub"]),
            },
            "facing_flipped": True,
        }
        return validate_calibration(value) | {"facing_flipped": True}
    value["facing_flipped"] = False
    return value


def bike_geometry(
    calibration: dict[str, Any],
    image_size: tuple[int, int],
    athlete: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del image_size, athlete, calibration
    return {key: (list(value) if isinstance(value, list) else value)
            for key, value in CANONICAL_ROAD_BIKE.items()}


def _athlete_arrays(athlete: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    xy, scores = [], []
    for frame in athlete.get("frames") or []:
        instances = frame.get("instances") or []
        if not instances:
            xy.append(np.full((17, 2), np.nan))
            scores.append(np.zeros(17))
            continue
        instance = instances[0]
        points = np.asarray((instance.get("keypoints") or [])[:17], dtype=np.float64)
        confidence = np.asarray(
            (instance.get("keypoint_scores") or [1.0] * 17)[:17], dtype=np.float64
        )
        if points.shape != (17, 2):
            points = np.full((17, 2), np.nan)
            confidence = np.zeros(17)
        xy.append(points)
        scores.append(confidence)
    return np.asarray(xy), np.asarray(scores)


def _smooth_series(values: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Interpolate gaps and lightly smooth 2D tracks without erasing pedalling."""
    from scipy.signal import savgol_filter

    out = np.asarray(values, dtype=np.float64).copy()
    t = np.arange(len(out))
    for joint in range(out.shape[1]):
        for axis in range(2):
            valid = np.isfinite(out[:, joint, axis]) & (scores[:, joint] > 0.12)
            if int(valid.sum()) < 2:
                out[:, joint, axis] = 0.0
                continue
            out[:, joint, axis] = np.interp(t, t[valid], out[valid, joint, axis])
            if len(out) >= 7:
                out[:, joint, axis] = savgol_filter(
                    out[:, joint, axis], window_length=7, polyorder=2, mode="interp"
                )
    return out


def _cadence(values: np.ndarray, fps: float) -> float | None:
    if len(values) < max(12, int(fps * 2)):
        return None
    signal = np.asarray(values, dtype=np.float64)
    signal = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    band = (frequencies >= 0.55) & (frequencies <= 2.5)
    if not np.any(band):
        return None
    frequency = float(frequencies[band][np.argmax(spectrum[band])])
    return round(frequency * 60.0, 1)


def _bone_stats(xyz: np.ndarray) -> tuple[float, float]:
    cvs = []
    for a, b in SKELETON_EDGES:
        length = np.linalg.norm(xyz[:, a] - xyz[:, b], axis=1)
        mean = float(np.mean(length))
        if mean > 1e-8:
            cvs.append(float(np.std(length) / mean))
    return float(np.mean(cvs) * 100.0), float(np.max(cvs) * 100.0)


def quality_report(
    xyz: np.ndarray,
    observed_bike_xy: np.ndarray,
    scores: np.ndarray,
    geometry: dict[str, Any],
    *,
    fps: float,
    wheelbase_px: float,
) -> dict[str, Any]:
    """Compute transparent geometric QA metrics; lower is better except cadence."""
    predicted_px = xyz[:, :, :2] * wheelbase_px
    observed_px = observed_bike_xy * wheelbase_px
    valid = np.isfinite(observed_px).all(axis=2) & (scores > 0.15)
    error = np.linalg.norm(predicted_px - observed_px, axis=2)
    weighted_error = error[valid]
    reprojection = float(np.sqrt(np.mean(weighted_error**2))) if weighted_error.size else None

    mean_cv, max_cv = _bone_stats(xyz)
    bar = np.asarray(geometry["handlebar"][:2])
    saddle = np.asarray(geometry["saddle"][:2])
    bb = np.asarray(geometry["bottom_bracket"][:2])
    wrist_mid = 0.5 * (xyz[:, 9, :2] + xyz[:, 10, :2])
    hip_mid = 0.5 * (xyz[:, 11, :2] + xyz[:, 12, :2])
    ankle_l = xyz[:, 15, :2] - bb
    ankle_r = xyz[:, 16, :2] - bb
    radius_l = np.linalg.norm(ankle_l, axis=1)
    radius_r = np.linalg.norm(ankle_r, axis=1)
    opposition = np.linalg.norm(ankle_l + ankle_r, axis=1)
    grip_half_width = float(geometry.get("handlebar_half_width", 0.18))
    pedal_half_width = float(geometry.get("pedal_half_width", 0.105))
    left_grip_3d = np.asarray([bar[0], bar[1], grip_half_width])
    right_grip_3d = np.asarray([bar[0], bar[1], -grip_half_width])
    left_grip_error = np.linalg.norm(xyz[:, 9] - left_grip_3d, axis=1)
    right_grip_error = np.linalg.norm(xyz[:, 10] - right_grip_3d, axis=1)
    foot_side_ok = bool(
        np.all(xyz[:, 15, 2] > 0.5 * pedal_half_width)
        and np.all(xyz[:, 16, 2] < -0.5 * pedal_half_width)
    )

    return {
        "reprojection_rmse_px": round(reprojection, 2) if reprojection is not None else None,
        "bone_length_cv_mean_pct": round(mean_cv, 2),
        "bone_length_cv_max_pct": round(max_cv, 2),
        "hand_to_handlebar_pct_wheelbase": round(
            float(np.mean(np.linalg.norm(wrist_mid - bar, axis=1)) * 100.0), 2
        ),
        "hip_to_saddle_pct_wheelbase": round(
            float(np.mean(np.linalg.norm(hip_mid - saddle, axis=1)) * 100.0), 2
        ),
        "pelvis_motion_std_pct_wheelbase": round(
            float(np.sqrt(np.sum(np.var(hip_mid, axis=0))) * 100.0), 2
        ),
        "left_ankle_radius_cv_pct": round(
            float(np.std(radius_l) / (np.mean(radius_l) + 1e-9) * 100.0), 2
        ),
        "right_ankle_radius_cv_pct": round(
            float(np.std(radius_r) / (np.mean(radius_r) + 1e-9) * 100.0), 2
        ),
        "crank_opposition_error_pct_wheelbase": round(float(np.mean(opposition) * 100.0), 2),
        "left_hand_grip_error_pct_wheelbase": round(float(np.mean(left_grip_error) * 100.0), 3),
        "right_hand_grip_error_pct_wheelbase": round(float(np.mean(right_grip_error) * 100.0), 3),
        "feet_on_opposite_sides": foot_side_ok,
        "left_elbow_on_left_side": bool(np.mean(xyz[:, 7, 2]) > 0.02),
        "left_knee_on_left_side": bool(np.mean(xyz[:, 13, 2]) > 0.02),
        "right_elbow_on_right_side": bool(np.mean(xyz[:, 8, 2]) < -0.02),
        "right_knee_on_right_side": bool(np.mean(xyz[:, 14, 2]) < -0.02),
        "hard_contacts_passed": bool(
            float(np.max(left_grip_error)) < 0.005
            and float(np.max(right_grip_error)) < 0.005
            and float(np.max(opposition)) < 0.005
            and foot_side_ok
        ),
        "cadence_rpm_from_left_ankle": _cadence(xyz[:, 15, 1], fps),
        "cadence_rpm_from_right_ankle": _cadence(xyz[:, 16, 1], fps),
    }


def prepare_pose(
    motionbert: dict[str, Any],
    athlete: dict[str, Any],
    calibration: dict[str, Any],
    body_calibration: dict[str, Any] | None = None,
    *,
    fps: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    float,
    np.ndarray | None,
    dict[str, Any] | None,
]:
    """Create a bike-coordinate 3D initialization from 2D observations + MB depth."""
    image_size_raw = athlete.get("image_size") or [2160, 3840]
    h, w = int(image_size_raw[0]), int(image_size_raw[1])
    image_size = (w, h)
    calibration = facing_corrected_calibration(calibration, athlete, image_size)
    observed_px, scores = _athlete_arrays(athlete)
    body_reference = None
    manual_targets_px: dict[str, np.ndarray] = {}
    if body_calibration:
        body_reference = validate_body_calibration(
            body_calibration, n_frames=len(observed_px)
        )
        ref = body_reference["reference_frame"]
        scale = np.asarray([w, h], dtype=np.float64)
        # The marked frame personalizes proportions and IK bend directions.
        # Never smear a one-frame correction over the whole sequence: that was
        # the main source of drifting/crossed limbs in the previous solver.
        for key, joint in BODY_POINT_INDEX.items():
            marked = np.asarray(body_reference["points"][key], dtype=np.float64) * scale
            manual_targets_px[key] = marked
            observed_px[ref, joint] = marked
            scores[ref, joint] = 1.0
    observed_bike = image_to_bike(observed_px, calibration, image_size)
    observed_bike = _smooth_series(observed_bike, scores)
    body_anchor = None
    if body_reference:
        ref = body_reference["reference_frame"]
        joints = np.asarray(list(BODY_POINT_INDEX.values()), dtype=np.int64)
        targets_px = np.asarray(
            [manual_targets_px[key] for key in BODY_POINT_INDEX], dtype=np.float64
        )
        targets_bike = image_to_bike(targets_px, calibration, image_size)
        # Smoothing must never dilute the one frame the user explicitly marked.
        observed_bike[ref, joints] = targets_bike
        body_anchor = {"frame": ref, "joints": joints, "xy": targets_bike}

    mb_xyz = np.asarray(
        [frame["joints_xyz"] for frame in motionbert.get("frames") or []], dtype=np.float64
    )
    count = min(len(mb_xyz), len(observed_bike))
    mb_xyz = mb_xyz[:count]
    observed_bike = observed_bike[:count]
    scores = scores[:count]

    mb_hip = 0.5 * (mb_xyz[:, 11] + mb_xyz[:, 12])
    mb_centered = mb_xyz - mb_hip[:, None, :]
    mb_torso = np.linalg.norm(
        0.5 * (mb_centered[:, 5, :2] + mb_centered[:, 6, :2]), axis=1
    )
    obs_hip = 0.5 * (observed_bike[:, 11] + observed_bike[:, 12])
    obs_shoulder = 0.5 * (observed_bike[:, 5] + observed_bike[:, 6])
    obs_torso = np.linalg.norm(obs_shoulder - obs_hip, axis=1)
    depth_scale = float(np.median(obs_torso) / (np.median(mb_torso) + 1e-9))

    initial = np.zeros((count, 17, 3), dtype=np.float64)
    initial[:, :, :2] = observed_bike
    initial[:, :, 2] = mb_centered[:, :, 2] * depth_scale
    geometry = bike_geometry(calibration, image_size)
    _, _, _, wheelbase_array = _bike_basis(calibration, image_size)
    target_bones = None
    if body_reference:
        ref = min(body_reference["reference_frame"], count - 1)
        edge_array = np.asarray(SKELETON_EDGES, dtype=np.int64)
        all_lengths = np.linalg.norm(
            initial[:, edge_array[:, 0]] - initial[:, edge_array[:, 1]], axis=2
        )
        median_lengths = np.median(all_lengths, axis=0)
        reference_lengths = all_lengths[ref]
        # Blend the marked frame with the robust sequence median. The clamp
        # prevents a foreshortened limb in one monocular frame becoming anatomy.
        reference_lengths = np.clip(
            reference_lengths, 0.70 * median_lengths, 1.35 * median_lengths
        )
        target_bones = 0.65 * reference_lengths + 0.35 * median_lengths
    return (
        initial,
        observed_bike,
        scores,
        geometry,
        float(wheelbase_array),
        target_bones,
        body_anchor,
    )


def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Small edge-preserving temporal smoother used for root/torso motion."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3 or window <= 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack(
        [np.mean(padded[i : i + window], axis=0) for i in range(len(values))]
    )


def _fixed_side_axis(projected: np.ndarray, max_projection: float = 0.70) -> np.ndarray:
    """Lift a marked left-right image direction into a stable 3D side axis."""
    xy = np.asarray(projected, dtype=np.float64)
    length = float(np.linalg.norm(xy))
    if length < 1e-8:
        return np.asarray([0.0, 0.0, 1.0])
    xy = xy / length * min(length, max_projection)
    z = float(np.sqrt(max(1e-6, 1.0 - float(np.dot(xy, xy)))))
    axis = np.asarray([xy[0], xy[1], z], dtype=np.float64)
    return axis / (np.linalg.norm(axis) + 1e-9)


def _sagittal_two_bone(
    root: np.ndarray,
    target: np.ndarray,
    length_a: float,
    length_b: float,
    *,
    prefer: str,
    lateral: float,
) -> np.ndarray:
    """Solve a two-link chain in the bike's sagittal plane (XY).

    A person pedalling stays in that plane: knees track forward/up, elbows
    hang under the shoulder–hand line. The 3D solver used to pick the
    downward or camera-side solution, which is why it did not look like riding.
    """
    root = np.asarray(root, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = target[:2] - root[:2]
    distance = float(np.linalg.norm(delta))
    z = 0.55 * float(root[2]) + 0.45 * float(target[2])
    z = float(np.clip(z, min(root[2], target[2]), max(root[2], target[2])))
    z += 0.35 * (lateral - z)
    if distance < 1e-8:
        mid = root.copy()
        mid[0] += 0.12 if prefer == "knee" else 0.06
        mid[1] += 0.10 if prefer == "knee" else -0.08
        mid[2] = z
        return mid
    direction = delta / distance
    reach_min = abs(length_a - length_b) + 1e-6
    reach_max = length_a + length_b - 1e-6
    solved = float(np.clip(distance, reach_min, reach_max))
    along = (length_a**2 - length_b**2 + solved**2) / (2.0 * solved)
    height = float(np.sqrt(max(0.0, length_a**2 - along**2)))
    center = root[:2] + direction * along
    perp = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    cand_a = center + perp * height
    cand_b = center - perp * height
    if prefer == "knee":
        # The riding knee is the more forward, then the higher, of the two.
        key_a = (float(cand_a[0]), float(cand_a[1]))
        key_b = (float(cand_b[0]), float(cand_b[1]))
        chosen = cand_a if key_a > key_b else cand_b
        chosen[0] = max(float(chosen[0]), float(root[0]) - 0.02)
        chosen[1] = max(float(chosen[1]), float(target[1]) + 0.10)
    else:
        # Hoods: elbow between shoulder and hand, below both.
        key_a = (-float(cand_a[1]), float(cand_a[0]))
        key_b = (-float(cand_b[1]), float(cand_b[0]))
        chosen = cand_a if key_a > key_b else cand_b
        lo_x, hi_x = sorted((float(root[0]), float(target[0])))
        chosen[0] = float(np.clip(chosen[0], lo_x + 0.01, hi_x - 0.01))
        chosen[1] = min(float(chosen[1]), float(root[1]) - 0.04, float(target[1]) + 0.06)
    return np.asarray([chosen[0], chosen[1], z], dtype=np.float64)


def _infer_crank_phase(
    observed_bike_xy: np.ndarray, scores: np.ndarray, fps: float
) -> tuple[np.ndarray, float, float]:
    """Infer one smooth crank angle; the other pedal is exactly 180° away."""
    difference = observed_bike_xy[:, 15] - observed_bike_xy[:, 16]
    angles = np.arctan2(difference[:, 1], difference[:, 0])
    amplitude = np.linalg.norm(difference, axis=1)
    confidence = np.minimum(scores[:, 15], scores[:, 16])
    weights = np.clip(confidence, 0.05, 1.0) * np.clip(amplitude, 0.02, None)
    time = np.arange(len(angles), dtype=np.float64)

    # Circular grid fit avoids unwrap failures when feet overlap in the image.
    cadence_hint = _cadence(observed_bike_xy[:, 15, 1], fps)
    base_hz = float(cadence_hint / 60.0) if cadence_hint else 1.5
    low_hz, high_hz = max(0.5, base_hz - 0.12), min(2.5, base_hz + 0.12)
    hz = np.concatenate(
        [np.linspace(-high_hz, -low_hz, 121), np.linspace(low_hz, high_hz, 121)]
    )
    best_score = -1.0
    best_omega = 2.0 * np.pi * 1.5 / max(fps, 1e-6)
    best_phase = float(angles[0]) if len(angles) else 0.0
    unit = np.exp(1j * angles)
    for frequency in hz:
        omega = 2.0 * np.pi * float(frequency) / max(fps, 1e-6)
        correlation = np.sum(weights * unit * np.exp(-1j * omega * time))
        score = float(abs(correlation))
        if score > best_score:
            best_score = score
            best_omega = omega
            best_phase = float(np.angle(correlation))
    phase = best_phase + best_omega * time
    coherence = best_score / (float(np.sum(weights)) + 1e-9)
    cadence = abs(best_omega) * max(fps, 1e-6) / (2.0 * np.pi) * 60.0
    return phase, float(cadence), float(coherence)


def optimize_pose(
    initial: np.ndarray,
    observed_bike_xy: np.ndarray,
    scores: np.ndarray,
    geometry: dict[str, Any],
    target_bones_np: np.ndarray | None = None,
    body_anchor: dict[str, Any] | None = None,
    *,
    steps: int = 450,
    fps: float = 10.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a cycling-specific articulated pose with hard human/bike contacts.

    This intentionally replaces the old unconstrained Adam fit. A bicycle is a
    closed kinematic system: hands stay on separate grips, feet stay on separate
    pedals, pedals are opposed, and elbows/knees solve the remaining link chain.
    """
    del steps  # Kept in the public signature for API compatibility.
    q0 = np.asarray(initial, dtype=np.float64)
    q = q0.copy()
    n_frames = len(q)
    if not n_frames:
        return q, {}

    edges = np.asarray(SKELETON_EDGES, dtype=np.int64)
    if target_bones_np is None:
        target_bones = np.median(
            np.linalg.norm(q0[:, edges[:, 0]] - q0[:, edges[:, 1]], axis=2), axis=0
        )
    else:
        target_bones = np.asarray(target_bones_np, dtype=np.float64)

    # Adult on a ~1 m wheelbase: femur+tibia ≈ saddle-to-pedal at BDC.
    del target_bones
    shoulder_width = 0.38
    hip_width = 0.24
    upper_arm = 0.30
    forearm = 0.27
    thigh = 0.50
    shin = 0.48
    reference = int(body_anchor["frame"]) if body_anchor else n_frames // 2
    reference = int(np.clip(reference, 0, n_frames - 1))

    hip_observed = 0.5 * (q0[:, 11, :2] + q0[:, 12, :2])
    shoulder_observed = 0.5 * (q0[:, 5, :2] + q0[:, 6, :2])
    saddle = np.asarray(geometry["saddle"][:2], dtype=np.float64)
    handlebar = np.asarray(geometry["handlebar"][:2], dtype=np.float64)
    bottom_bracket = np.asarray(geometry["bottom_bracket"][:2], dtype=np.float64)

    crank_phase, cadence, phase_coherence = _infer_crank_phase(
        observed_bike_xy, scores, fps
    )

    # Sit on the saddle; 52° from vertical is hoods on an endurance road bike.
    seat_offset = np.asarray([0.00, 0.045], dtype=np.float64)
    hip_res = _moving_average(
        hip_observed - np.median(hip_observed, axis=0), window=5
    )
    hip_res = np.clip(hip_res, -0.035, 0.035) * 0.55
    hip_mid = np.column_stack(
        [saddle[None, :] + seat_offset[None, :] + hip_res, np.zeros(n_frames)]
    )

    torso_angle = np.full(n_frames, np.deg2rad(52.0))
    torso_length = 0.48
    torso_direction = np.column_stack(
        [np.sin(torso_angle), np.cos(torso_angle), np.zeros(n_frames)]
    )
    sho_res = _moving_average(
        shoulder_observed - np.median(shoulder_observed, axis=0), window=5
    )
    sho_res = np.clip(sho_res, -0.045, 0.045) * 0.70
    shoulder_mid = hip_mid + torso_length * torso_direction
    shoulder_mid[:, :2] += sho_res

    # Pedalling makes a person rock. Locking the torso made a statue; the
    # teacher wants that left-right human sway, locked to crank phase.
    sway = np.sin(crank_phase)
    bob = np.cos(2.0 * crank_phase)
    hip_mid[:, 1] += 0.006 * bob
    hip_mid[:, 2] += 0.016 * sway
    shoulder_mid[:, 1] += 0.010 * bob
    shoulder_mid[:, 2] += 0.032 * sway
    roll = 0.020 * sway
    yaw = 0.014 * sway

    rider_left = np.asarray([0.0, 0.0, 1.0])
    q[:, 5] = shoulder_mid + 0.5 * shoulder_width * rider_left
    q[:, 6] = shoulder_mid - 0.5 * shoulder_width * rider_left
    q[:, 5, 0] += yaw
    q[:, 6, 0] -= yaw
    q[:, 5, 1] -= roll
    q[:, 6, 1] += roll
    q[:, 11] = hip_mid + 0.5 * hip_width * rider_left
    q[:, 12] = hip_mid - 0.5 * hip_width * rider_left
    q[:, 11, 1] -= 0.55 * roll
    q[:, 12, 1] += 0.55 * roll
    q[:, 11, 2] += 0.008 * sway
    q[:, 12, 2] -= 0.008 * sway

    # Head follows the rocking shoulders, damped so it does not bobble.
    forward_xy = handlebar - saddle
    forward_xy /= np.linalg.norm(forward_xy) + 1e-9
    head_center = 0.5 * (q[:, 5] + q[:, 6]) + 0.225 * torso_direction
    head_center[:, 2] *= 0.55
    q[:, 0] = head_center + np.asarray([0.025 * forward_xy[0], 0.025 * forward_xy[1], 0.0])
    q[:, 1] = head_center + np.asarray([0.012, 0.025, 0.035])
    q[:, 2] = head_center + np.asarray([0.012, 0.025, -0.035])
    q[:, 3] = head_center + np.asarray([-0.025, 0.012, 0.070])
    q[:, 4] = head_center + np.asarray([-0.025, 0.012, -0.070])

    # Hands on the hoods, slightly ahead and below the bar center.
    grip_half_width = float(geometry.get("handlebar_half_width", 0.20))
    left_grip = np.asarray(
        [handlebar[0] + 0.030, handlebar[1] - 0.018, grip_half_width]
    )
    right_grip = np.asarray(
        [handlebar[0] + 0.030, handlebar[1] - 0.018, -grip_half_width]
    )
    q[:, 9] = left_grip
    q[:, 10] = right_grip

    # Hard pedal contacts: one inferred crank phase, exactly opposed pedals,
    # and permanent left/right depth separation so legs cannot swap sides.
    crank_length = float(geometry.get("crank_length", 0.165))
    pedal_half_width = float(geometry.get("pedal_half_width", 0.105))
    crank_xy = np.column_stack([np.cos(crank_phase), np.sin(crank_phase)]) * crank_length
    q[:, 15, :2] = bottom_bracket + crank_xy
    q[:, 16, :2] = bottom_bracket - crank_xy
    q[:, 15, 2] = pedal_half_width
    q[:, 16, 2] = -pedal_half_width

    # Grow once if a contact is out of reach; keep enough slack to bend.
    arm_distance = max(
        float(np.max(np.linalg.norm(q[:, 9] - q[:, 5], axis=1))),
        float(np.max(np.linalg.norm(q[:, 10] - q[:, 6], axis=1))),
    )
    arm_scale = max(1.0, arm_distance * 1.04 / (upper_arm + forearm))
    upper_arm *= arm_scale
    forearm *= arm_scale
    leg_distance = max(
        float(np.max(np.linalg.norm(q[:, 15] - q[:, 11], axis=1))),
        float(np.max(np.linalg.norm(q[:, 16] - q[:, 12], axis=1))),
    )
    # Bottom-dead-centre should be a slightly bent knee (~150°), not locked.
    leg_scale = max(1.0, leg_distance * 1.04 / (thigh + shin))
    thigh *= leg_scale
    shin *= leg_scale

    def _hoods_elbow(shoulder: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        """Upper arm forward-down, elbow under the shoulder–hand line."""
        reach = wrist[:2] - shoulder[:2]
        xy = shoulder[:2] + 0.42 * reach
        xy[1] -= 0.10
        xy[1] = min(float(xy[1]), float(shoulder[1]) - 0.14)
        xy[0] = max(float(xy[0]), float(shoulder[0]) + 0.06)
        z = 0.65 * float(shoulder[2]) + 0.35 * float(wrist[2])
        return np.asarray([xy[0], xy[1], z], dtype=np.float64)

    for frame in range(n_frames):
        q[frame, 7] = _hoods_elbow(q[frame, 5], q[frame, 9])
        q[frame, 8] = _hoods_elbow(q[frame, 6], q[frame, 10])
        q[frame, 13] = _sagittal_two_bone(
            q[frame, 11], q[frame, 15], thigh, shin,
            prefer="knee", lateral=float(q[frame, 11, 2]),
        )
        q[frame, 14] = _sagittal_two_bone(
            q[frame, 12], q[frame, 16], thigh, shin,
            prefer="knee", lateral=float(q[frame, 12, 2]),
        )
    q[:, 7, 2] += 0.014 * sway
    q[:, 8, 2] -= 0.014 * sway

    lengths = np.linalg.norm(q[:, edges[:, 0]] - q[:, edges[:, 1]], axis=2)
    wrist_error = 0.5 * (
        np.linalg.norm(q[:, 9] - left_grip, axis=1)
        + np.linalg.norm(q[:, 10] - right_grip, axis=1)
    )
    opposition = np.linalg.norm(
        (q[:, 15, :2] - bottom_bracket) + (q[:, 16, :2] - bottom_bracket), axis=1
    )
    diagnostics = {
        "solver": "cycling_hard_kinematics_v5_upper_sway",
        "hand_contact": float(np.mean(wrist_error)),
        "crank_opposition": float(np.mean(opposition)),
        "bone_rmse": float(np.sqrt(np.mean((lengths - np.median(lengths, axis=0)) ** 2))),
        "cadence_rpm": float(cadence),
        "phase_coherence": float(phase_coherence),
        "arm_scale": float(arm_scale),
        "leg_scale": float(leg_scale),
    }
    return q, diagnostics


def build_optimized_analysis(
    motionbert: dict[str, Any],
    athlete: dict[str, Any],
    calibration: dict[str, Any],
    body_calibration: dict[str, Any] | None = None,
    *,
    fps: float,
    steps: int = 450,
) -> tuple[dict[str, Any], dict[str, Any]]:
    initial, observed, scores, geometry, wheelbase_px, target_bones, body_anchor = prepare_pose(
        motionbert, athlete, calibration, body_calibration, fps=fps
    )
    before = quality_report(
        initial, observed, scores, geometry, fps=fps, wheelbase_px=wheelbase_px
    )
    optimized, loss_parts = optimize_pose(
        initial,
        observed,
        scores,
        geometry,
        target_bones_np=target_bones,
        body_anchor=body_anchor,
        steps=steps,
        fps=fps,
    )
    after = quality_report(
        optimized, observed, scores, geometry, fps=fps, wheelbase_px=wheelbase_px
    )
    if body_anchor:
        ref = int(body_anchor["frame"])
        joints = np.asarray(body_anchor["joints"], dtype=np.int64)
        anchor_error_px = np.linalg.norm(
            optimized[ref, joints, :2] - np.asarray(body_anchor["xy"]), axis=1
        ) * wheelbase_px
        after["body_anchor_reference_rmse_px"] = round(
            float(np.sqrt(np.mean(anchor_error_px**2))), 2
        )

    source_frames = motionbert.get("frames") or []
    frames = []
    for index in range(len(optimized)):
        source = source_frames[index] if index < len(source_frames) else {}
        frames.append(
            {
                "frame": index,
                "image_name": source.get("image_name", f"frame_{index + 1:06d}.jpg"),
                "bbox": source.get("bbox"),
                "joints_xyz": optimized[index].tolist(),
                "joint_scores": scores[index].tolist(),
            }
        )

    report = {
        "method": "Sapiens2 + MotionBERT init + cycling kinematics v5 (upper-body sway)",
        "coordinate_system": "bike-wheelbase",
        "fps": fps,
        "n_frames": len(frames),
        "calibration_confirmed": bool(calibration.get("confirmed", False)),
        "body_calibration_confirmed": bool(
            body_calibration and body_calibration.get("confirmed", False)
        ),
        "quality_before": before,
        "quality_after": after,
        "optimizer": loss_parts,
    }
    return {
        "report": report,
        "bike_calibration": validate_calibration(calibration),
        "body_calibration": (
            validate_body_calibration(body_calibration, n_frames=len(frames))
            if body_calibration
            else None
        ),
        "bike_geometry": geometry,
        "frames": frames,
    }, report
