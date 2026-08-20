"""Cycling metrics from monocular 3D joint sequences (COCO-17)."""

from __future__ import annotations

from typing import Any

import numpy as np

L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

# Hover guides: how computed + how to read (Chinese)
METRIC_GUIDES: dict[str, dict[str, str]] = {
    "form_index": {
        "label": "动作质量指数",
        "hint": "综合评分",
        "guide": (
            "由「左右膝对称性、左踝轨迹圆滑度、骨盆稳定性」三项取平均得到（满分 100）。"
            "用于快速把握整体骑姿质量。越接近 85–100 分越好；低于 70 分建议结合下方单项排查。"
        ),
    },
    "cadence_rpm": {
        "label": "踏频",
        "hint": "每分钟转数",
        "guide": (
            "由左踝关节上下位移序列做频谱分析，取主频后 ×60 得到 rpm。"
            "表示踩踏节奏。室内/公路常见训练区约 80–100 rpm；过低易增加关节负荷，过高需看是否跟得上。"
        ),
    },
    "knee_left_deg": {
        "label": "左膝活动度",
        "hint": "ROM = 最大−最小",
        "guide": (
            "由左髋—左膝—左踝三点夹角随时间变化，取极差（ROM）与均值。"
            "反映左侧膝在踩踏周期中的屈曲伸展幅度。需与右侧对照；单侧异常偏大/偏小都值得关注。"
        ),
    },
    "knee_right_deg": {
        "label": "右膝活动度",
        "hint": "ROM = 最大−最小",
        "guide": (
            "由右髋—右膝—右踝三点夹角随时间变化，取极差（ROM）与均值。"
            "反映右侧膝活动幅度。与左侧差异过大时，可能提示发力不均或坐姿偏移。"
        ),
    },
    "knee_symmetry_pct": {
        "label": "左右膝对称性",
        "hint": "越高越对称",
        "guide": (
            "比较左右膝平均夹角差异，映射为 0–100 分：差异越小分越高。"
            "表示左右踩踏模式是否接近。≥85 较均衡；70–85 轻度不对称；<70 明显不对称，建议结合视频复核。"
        ),
    },
    "torso_lean_deg": {
        "label": "躯干前倾角",
        "hint": "相对竖直方向",
        "guide": (
            "取双肩中点与双髋中点连线，相对竖直方向的倾角绝对值，再对时间求平均（并报告 ROM）。"
            "表示上身前倾程度，与气动姿势、车把高度相关。数值本身无统一“最优角”，需对照车型与训练目标解读；"
            "同一运动员跨场次对比更有意义。"
        ),
    },
    "ankle_path_circularity_l_pct": {
        "label": "左踝轨迹圆滑度",
        "hint": "越高越圆",
        "guide": (
            "统计左踝在侧视投影上的轨迹到圆心距离，用变异系数换算为 0–100 分（越圆越高）。"
            "表示踩踏圆周是否平滑。越高越好；明显偏低可能提示“方踩”、脚跟或拉蹬不连贯。"
        ),
    },
    "ankle_path_circularity_r_pct": {
        "label": "右踝轨迹圆滑度",
        "hint": "越高越圆",
        "guide": (
            "统计右踝轨迹圆滑度（算法同左侧）。"
            "越高越好；左右差异大时可对照原视频看是否单侧代偿。"
        ),
    },
    "hip_stability_pct": {
        "label": "骨盆稳定性",
        "hint": "上下晃动越小越高",
        "guide": (
            "取左右髋中点高度随时间的标准差，映射为 0–100 分：晃动越小分越高。"
            "表示骨盆/核心在踩踏中是否稳定。越接近 100 越好；过低常见于座位高度不适或核心控制不足。"
        ),
    },
}


def _angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1) + 1e-9, np.linalg.norm(v2) + 1e-9
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _safe(j: np.ndarray, idx: int) -> np.ndarray | None:
    if j is None or idx >= len(j):
        return None
    p = j[idx]
    if np.any(np.isnan(p)):
        return None
    return p


def compute_metrics(frames: list[dict], fps: float = 10.0) -> dict[str, Any]:
    ankles_l, ankles_r = [], []
    knee_l, knee_r = [], []
    torso_lean = []
    hip_y = []
    valid = 0

    for fr in frames:
        raw = fr.get("joints_xyz")
        if not raw:
            continue
        j = np.asarray(raw, dtype=float)
        valid += 1
        la, ra = _safe(j, L_ANK), _safe(j, R_ANK)
        lk, rk = _safe(j, L_KNE), _safe(j, R_KNE)
        lh, rh = _safe(j, L_HIP), _safe(j, R_HIP)
        ls, rs = _safe(j, L_SHO), _safe(j, R_SHO)
        if la is not None:
            ankles_l.append(la)
        if ra is not None:
            ankles_r.append(ra)
        if lh is not None and lk is not None and la is not None:
            knee_l.append(_angle(lh, lk, la))
        if rh is not None and rk is not None and ra is not None:
            knee_r.append(_angle(rh, rk, ra))
        if lh is not None and rh is not None and ls is not None and rs is not None:
            hip = 0.5 * (lh + rh)
            sho = 0.5 * (ls + rs)
            torso = sho - hip
            lean = float(np.degrees(np.arctan2(torso[0], max(torso[1], 1e-6))))
            torso_lean.append(abs(lean))
            hip_y.append(float(hip[1]))

    def series_stats(xs: list[float]) -> dict:
        if not xs:
            return {"mean": None, "min": None, "max": None, "rom": None}
        a = np.asarray(xs, dtype=float)
        return {
            "mean": round(float(np.mean(a)), 2),
            "min": round(float(np.min(a)), 2),
            "max": round(float(np.max(a)), 2),
            "rom": round(float(np.ptp(a)), 2),
        }

    cadence_rpm = None
    if len(ankles_l) > fps:
        y = np.asarray([p[1] for p in ankles_l], dtype=float)
        y = y - np.mean(y)
        spec = np.abs(np.fft.rfft(y - y.mean()))
        freqs = np.fft.rfftfreq(len(y), d=1.0 / fps)
        if len(freqs) > 2:
            band = (freqs > 0.4) & (freqs < 3.0)
            if np.any(band):
                i = int(np.argmax(spec[band]))
                f0 = float(freqs[band][i])
                cadence_rpm = round(f0 * 60.0, 1)

    def circularity(pts: list[np.ndarray]) -> float | None:
        if len(pts) < 12:
            return None
        p = np.asarray(pts)[:, [0, 1]]
        c = p.mean(0)
        r = np.linalg.norm(p - c, axis=1)
        cv = float(np.std(r) / (np.mean(r) + 1e-9))
        return round(float(np.clip(1.0 - cv, 0, 1) * 100), 1)

    sym_knee = None
    if knee_l and knee_r:
        sym_knee = float(
            round(100.0 - min(100.0, abs(float(np.mean(knee_l)) - float(np.mean(knee_r))) * 1.2), 1)
        )

    hip_stability = None
    if hip_y:
        hip_stability = float(round(float(max(0.0, 100.0 - float(np.std(hip_y)) * 400)), 1))

    metrics: dict[str, Any] = {
        "cadence_rpm": cadence_rpm,
        "knee_left_deg": series_stats(knee_l),
        "knee_right_deg": series_stats(knee_r),
        "knee_symmetry_pct": sym_knee,
        "torso_lean_deg": series_stats(torso_lean),
        "ankle_path_circularity_l_pct": circularity(ankles_l),
        "ankle_path_circularity_r_pct": circularity(ankles_r),
        "hip_stability_pct": hip_stability,
        "coverage_pct": round(100.0 * valid / max(1, len(frames)), 1),
        "duration_sec": round(len(frames) / max(fps, 1e-6), 2),
        "fps": fps,
        "guides": METRIC_GUIDES,
    }

    parts = [x for x in [sym_knee, metrics["ankle_path_circularity_l_pct"], hip_stability] if x is not None]
    metrics["form_index"] = float(round(float(np.mean(parts)), 1)) if parts else None

    bullets = []
    if cadence_rpm:
        bullets.append(f"分析时段估计踏频约 {cadence_rpm:.0f} rpm。")
    if sym_knee is not None:
        if sym_knee >= 85:
            level = "较均衡"
        elif sym_knee >= 70:
            level = "轻度不对称"
        else:
            level = "明显不对称"
        bullets.append(f"左右膝模式{level}（对称性 {sym_knee:.0f}/100）。")
    if metrics["torso_lean_deg"]["mean"] is not None:
        bullets.append(
            f"躯干前倾角均值约 {metrics['torso_lean_deg']['mean']:.1f}°"
            f"（活动幅度 {metrics['torso_lean_deg']['rom']:.1f}°）。"
        )
    if metrics["ankle_path_circularity_l_pct"] is not None:
        bullets.append(
            f"左踝轨迹圆滑度约 {metrics['ankle_path_circularity_l_pct']:.0f}/100"
            "（越高表示踩踏圆周更平滑）。"
        )
    if hip_stability is not None:
        bullets.append(f"骨盆稳定性约 {hip_stability:.0f}/100。")
    bullets.append(
        "重建模式：单镜头 2D 关键点抬升为相对尺度三维骨架，适合动作模式回顾，"
        "非多机几何测量。"
    )
    metrics["narrative"] = bullets
    return metrics
