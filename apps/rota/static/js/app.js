import { RotaViewer } from "./viewer3d.js?v=20260820a";

const CALIBRATION_KEYS = ["rear_hub", "front_hub", "bottom_bracket", "saddle", "handlebar"];
const BODY_CALIBRATION_KEYS = [
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle",
];
const BODY_GUIDE_EDGES = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
];

async function loadAnalysis(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error("分析数据加载失败");
  return r.json();
}

function fmt(v, digits = 1) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function guideOf(m, key) {
  return (m.guides && m.guides[key]) || {};
}

function metricCard(key, label, value, unit, hint, guide) {
  const tip = guide || "暂无说明";
  return `<div class="metric has-tip" tabindex="0" data-key="${esc(key)}">
    <div class="label">
      <span>${esc(label)}</span>
      <span class="tip-mark" aria-hidden="true">?</span>
    </div>
    <div class="value">${value}<span class="unit">${esc(unit)}</span></div>
    <div class="hint">${esc(hint)}</div>
    <div class="tip-bubble" role="tooltip">${esc(tip)}</div>
  </div>`;
}

function renderPipeline(steps) {
  document.getElementById("pipeline").innerHTML = (steps || [])
    .map((s) => `<div class="pipe-step ${s.state}"><div class="dot"></div><div>${esc(s.label)}</div></div>`)
    .join("");
}

function renderMetrics(m) {
  const g = (k) => guideOf(m, k);
  const geometryCards = m.geometry_quality
    ? [
        metricCard(
          "hard_contacts_passed",
          "人车硬约束",
          m.geometry_quality.hard_contacts_passed ? "通过" : "未通过",
          "",
          "双手、双脚和曲柄联动检查",
          "左右手必须分别位于左右握把；左右脚必须位于车架两侧的脚踏上；两侧曲柄必须保持 180° 反相。"
        ),
        metricCard(
          "reprojection_rmse_px",
          "二维重投影误差",
          fmt(m.geometry_quality.reprojection_rmse_px, 1),
          "px",
          "表示对二维检测的主动纠偏量",
          "物理约束后的三维关节与原始二维检测之间的均方根差异。检测点违反骑行逻辑时，重建器会主动偏离它，因此该值不是越低就一定越好。"
        ),
        metricCard(
          "bone_length_cv_max_pct",
          "最大骨长波动",
          fmt(m.geometry_quality.bone_length_cv_max_pct, 1),
          "%",
          "越低越接近刚性骨架",
          "统计各肢段长度随时间的变异系数，并显示其中最大值。正常骨架的骨长不应逐帧伸缩。"
        ),
        metricCard(
          "hand_to_handlebar_pct_wheelbase",
          "手—车把误差",
          fmt(m.geometry_quality.hand_to_handlebar_pct_wheelbase, 1),
          "%轴距",
          "硬约束目标为 0",
          "左右手分别与左右握把的接触误差，以自行车轴距为尺度。新版运动学求解器将其作为不可违反的硬约束。"
        ),
        metricCard(
          "crank_opposition_error_pct_wheelbase",
          "曲柄对侧误差",
          fmt(m.geometry_quality.crank_opposition_error_pct_wheelbase, 1),
          "%轴距",
          "硬约束目标为 0",
          "衡量左右脚踏相对中轴是否严格反相。新版求解器由同一个曲柄相位生成两侧脚踏，因此始终相差 180°。"
        ),
      ]
    : [];
  if (m.geometry_quality?.body_anchor_rmse_px != null) {
    geometryCards.splice(
      1,
      0,
      metricCard(
        "body_anchor_rmse_px",
        "人工人体锚点误差",
        fmt(m.geometry_quality.body_anchor_rmse_px, 1),
        "px",
        "越低越贴近手工标注",
        "标定帧中约束后三维关节重新投影到画面后，与人工拖动的 12 个关节点之间的均方根误差。"
      )
    );
  }
  const cards = geometryCards.concat([
    metricCard(
      "cadence_rpm",
      g("cadence_rpm").label || "踏频",
      fmt(m.cadence_rpm, 0),
      "rpm",
      g("cadence_rpm").hint || "每分钟转数",
      g("cadence_rpm").guide
    ),
    metricCard(
      "knee_left_deg",
      g("knee_left_deg").label || "左膝活动度",
      fmt(m.knee_left_deg?.rom, 0),
      "°",
      `均值 ${fmt(m.knee_left_deg?.mean)}°`,
      g("knee_left_deg").guide
    ),
    metricCard(
      "knee_right_deg",
      g("knee_right_deg").label || "右膝活动度",
      fmt(m.knee_right_deg?.rom, 0),
      "°",
      `均值 ${fmt(m.knee_right_deg?.mean)}°`,
      g("knee_right_deg").guide
    ),
    metricCard(
      "knee_symmetry_pct",
      g("knee_symmetry_pct").label || "左右膝对称性",
      fmt(m.knee_symmetry_pct, 0),
      "%",
      g("knee_symmetry_pct").hint || "越高越对称",
      g("knee_symmetry_pct").guide
    ),
    metricCard(
      "torso_lean_deg",
      g("torso_lean_deg").label || "躯干前倾角",
      fmt(m.torso_lean_deg?.mean, 1),
      "°",
      `活动幅度 ${fmt(m.torso_lean_deg?.rom)}°`,
      g("torso_lean_deg").guide
    ),
    metricCard(
      "ankle_path_circularity_l_pct",
      g("ankle_path_circularity_l_pct").label || "左踝轨迹圆滑度",
      fmt(m.ankle_path_circularity_l_pct, 0),
      "%",
      g("ankle_path_circularity_l_pct").hint || "越高越圆",
      g("ankle_path_circularity_l_pct").guide
    ),
    metricCard(
      "ankle_path_circularity_r_pct",
      g("ankle_path_circularity_r_pct").label || "右踝轨迹圆滑度",
      fmt(m.ankle_path_circularity_r_pct, 0),
      "%",
      g("ankle_path_circularity_r_pct").hint || "越高越圆",
      g("ankle_path_circularity_r_pct").guide
    ),
    metricCard(
      "hip_stability_pct",
      g("hip_stability_pct").label || "骨盆稳定性",
      fmt(m.hip_stability_pct, 0),
      "%",
      g("hip_stability_pct").hint || "上下晃动越小越高",
      g("hip_stability_pct").guide
    ),
  ]);

  const formG = g("form_index");
  const hero = m.form_index == null && m.geometry_quality
    ? `<div class="form-hero">
        <div>
          <div class="label" style="font-family:var(--font-mono);font-size:.7rem;letter-spacing:.08em;opacity:.7">几何质检模式</div>
          <div class="big">约束后<span>非临床结论</span></div>
        </div>
        <div style="color:var(--mist);max-width:18rem;font-size:.9rem;line-height:1.45">
          优先检查重投影、固定骨长与人车接触；车辆与人体确认前不输出综合评分
        </div>
      </div>`
    : `<div class="form-hero has-tip" tabindex="0">
      <div>
        <div class="label" style="font-family:var(--font-mono);font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;opacity:.7">
          ${esc(formG.label || "动作质量指数")}
          <span class="tip-mark" aria-hidden="true">?</span>
        </div>
        <div class="big">${fmt(m.form_index, 0)}<span>/100</span></div>
      </div>
      <div style="color:var(--mist);max-width:16rem;font-size:.9rem;line-height:1.45">
        ${esc(formG.hint || "综合评分")} · 悬停查看算法与解读
      </div>
      <div class="tip-bubble" role="tooltip">${esc(formG.guide || "")}</div>
    </div>`;
  document.getElementById("metricsGrid").innerHTML = hero + cards.join("");

  document.getElementById("narrative").innerHTML = (m.narrative || []).map((t) => `<li>${esc(t)}</li>`).join("");
  document.getElementById("disclaimer").textContent = m._disclaimer || "";
}

function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2800);
}

function syncVideoToFrame(video, frameIdx, fps) {
  if (!video || !video.duration || !Number.isFinite(video.duration)) return;
  const t = Math.min(Math.max(0, frameIdx / Math.max(fps, 1e-6)), Math.max(0, video.duration - 0.05));
  if (Math.abs(video.currentTime - t) > 0.08) {
    try {
      video.currentTime = t;
    } catch (_) {
      /* ignore seek races while loading */
    }
  }
}

async function boot() {
  const viewer = new RotaViewer(document.getElementById("stage3d"));
  const video = document.getElementById("sourceVideo");
  const videoViewport = video.closest(".video-viewport");
  const calibrationOverlay = document.getElementById("calibrationOverlay");
  const bikeGuide = document.getElementById("bikeGuide");
  const bodyCalibrationOverlay = document.getElementById("bodyCalibrationOverlay");
  const bodyGuide = document.getElementById("bodyGuide");
  const calibrationStatus = document.getElementById("calibrationStatus");
  const btnCalibrate = document.getElementById("btnCalibrate");
  const btnApplyCalibration = document.getElementById("btnApplyCalibration");
  const btnBodyCalibrate = document.getElementById("btnBodyCalibrate");
  const btnApplyBodyCalibration = document.getElementById("btnApplyBodyCalibration");
  let analysis = await loadAnalysis("/api/demo");
  let syncingFromVideo = false;
  let calibration = null;
  let calibrationEditing = false;
  let draggingPoint = null;
  let bodyCalibration = null;
  let bodyTracking = null;
  let bodyCalibrationEditing = false;
  let draggingBodyPoint = null;

  function videoContentRect() {
    const width = videoViewport.clientWidth;
    const height = videoViewport.clientHeight;
    const sourceWidth = video.videoWidth || 16;
    const sourceHeight = video.videoHeight || 9;
    const scale = Math.min(width / sourceWidth, height / sourceHeight);
    const renderedWidth = sourceWidth * scale;
    const renderedHeight = sourceHeight * scale;
    return {
      left: (width - renderedWidth) / 2,
      top: (height - renderedHeight) / 2,
      width: renderedWidth,
      height: renderedHeight,
    };
  }

  function renderCalibration() {
    if (!calibration?.points) return;
    const rect = videoContentRect();
    const pixels = {};
    for (const key of CALIBRATION_KEYS) {
      const point = calibration.points[key];
      const x = rect.left + point[0] * rect.width;
      const y = rect.top + point[1] * rect.height;
      pixels[key] = [x, y];
      const handle = calibrationOverlay.querySelector(`[data-point="${key}"]`);
      handle.style.left = `${x}px`;
      handle.style.top = `${y}px`;
    }
    const p = (key) => pixels[key].map((value) => value.toFixed(1)).join(",");
    const wheelRadius = Math.hypot(
      pixels.front_hub[0] - pixels.rear_hub[0],
      pixels.front_hub[1] - pixels.rear_hub[1]
    ) * 0.32;
    bikeGuide.setAttribute("viewBox", `0 0 ${videoViewport.clientWidth} ${videoViewport.clientHeight}`);
    bikeGuide.innerHTML = `
      <g fill="none" stroke="rgba(184,240,110,.82)" stroke-width="2">
        <circle cx="${pixels.rear_hub[0]}" cy="${pixels.rear_hub[1]}" r="${wheelRadius}" />
        <circle cx="${pixels.front_hub[0]}" cy="${pixels.front_hub[1]}" r="${wheelRadius}" />
        <polyline points="${p("rear_hub")} ${p("saddle")} ${p("bottom_bracket")} ${p("rear_hub")}" />
        <polyline points="${p("saddle")} ${p("handlebar")} ${p("bottom_bracket")} ${p("front_hub")} ${p("handlebar")}" />
      </g>`;
  }

  function renderBodyCalibration() {
    const onReferenceFrame = bodyCalibration?.reference_frame === viewer.idx;
    const visible = !!bodyCalibration?.points && (bodyCalibrationEditing || onReferenceFrame);
    bodyCalibrationOverlay.classList.toggle("visible", visible);
    if (!visible) return;
    const rect = videoContentRect();
    const pixels = {};
    for (const key of BODY_CALIBRATION_KEYS) {
      const point = bodyCalibration.points[key];
      if (!point) continue;
      const x = rect.left + point[0] * rect.width;
      const y = rect.top + point[1] * rect.height;
      pixels[key] = [x, y];
      const handle = bodyCalibrationOverlay.querySelector(`[data-body-point="${key}"]`);
      handle.style.left = `${x}px`;
      handle.style.top = `${y}px`;
    }
    bodyGuide.setAttribute("viewBox", `0 0 ${videoViewport.clientWidth} ${videoViewport.clientHeight}`);
    bodyGuide.innerHTML = BODY_GUIDE_EDGES
      .filter(([a, b]) => pixels[a] && pixels[b])
      .map(([a, b]) => `<line x1="${pixels[a][0]}" y1="${pixels[a][1]}" x2="${pixels[b][0]}" y2="${pixels[b][1]}" />`)
      .join("");
  }

  function setBodyZoomOrigin() {
    if (!bodyCalibration?.points) return;
    const rect = videoContentRect();
    const points = BODY_CALIBRATION_KEYS
      .map((key) => bodyCalibration.points[key])
      .filter(Boolean);
    if (!points.length) return;
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    const centerX = rect.left + 0.5 * (Math.min(...xs) + Math.max(...xs)) * rect.width;
    const centerY = rect.top + 0.5 * (Math.min(...ys) + Math.max(...ys)) * rect.height;
    videoViewport.style.setProperty("--body-zoom-x", `${centerX}px`);
    videoViewport.style.setProperty("--body-zoom-y", `${centerY}px`);
  }

  function updateCalibrationStatus() {
    if (calibrationEditing) {
      calibrationStatus.textContent = "拖动车辆五点；完成后点击“应用约束”";
    } else if (bodyCalibrationEditing) {
      calibrationStatus.textContent = "拖动肩、肘、腕、髋、膝、踝；完成后点击“应用人体”";
    } else {
      const bikeState = calibration?.confirmed ? "车辆已确认" : "车辆未确认";
      const anchorError = analysis?.metrics?.geometry_quality?.body_anchor_rmse_px;
      const bodyState = bodyCalibration?.confirmed
        ? `人体已标定（第 ${bodyCalibration.reference_frame + 1} 帧${anchorError != null ? ` · 锚点 ${fmt(anchorError, 1)}px` : ""}）`
        : "人体未标定";
      calibrationStatus.textContent = `${bikeState} · ${bodyState}`;
    }
  }

  function setCalibrationEditing(value) {
    if (value) setBodyCalibrationEditing(false);
    calibrationEditing = value;
    calibrationOverlay.classList.toggle("editing", value);
    btnApplyCalibration.disabled = !value;
    btnCalibrate.textContent = value ? "结束拖动" : "校正五点";
    updateCalibrationStatus();
  }

  function setBodyCalibrationEditing(value) {
    if (value && calibrationEditing) {
      calibrationEditing = false;
      calibrationOverlay.classList.remove("editing");
      btnApplyCalibration.disabled = true;
      btnCalibrate.textContent = "校正五点";
    }
    bodyCalibrationEditing = value;
    if (value) setBodyZoomOrigin();
    videoViewport.classList.toggle("body-zoom", value);
    bodyCalibrationOverlay.classList.toggle("editing", value);
    btnApplyBodyCalibration.disabled = !value;
    btnBodyCalibrate.textContent = value
      ? "结束人体拖动"
      : bodyCalibration?.confirmed
        ? "重标人体"
        : "人体标定";
    renderBodyCalibration();
    updateCalibrationStatus();
  }

  function updateDraggedPoint(event) {
    if (!draggingPoint || !calibrationEditing || !calibration?.points) return;
    const overlayRect = calibrationOverlay.getBoundingClientRect();
    const content = videoContentRect();
    const localX = event.clientX - overlayRect.left;
    const localY = event.clientY - overlayRect.top;
    calibration.points[draggingPoint] = [
      Math.min(1, Math.max(0, (localX - content.left) / content.width)),
      Math.min(1, Math.max(0, (localY - content.top) / content.height)),
    ];
    calibration.confirmed = false;
    renderCalibration();
  }

  calibrationOverlay.querySelectorAll(".calibration-point").forEach((handle) => {
    handle.addEventListener("pointerdown", (event) => {
      if (!calibrationEditing) return;
      draggingPoint = handle.dataset.point;
      handle.setPointerCapture(event.pointerId);
      event.preventDefault();
    });
    handle.addEventListener("pointermove", updateDraggedPoint);
    handle.addEventListener("pointerup", (event) => {
      if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
      draggingPoint = null;
    });
    handle.addEventListener("pointercancel", () => {
      draggingPoint = null;
    });
  });

  function captureBodyCalibration() {
    const tracked = bodyTracking?.frames?.[viewer.idx];
    if (!tracked?.points || !BODY_CALIBRATION_KEYS.every((key) => tracked.points[key])) {
      throw new Error("当前帧缺少完整人体关键点");
    }
    bodyCalibration = {
      version: 1,
      confirmed: false,
      reference_frame: viewer.idx,
      points: Object.fromEntries(
        BODY_CALIBRATION_KEYS.map((key) => [key, [...tracked.points[key]]])
      ),
    };
  }

  function updateDraggedBodyPoint(event) {
    if (!draggingBodyPoint || !bodyCalibrationEditing || !bodyCalibration?.points) return;
    const overlayRect = bodyCalibrationOverlay.getBoundingClientRect();
    const content = videoContentRect();
    const zoomX = overlayRect.width / Math.max(1, videoViewport.clientWidth);
    const zoomY = overlayRect.height / Math.max(1, videoViewport.clientHeight);
    const localX = (event.clientX - overlayRect.left) / zoomX;
    const localY = (event.clientY - overlayRect.top) / zoomY;
    bodyCalibration.points[draggingBodyPoint] = [
      Math.min(1, Math.max(0, (localX - content.left) / content.width)),
      Math.min(1, Math.max(0, (localY - content.top) / content.height)),
    ];
    bodyCalibration.confirmed = false;
    renderBodyCalibration();
  }

  bodyCalibrationOverlay.querySelectorAll(".body-calibration-point").forEach((handle) => {
    handle.addEventListener("pointerdown", (event) => {
      if (!bodyCalibrationEditing) return;
      draggingBodyPoint = handle.dataset.bodyPoint;
      handle.setPointerCapture(event.pointerId);
      event.preventDefault();
    });
    handle.addEventListener("pointermove", updateDraggedBodyPoint);
    handle.addEventListener("pointerup", (event) => {
      if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
      draggingBodyPoint = null;
    });
    handle.addEventListener("pointercancel", () => {
      draggingBodyPoint = null;
    });
  });

  function setPlaying(playing) {
    viewer.playing = playing;
    document.getElementById("btnPlay").textContent = playing ? "暂停" : "播放";
    if (!video.src) return;
    if (playing) {
      const p = video.play();
      if (p && typeof p.catch === "function") p.catch(() => {});
    } else {
      video.pause();
    }
  }

  function apply(a) {
    analysis = a;
    a.metrics._disclaimer = a.disclaimer || a.note || "";
    document.getElementById("sessionTitle").textContent = a.title;
    document.getElementById("sessionSub").textContent =
      `${a.source} · ${a.n_frames} 帧 @ ${a.fps} fps`;
    renderPipeline(a.pipeline);
    renderMetrics(a.metrics);
    viewer.setFrames(a.frames, a.fps, a.bike_geometry, a.coordinate_system);
    calibration = JSON.parse(JSON.stringify(a.bike_calibration || calibration || {}));
    bodyCalibration = a.body_calibration
      ? JSON.parse(JSON.stringify(a.body_calibration))
      : null;
    bodyTracking = a.body_tracking || bodyTracking;
    renderCalibration();
    setCalibrationEditing(false);
    setBodyCalibrationEditing(false);
    renderBodyCalibration();
    document.getElementById("scrub").max = Math.max(0, (a.frames || []).length - 1);
    document.getElementById("scrubVal").textContent = `0 / ${Math.max(0, a.n_frames - 1)}`;

    if (a.video_url) {
      if (video.getAttribute("src") !== a.video_url) {
        video.src = a.video_url;
        video.load();
      }
    } else {
      video.removeAttribute("src");
      video.load();
    }
    setPlaying(true);
  }

  apply(analysis);

  viewer.onFrame = (i) => {
    document.getElementById("scrub").value = String(i);
    document.getElementById("scrubVal").textContent = `${i} / ${analysis.n_frames - 1}`;
    if (!syncingFromVideo) syncVideoToFrame(video, i, analysis.fps || 10);
    renderBodyCalibration();
  };

  document.getElementById("scrub").addEventListener("input", (e) => {
    const i = Number(e.target.value);
    setPlaying(false);
    viewer.showFrame(i);
    syncVideoToFrame(video, i, analysis.fps || 10);
  });

  document.getElementById("btnPlay").addEventListener("click", () => {
    setPlaying(!viewer.playing);
  });
  document.getElementById("btnPrint").addEventListener("click", () => window.print());

  btnCalibrate.addEventListener("click", () => {
    setPlaying(false);
    setCalibrationEditing(!calibrationEditing);
  });
  btnApplyCalibration.addEventListener("click", async () => {
    if (!calibration?.points) return;
    try {
      btnApplyCalibration.disabled = true;
      calibrationStatus.textContent = "正在执行固定骨长与人车约束优化…";
      toast("正在优化 T014 三维序列…");
      const response = await fetch("/api/demo/calibration", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: calibration.points, confirmed: true }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.message || "优化失败");
      apply(result.analysis);
      toast("五点已保存，约束优化完成");
    } catch (error) {
      console.error(error);
      calibrationStatus.textContent = "优化失败，请检查服务日志";
      btnApplyCalibration.disabled = false;
      toast("约束优化失败");
    }
  });

  btnBodyCalibrate.addEventListener("click", () => {
    setPlaying(false);
    if (!bodyCalibrationEditing) {
      try {
        if (!bodyCalibration?.points || bodyCalibration.reference_frame !== viewer.idx) {
          captureBodyCalibration();
        }
        setBodyCalibrationEditing(true);
      } catch (error) {
        console.error(error);
        toast(error.message || "当前帧无法标定");
      }
    } else {
      setBodyCalibrationEditing(false);
    }
  });

  btnApplyBodyCalibration.addEventListener("click", async () => {
    if (!bodyCalibration?.points) return;
    try {
      btnApplyBodyCalibration.disabled = true;
      calibrationStatus.textContent = "正在生成个体骨长模板并重优化整段视频…";
      toast("正在应用人体骨长标定…");
      const response = await fetch("/api/demo/body-calibration", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reference_frame: bodyCalibration.reference_frame,
          points: bodyCalibration.points,
          confirmed: true,
        }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) {
        throw new Error(result.detail || result.message || "人体标定优化失败");
      }
      apply(result.analysis);
      viewer.showFrame(bodyCalibration?.reference_frame || 0);
      setPlaying(false);
      const anchorError = result.quality?.quality_after?.body_anchor_rmse_px;
      toast(
        anchorError != null
          ? `人体标定生效 · 锚点误差 ${fmt(anchorError, 1)}px`
          : "人体模板已保存，整段姿态已重新优化"
      );
    } catch (error) {
      console.error(error);
      try {
        const recovered = await loadAnalysis("/api/demo");
        if (recovered.body_calibration?.confirmed) {
          apply(recovered);
          viewer.showFrame(recovered.body_calibration.reference_frame || 0);
          setPlaying(false);
          toast("计算已完成，页面已从服务器恢复结果");
          return;
        }
      } catch (recoveryError) {
        console.warn("标定结果暂时无法重新读取", recoveryError);
      }
      calibrationStatus.textContent = "连接中断；优化结果可能已保存，重新打开页面即可确认";
      btnApplyBodyCalibration.disabled = false;
      toast("连接中断，不代表优化失败");
    }
  });

  video.addEventListener("timeupdate", () => {
    if (viewer.playing || !video.duration) return;
    syncingFromVideo = true;
    const i = Math.min(
      analysis.n_frames - 1,
      Math.max(0, Math.round(video.currentTime * (analysis.fps || 10)))
    );
    viewer.showFrame(i);
    syncingFromVideo = false;
  });

  const fileInput = document.getElementById("file");
  document.getElementById("btnUpload").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", async () => {
    if (!fileInput.files?.length) return;
    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    toast("正在上传并跑检测（Sapiens2 → MotionBERT → 约束 3D），请稍候…");
    try {
      const r = await fetch("/api/jobs", { method: "POST", body: fd });
      const payload = await r.json().catch(() => ({}));
      if (!r.ok) {
        throw new Error(payload.detail || payload.error || `HTTP ${r.status}`);
      }
      apply(await loadAnalysis(`/api/jobs/${payload.job_id}`));
      toast("检测完成，场次已加载");
    } catch (e) {
      console.error(e);
      toast(`上传或检测失败：${e.message || e}`);
    } finally {
      fileInput.value = "";
    }
  });

  document.getElementById("btnDemo").addEventListener("click", async () => {
    try {
      toast("正在加载演示…");
      apply(await loadAnalysis("/api/demo"));
      toast("演示场次已加载");
    } catch (e) {
      console.error(e);
      toast("演示加载失败，请刷新重试");
    }
  });

  video.addEventListener("loadedmetadata", () => {
    renderCalibration();
    renderBodyCalibration();
  });
  window.addEventListener("resize", () => {
    renderCalibration();
    renderBodyCalibration();
  });
}

boot().catch((e) => {
  console.error(e);
  toast("工作台启动失败（三维引擎）。请强制刷新一次。");
});
