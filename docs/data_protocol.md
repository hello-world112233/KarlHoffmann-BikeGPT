# Data Protocol

## 视频登记必填字段

- `video_id`：唯一编号，例如 `CX20260715_A001_START_SIDE_001`
- `athlete_id`：匿名运动员编号，不使用真实姓名
- `session_date`：采集日期
- `location`：采集地点
- `scene`：训练场景，例如 `standing_start`、`rolling_start`、`sprint`、`aero_descent`
- `camera_view`：机位，例如 `side`、`front`、`rear_oblique`、`high_side`
- `fps`：帧率
- `resolution`：分辨率
- `duration_sec`：视频时长
- `occlusion_level`：遮挡等级，`none`、`mild`、`moderate`、`severe`
- `lighting`：光照情况
- `coach_note`：教练或现场观察备注
- `consent_scope`：使用范围，例如 `internal_research`、`training_service`

## 目录约定

```text
data/
├── raw_videos/
├── registry/
├── frames/
├── inference/
├── annotations/
└── review/
```

每条视频必须能从 `video_id` 找到原始视频、抽帧、推理结果、标注和复盘材料。

