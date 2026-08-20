# 多视角三维重建工作流（交接用）

## 在整条链路里各模块的位置

```text
三机位视频
  → scripts/sync_cameras.py          # 音频自动对齐
  → Sapiens2 baseline（已有，AutoDL GPU）
  → scripts/select_athlete.py        # 只保留运动员
  → scripts/reconstruct_3d.py        # CAD 骨架 + 自行车约束多视角拟合
  → joints3d.json / angles.json
```

Sapiens2 **仍然是 2D 关键点前端**，不负责三维。三维由 `src/bike_ai/reconstruct/` 完成。

## 今天新增的代码

| 路径 | 作用 |
|---|---|
| `src/bike_ai/sync/audio_sync.py` | 音频互相关对齐 |
| `src/bike_ai/tracking/keypoints.py` | 读 Sapiens2 JSON，切 COCO-17 |
| `src/bike_ai/tracking/athlete.py` | 多人里选运动员 |
| `src/bike_ai/reconstruct/` | 骨架 CAD、自行车曲柄约束、多视角拟合、关节角 |
| `scripts/sync_cameras.py` | CLI：对齐 |
| `scripts/select_athlete.py` | CLI：选人 |
| `scripts/reconstruct_3d.py` | CLI：三维重建 |
| `scripts/run_multiview_3d.py` | 一键跑通一场 session |
| `configs/cameras_example.yaml` | 相机标定文件模板 |

## 机位命名原则

三个机位使用稳定的中性编号 `A / B / C`，不要求人工判断哪个是正面、侧面或后面。
相机在三维空间中的位置和朝向由 `cameras.yaml` 内真实标定得到的 `K / R / t` 决定。
只要视频的 A/B/C 身份与标定文件中的 A/B/C 一致，三维重建不需要语义视角名称。

## 你上传三视角后怎么跑

在 `bike-ai-data`（或任意目录）建一场：

```text
sessions/T001/
  camera_a/original.mp4
  camera_b/original.mp4
  camera_c/original.mp4
  cameras.yaml
  pose/A/sapiens2_predictions.json
  pose/B/sapiens2_predictions.json
  pose/C/sapiens2_predictions.json
```

然后：

```bash
cd /root/autodl-tmp/bike-ai-platform   # 或本仓库根目录
pip install -e ".[dev]"
# 需要: ffmpeg, scipy

python scripts/run_multiview_3d.py sessions/T001 \
  --fps 30 --height-m 1.78 --crank-m 0.170
```

BikeTrialMatcher 完成整个项目上传后，可以直接读取服务器端清单。先只检查数据是否到齐：

```bash
python scripts/run_project_manifest.py \
  --manifest /root/autodl-tmp/bike_projects/PROJECT/project_manifest.json \
  --cameras /root/autodl-tmp/bike_projects/PROJECT/cameras.yaml \
  --validate-only
```

验证通过后去掉 `--validate-only`，程序会自动逐个处理所有已确认、已选择且三机位完整的 Trial。服务器端不再进行文件配对。

输出在 `sessions/athleteA_start_001/output/`。

## 相机标定

`cameras.yaml` 必须是真实标定（OpenCV）。`configs/cameras_example.yaml` 只是格式示例，**不能**当真实外参加精度验收。标定文件可以直接把三台相机命名为 A、B、C；无需再转换成正面、侧面、后面。

标定脚本可以后续接到仓库；当前三维拟合入口已经预留好 K/R/t。

## 依赖

在 `pyproject.toml` 中新增了 `scipy`（优化与互相关）。音频对齐需要系统安装 `ffmpeg`。

## 测试

```bash
pytest tests/test_multiview_3d.py -q
```
