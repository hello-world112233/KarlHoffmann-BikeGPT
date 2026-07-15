# Next Steps

这份清单只回答一个问题：接下来怎么一步一步做。

## 现在的优先级

当前优先级不是完整网页，而是第一条数据-模型闭环：

```text
真实视频 -> 登记 -> 抽帧 -> Sapiens2 baseline -> 失败样本 -> 标注准备
```

## Step 1：准备本地环境

在 VS Code 终端里进入项目根目录：

```bash
cd /Users/chloe/Documents/bike-mac/bike-ai-platform
```

创建环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

如果你暂时不想装完整环境，也可以先用系统 Python 跑轻量脚本。

## Step 2：拿一个真实视频做第一条样本

把一个测试视频放到：

```text
data/raw_videos/
```

例如：

```text
data/raw_videos/test_start_side.mp4
```

## Step 3：登记视频

示例命令：

```bash
python scripts/register_video.py data/raw_videos/test_start_side.mp4 \
  --video-id CX20260715_A001_START_SIDE_001 \
  --athlete-id A001 \
  --scene standing_start \
  --camera-view side \
  --occlusion mild \
  --coach-note "第一条测试样本，侧面机位，含原地出发"
```

成功后会生成：

```text
data/registry/CX20260715_A001_START_SIDE_001.json
```

## Step 4：抽帧

```bash
python scripts/extract_frames.py data/raw_videos/test_start_side.mp4 \
  --out-dir data/frames/CX20260715_A001_START_SIDE_001 \
  --fps 5
```

成功后你会看到：

```text
data/frames/CX20260715_A001_START_SIDE_001/frame_000000.jpg
data/frames/CX20260715_A001_START_SIDE_001/frame_000001.jpg
...
```

## Step 5：同步到 AutoDL

当前建议先用 Git 同步代码，用 `scp` 或 VS Code Remote SSH 上传少量测试视频/抽帧。

长期做法：

```text
Mac:    /Users/chloe/Documents/bike-mac/bike-ai-platform
AutoDL: /root/autodl-tmp/bike-ai-platform
Data:   /root/autodl-tmp/bike-ai-data
```

## Step 6：实现 Sapiens2 baseline

需要在 AutoDL 上完成：

- 确认 Sapiens2 repo 和 checkpoint 路径
- 实现 `src/bike_ai/models/sapiens2.py`
- 让 `scripts/run_baseline.py` 能读取抽帧目录并输出推理结果

第一版输出目标：

```text
data/inference/<video_id>/sapiens2/
├── predictions.json
├── coco17.json
├── overlay_frames/
└── run_meta.json
```

## Step 7：人工看失败样本

第一阶段不用做复杂 UI，先把 overlay 图片打开看，记录：

- 哪些帧错了
- 错在哪里
- 错误类型是什么
- 明天要不要补采同类场景

## 近期不要做什么

先不要做：

- 登录系统
- 复杂 dashboard
- 很漂亮的首页
- 多用户权限
- 自动训练全流程

这些以后会需要，但现在还不是最短路径。

## 近期必须做什么

必须做：

- 数据登记规范
- 抽帧
- baseline
- 失败样本
- 标注准备
- 实验记录

它们会直接决定你后面能不能训练出自行车专项模型。

