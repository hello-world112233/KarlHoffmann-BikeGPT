# Bike AI Platform

面向场地自行车训练场景的 AI 姿态分析、数据集建设与模型改进平台。

这个项目不是一次性 demo。它的目标是建立一个长期可用的闭环系统：

```text
真实采集 -> 数据入库 -> baseline 推理 -> 失败样本分析
       -> 人工标注 -> 专项训练集 -> 模型改进 -> 教练复盘
```

## 当前阶段目标

驻队期间优先完成 `v0.1` 到 `v0.2`：

- 建立视频数据登记和目录规范
- 形成长兴驻队采集协议
- 接入 Sapiens2/RTMPose baseline 推理入口
- 保存模型输出、可视化和失败样本
- 为后续人工标注和自行车专项模型训练打基础

## 项目结构

```text
bike-ai-platform/
├── apps/                 # 后续 Web/API 产品入口
├── configs/              # 本地、AutoDL、模型配置
├── data/                 # 本地数据根目录，不进 Git
├── datasets/             # 数据集说明、版本记录，不放大文件
├── docs/                 # 产品、采集、标注、科研路线文档
├── experiments/          # 每次 baseline/训练/评估实验记录
├── notebooks/            # 分析 notebook
├── scripts/              # 命令行入口
├── src/bike_ai/          # 长期可复用 Python 包
└── tests/                # 单元测试
```

## 开发原则

1. 代码进 Git，原始视频、模型权重、大文件不进 Git。
2. 每条视频必须先登记元数据，再进入推理和标注流程。
3. 每次模型推理必须记录模型版本、配置、输入数据版本和输出路径。
4. 失败样本不是垃圾，是模型改进和论文问题的来源。
5. Web 页面只服务闭环，不为了展示而展示。

## 系统数据模型

本项目采用三层分类法：

```text
来源域 Domain -> 动作/事件 Event -> 视角与质量 View & Quality
```

说明见 [docs/system_data_model.md](docs/system_data_model.md)。

## AutoDL 建议路径

```text
/root/autodl-tmp/bike-ai-platform
```

建议将大数据和模型放在：

```text
/root/autodl-tmp/bike-ai-data
/root/autodl-tmp/bike-ai-models
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/scan_videos.py --help
python scripts/propose_events.py --help
python scripts/create_event_index.py --help
python scripts/cut_clips.py --help
python scripts/register_video.py --help
python scripts/extract_frames.py --help
python scripts/run_baseline.py --help
```

## 当前最重要工作流

先对已有固定座视频和比赛全程视频做盘点：

```bash
python scripts/scan_videos.py \
  --root /root/autodl-tmp/bike-ai-data/raw_videos \
  --out-dir /root/autodl-tmp/bike-ai-data/registry/video_inventory
```

说明见 [docs/video_inventory_workflow.md](docs/video_inventory_workflow.md)。

比赛全程视频先建立事件索引，再切短片段：

```bash
python scripts/propose_events.py \
  --inventory-csv /root/autodl-tmp/bike-ai-data/registry/video_inventory/video_inventory.csv \
  --out-csv /root/autodl-tmp/bike-ai-data/registry/event_index/auto_competition_events.csv
```

如果需要手动精修，再生成手工索引模板：

```bash
python scripts/create_event_index.py \
  --inventory-csv /root/autodl-tmp/bike-ai-data/registry/video_inventory/video_inventory.csv \
  --out-csv /root/autodl-tmp/bike-ai-data/registry/event_index/competition_events.csv
```

说明见 [docs/event_index_workflow.md](docs/event_index_workflow.md)。
自动粗筛说明见 [docs/auto_event_proposal.md](docs/auto_event_proposal.md)。
