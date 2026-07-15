# Video Inventory Workflow

这一步是“来真的”的第一步：不是先写网页，而是把你已有的大量视频变成可判断、可取舍、可安排补采的数据资产。

## 目标

对 AutoDL 上的视频目录做一次自动盘点，输出：

- 每个视频的路径、大小、时长、帧率、分辨率
- 基础画面质量：亮度、模糊程度、运动变化
- 自动推断类别：固定座、比赛全程、真实训练、未知
- 首批 baseline 推荐列表
- 需要先切片的比赛全程视频

## 服务器目录建议

```text
/root/autodl-tmp/bike-ai-data/raw_videos/
├── trainer_static/
├── competition_full/
└── field_training/
```

比赛全程视频先放 `competition_full/`，不要急着按出发、弯道、冲刺分类。真正分类的是后面切出来的事件片段。

## 在 AutoDL 上运行

进入项目目录：

```bash
cd /root/autodl-tmp/bike-ai-platform
```

扫描视频：

```bash
python scripts/scan_videos.py \
  --root /root/autodl-tmp/bike-ai-data/raw_videos \
  --out-dir /root/autodl-tmp/bike-ai-data/registry/video_inventory \
  --sample-frames 24
```

输出：

```text
/root/autodl-tmp/bike-ai-data/registry/video_inventory/video_inventory.csv
/root/autodl-tmp/bike-ai-data/registry/video_inventory/video_inventory.jsonl
/root/autodl-tmp/bike-ai-data/registry/video_inventory/video_inventory_report.md
```

## 如何读报告

先打开：

```text
video_inventory_report.md
```

重点看：

1. `competition_full` 有多少。
2. 哪些视频被推荐为 first batch。
3. 哪些视频备注里有 `full_video_should_be_indexed_and_clipped`。
4. 哪些视频是 `unknown`，需要手动整理。

## 下一步

1. 先挑 5-10 个固定座视频跑 baseline。
2. 对比赛全程视频先做事件索引，不要整场都跑。
3. 从比赛全程里切出出发、冲刺、遮挡、远景片段。
4. 对切出来的片段跑 Sapiens2/RTMPose baseline。
5. 根据失败类型决定长兴需要补采什么。

