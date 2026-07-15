# Auto Event Proposal

手动看完整比赛视频效率太低。第一版自动索引工具会先从视频信号本身提取候选片段：

- 开头窗口：通常包含出发或启动准备。
- 高运动窗口：可能对应加速、冲刺、弯道或多人快速变化。
- 低清晰度 + 高运动窗口：可能是 hard case，例如运动模糊、远景、遮挡。

它不是最终的智能理解模型，而是“第一遍自动粗筛”。目标是让人只看少量候选片段，而不是看整场比赛。

## 运行

```bash
python scripts/propose_events.py \
  --inventory-csv /root/autodl-tmp/bike-ai-data/registry/video_inventory/video_inventory.csv \
  --out-csv /root/autodl-tmp/bike-ai-data/registry/event_index/auto_competition_events.csv \
  --clip-seconds 12 \
  --stride-seconds 4 \
  --max-events-per-video 4 \
  --sample-fps 2
```

## 切片

```bash
python scripts/cut_clips.py \
  --event-csv /root/autodl-tmp/bike-ai-data/registry/event_index/auto_competition_events.csv \
  --out-root /root/autodl-tmp/bike-ai-data/clips/competition_video_auto \
  --overwrite
```

如果切片边界不准：

```bash
python scripts/cut_clips.py \
  --event-csv /root/autodl-tmp/bike-ai-data/registry/event_index/auto_competition_events.csv \
  --out-root /root/autodl-tmp/bike-ai-data/clips/competition_video_auto \
  --reencode \
  --overwrite
```

## 后续升级

下一阶段自动索引可以接入：

- 人/车检测：确认画面里运动员足够大。
- 姿态 baseline：根据关键点置信度和抖动找 hard cases。
- 赛道几何：区分直道和弯道。
- OCR/音频/发令事件：辅助定位出发。

