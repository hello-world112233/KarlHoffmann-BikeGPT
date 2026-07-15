# Event Index Workflow

比赛视频是全程视频，不应该整段直接跑昂贵的姿态模型。正确做法是先建立事件索引，再切成短片段。

## 1. 生成事件索引模板

```bash
python scripts/create_event_index.py \
  --inventory-csv /root/autodl-tmp/bike-ai-data/registry/video_inventory/video_inventory.csv \
  --out-csv /root/autodl-tmp/bike-ai-data/registry/event_index/competition_events.csv
```

然后打开：

```text
/root/autodl-tmp/bike-ai-data/registry/event_index/competition_events.csv
```

每个全程视频先会有一行默认 `start` 片段。你需要看视频，把时间改成真实事件。

## 2. 事件类型

可用 `event_type`：

- `start`
- `first_lap_acceleration`
- `straight`
- `curve`
- `sprint`
- `occlusion`
- `multi_rider`
- `far_view`
- `hard_case`
- `other`

## 3. 时间格式

建议写成：

```text
00:00:12
00:01:35.5
```

每个片段建议 8-20 秒。第一轮不要切太多，每个全程视频先切 2-4 段。

## 4. 切片

先快速无损切片：

```bash
python scripts/cut_clips.py \
  --event-csv /root/autodl-tmp/bike-ai-data/registry/event_index/competition_events.csv \
  --out-root /root/autodl-tmp/bike-ai-data/clips/competition_video \
  --overwrite
```

如果发现切片开头/结尾不准，用重新编码：

```bash
python scripts/cut_clips.py \
  --event-csv /root/autodl-tmp/bike-ai-data/registry/event_index/competition_events.csv \
  --out-root /root/autodl-tmp/bike-ai-data/clips/competition_video \
  --reencode \
  --overwrite
```

## 5. 片段输出

输出结构：

```text
/root/autodl-tmp/bike-ai-data/clips/competition_video/
├── start/
├── sprint/
├── curve/
└── hard_case/
```

这些短片段才是第一轮 baseline 的输入。

