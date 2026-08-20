# ROTA — Monocular Cycling Intelligence

SaaS-style workspace for **single-camera** bike video analysis:

```text
video → 2D keypoints → monocular 3D skeleton → cycling metrics → report
```

## Run

```bash
cd /root/autodl-tmp/bike-ai-platform/apps/rota
pip install fastapi uvicorn python-multipart aiofiles
python app.py
```

Open: `http://127.0.0.1:8787` (or AutoDL port mapping to **8787**)

## What leaders see

- Landing brand page
- Workspace with pipeline, orbitable 3D rider + bike guide
- Cadence / knee ROM / symmetry / torso lean / ankle circularity / form index
- Printable analysis report (`Export report`)

## Demo data

Bundled from T014 monocular pilot (`data/demo/joints.json`).

Upload runs the full monocular pipeline on the server (Sapiens2 → athlete lock → MotionBERT → cycling constraints). Expect several minutes on first GPU run; default cap is 120 frames @ 10 fps. Tune with env `ROTA_MAX_FRAMES`, `ROTA_PIPELINE_FPS`.
