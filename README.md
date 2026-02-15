---
title: SafetyVision AI
emoji: 🏭
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
---

# SafetyVision AI – Industrial Safety Monitoring

Real-time industrial safety monitoring powered by 5 YOLOv8 models.

## Models

| Model  | Purpose                  | mAP50 |
| ------ | ------------------------ | ----- |
| People | Headcount & tracking     | 0.944 |
| PPE    | Equipment compliance     | 0.977 |
| Fire   | Fire & smoke detection   | 0.902 |
| Spill  | Liquid spill detection   | 0.987 |
| Fall   | Fall & posture detection | 0.977 |

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser.
