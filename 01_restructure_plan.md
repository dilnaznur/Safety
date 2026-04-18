# SafetyVision AI: Restructure Feasibility and Migration Plan

Date: 2026-04-18

## Executive Decision

Yes, your current codebase can be restructured into the requested architecture.

Recommended path: **refactor in place with selective rewrites** (not full rewrite).

Why:

- Core backend already has multi-model loading, image detection, websocket streaming, and upload UI.
- Existing modules (`config.py`, `demo_mode.py`, `surveillance_connector.py`, `performance.py`) already map well to a split demo/product architecture.
- A full rewrite would duplicate working functionality and increase delivery risk before deployment.

---

## 1) Can Existing Code Fit the Target Architecture?

Target architecture:

```
safetyvision/
├── demo/
├── product/
└── shared/
```

Feasibility: **High**

Current-to-target mapping:

- `app.py` -> split into:
  - `demo/app.py` (upload + offline file processing for Hugging Face)
  - `product/server.py` (RTSP/local monitoring + websocket dashboard)
- `DetectionEngine` and `ModelManager` in `app.py` -> `shared/detection_logic.py`
- `config.py` -> reuse in `product/` (extend to support YAML)
- `surveillance_connector.py` -> move/reuse in `product/`
- `demo_mode.py` -> mostly for product demo/testing; optional in final split
- `static/index.html` + static JS -> move to `product/frontend/`
- `models/*.pt` -> copied/symlinked into `demo/models/` and `product/models/`

---

## 2) What to Keep vs Rewrite

## Keep (minimal changes)

- Model loading logic (YOLO `.pt` loading) from `ModelManager`
- Detection logic in `DetectionEngine.process_frame`
- Existing class/label conventions (PPE classes, fall/fire classes, spill severity)
- WebSocket manager and live stream flow for local product
- Existing frontend dashboard structure (can be restyled later)
- Supporting modules:
  - `config.py`
  - `surveillance_connector.py`
  - `performance.py`

## Rewrite or significant refactor

- API surface split by product boundaries:
  - Demo: must return annotated image/video artifacts + JSON from file upload
  - Product: must focus on RTSP 24/7 monitoring and live alert stream
- Add **video file processing endpoint** for demo output (currently upload endpoint sets stream source but does not return processed video artifact)
- Add YAML-driven camera config (`product/config.yaml`) for client RTSP setup
- Add Windows one-command installer (`product/install.bat`)
- Decompose current monolithic `app.py` into shared + product/demo entrypoints

## Optional later improvements

- Add tracking IDs to PPE compliance calculations for per-person compliance over time
- Add persistence layer for alerts/events (SQLite) for audit trail
- Add worker queue for video batch processing in demo mode

---

## 3) Faster: Start Fresh or Refactor?

**Faster choice: Refactor (hybrid) with selective new files.**

Estimated effort:

- Refactor path: 1.5 to 3 days for a working two-product split
- Full rewrite: 4 to 8+ days to reach same stability/features

Risk profile:

- Refactor: lower risk, reuses tested flows
- Full rewrite: higher risk of regressions in detection, websocket, UI behavior

---

## Concrete Action Plan (Files to Create/Modify)

## Phase 1: Scaffold New Structure

Create:

- `demo/app.py`
- `demo/requirements.txt`
- `demo/models/` (copy `.pt` models)
- `product/server.py`
- `product/detector.py`
- `product/config.yaml`
- `product/frontend/index.html`
- `product/frontend/app.js`
- `product/models/` (copy `.pt` models)
- `product/install.bat`
- `shared/detection_logic.py`

Modify:

- `README.md` (new run instructions for both products)
- `requirements.txt` (keep root compatibility during migration)

## Phase 2: Extract Shared Detection Layer

Create `shared/detection_logic.py` with:

- `ModelManager` (lazy model loading)
- `DetectionEngine`
- Shared DTO/helpers for normalization and alert formatting

Modify:

- `app.py` to import from shared (temporary compatibility)

## Phase 3: Build Product 1 (Demo / Hugging Face)

Create `demo/app.py` (Gradio preferred for HF simplicity):

- Upload image/video
- Run shared detection
- Return:
  - Annotated image/video file
  - JSON detection output
  - KPI summary (people count, PPE compliance, alerts)

Create `demo/requirements.txt`:

- `gradio`
- `ultralytics`
- `opencv-python-headless`
- `numpy`

## Phase 4: Build Product 2 (Client Local Monitoring)

Create `product/server.py` (FastAPI + WebSocket):

- Read `product/config.yaml`
- Connect multiple RTSP streams
- Process at 1 FPS (configurable)
- Emit live updates over websocket

Create `product/detector.py`:

- Per-camera processing loop
- Alert conditions: no helmet, fire, spills, falls

Create `product/frontend/*`:

- Dashboard with live feed panels
- Alert stream panel
- Camera status indicators

Create `product/install.bat`:

- Create venv
- Install requirements
- Start server

## Phase 5: Migration and Compatibility

Modify:

- Keep `app.py` as compatibility gateway (optional), or deprecate after validation
- Update docs (`QUICK_START.md`, `ARCHITECTURE.md`) to reflect split layout

Validation checklist:

- Demo image upload returns annotated image + JSON
- Demo video upload returns annotated video + JSON
- Product loads cameras from YAML and raises live alerts at 1 FPS
- Product runs offline on localhost

---

## Suggested Migration Order

1. Extract shared detection first
2. Implement demo app and validate HF deployment
3. Implement product server with YAML camera config
4. Wire product frontend dashboard
5. Add installer and docs

This order minimizes risk and gives you a visible demo early for judges while preserving progress toward client delivery.
