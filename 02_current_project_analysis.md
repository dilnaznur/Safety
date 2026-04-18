# SafetyVision AI: Current Project Analysis

Date: 2026-04-18

This report answers the requested six questions based on the current repository state.

---

## 1) Current Folder/File Structure (Tree View)

```
guard/
├── .env
├── .env.example
├── .gitignore
├── app.py
├── app_integration.py
├── ARCHITECTURE.md
├── config.py
├── demo_mode.py
├── DEPLOYMENT_CHECKLIST.md
├── docker-compose.prod.yml
├── Dockerfile
├── Dockerfile.prod
├── generate_precomputed_demo.py
├── performance.py
├── Procfile
├── QUICK_START.md
├── README.md
├── README_NEW.md
├── RELEASE_NOTES.md
├── requirements.txt
├── surveillance_connector.py
├── models/
│   ├── fall_best.pt
│   ├── fire_best.pt
│   ├── people_best.pt
│   ├── ppe_best.pt
│   └── spill_best.pt
├── static/
│   ├── index.html
│   ├── js/
│   │   └── i18n.js
│   └── vendor/
│       ├── chart.umd.min.js
│       ├── tailwind.min.js
│       └── fontawesome/
│           ├── css/
│           │   └── all.min.css
│           └── webfonts/
│               ├── fa-brands-400.ttf
│               ├── fa-brands-400.woff2
│               ├── fa-regular-400.ttf
│               ├── fa-regular-400.woff2
│               ├── fa-solid-900.ttf
│               └── fa-solid-900.woff2
└── uploads/
```

---

## 2) Does the backend correctly load `.pt` model files?

**Yes**, model loading exists and is correct in structure.

Model loading code (from `app.py`, `ModelManager`):

```python
self.model_paths = {
    "people": "models/people_best.pt",
    "ppe":    "models/ppe_best.pt",
    "fire":   "models/fire_best.pt",
    "spill":  "models/spill_best.pt",
    "fall":   "models/fall_best.pt",
}

for name, path in self.model_paths.items():
    if os.path.exists(path):
        self.models[name] = YOLO(path)
```

And those files are present under `models/`.

---

## 3) Is there a working `/detect` endpoint for image/video upload and results?

**Partially.**

What exists:

- `POST /api/detect-image`
  - Accepts uploaded image
  - Runs detection
  - Returns JSON (`detections`, `alerts`, `stats`, dimensions)

- `POST /api/upload-video`
  - Accepts uploaded video file
  - Saves upload and sets it as stream source
  - Returns `{ success, source }`

Gap vs your requirement:

- There is **no single `/detect` endpoint** that handles both image and video and returns annotated media + full JSON results for both.
- Video upload endpoint currently changes source for live processing, but does not return processed/annotated output video artifact.

---

## 4) Does frontend have file upload feature?

**Yes.**

In `static/index.html`:

- File input for video upload (`id="videoUpload"`)
- File input for image upload (`id="imageUpload"`)

Frontend JS behavior:

- Video upload triggers `POST /api/upload-video`
- Image upload triggers `POST /api/detect-image?mode=...`

So both upload buttons are present and wired.

---

## 5) Any broken imports, missing dependencies, or obvious errors?

Findings:

- `get_errors` reports unresolved imports in `app.py` for packages like `fastapi`, `ultralytics`, `cv2`, `numpy`, `httpx`, `dotenv`, `uvicorn`.
- However, these packages are present in `requirements.txt`.

Interpretation:

- Most likely editor/interpreter environment mismatch (Python env not selected), not necessarily broken source code.

Other observations:

- `requirements.txt` includes optional `onvif-zeep` comment-marked as optional, which is fine.
- No syntax/compile issues were reported in:
  - `config.py`
  - `surveillance_connector.py`
  - `demo_mode.py`
  - `performance.py`
  - `app_integration.py`

---

## 6) What is the entry point to run the project?

Primary entrypoint:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Evidence:

- `Procfile`: `web: uvicorn app:app --host 0.0.0.0 --port $PORT`
- `app.py` has `if __name__ == "__main__": uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)`
- `README.md` also documents uvicorn startup.

Container entrypoint:

- `Dockerfile` runs uvicorn on port 7860.

---

## Bottom Line

Your current project is already a strong base for both requested products, but it is currently a **single integrated app** rather than the desired split architecture.

To satisfy your exact requirements, the highest-impact missing piece is a dedicated demo pipeline that returns annotated output files for both image and video in a deployable Hugging Face app folder.
