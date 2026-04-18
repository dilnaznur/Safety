# 🚀 Quick Deploy: Render + Vercel

## What goes where

- **Render**: Python backend, YOLO models, `/docs`, `/health`, `/api/*`
- **Vercel**: static frontend only, serves `static/index.html`

This repo is too large for a Python backend on Vercel because of the `.pt` model files.

## 1. Deploy backend to Render

1. Push the repo to GitHub.
2. Create a **Render Web Service** from the repo.
3. Use:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port 8000`
4. Add environment variables:
   - `BOT_TOKEN`
   - `CHAT_ID`

Test backend:

```text
https://safetyvision-guard.onrender.com/health
https://safetyvision-guard.onrender.com/docs
```

## 2. Deploy frontend to Vercel

1. Create a Vercel project from the same GitHub repo.
2. Do **not** configure a Python build.
3. Vercel will serve the static UI from `static/index.html`.

The frontend is already configured to call the Render backend at:

```text
https://safetyvision-guard.onrender.com
```

If your Render URL is different, set `window.__API_BASE__` or edit the default in `static/index.html`.

## 3. Verify

- Vercel URL: open `/`
- Render URL: open `/docs`

If Vercel shows the dashboard and Render shows Swagger UI, the split deployment is correct.
