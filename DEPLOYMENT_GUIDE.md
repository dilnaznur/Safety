# Deployment Guide

## Architecture

- **Render** hosts the FastAPI backend and YOLO models.
- **Vercel** hosts the static frontend only.
- The frontend calls the backend through `API_BASE`.

## Render backend

Use the following settings in Render:

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app:app --host 0.0.0.0 --port 8000`
- Environment Variables: `BOT_TOKEN`, `CHAT_ID`

Useful URLs on Render:

- `/health`
- `/docs`
- `/api/config`

## Vercel frontend

Vercel should only serve the UI from `static/index.html` and the local assets under `static/`.

Do not use the Python builder on Vercel. The model files are too large for the Vercel function limit.

## Notes

- If you want to use a different backend URL, set `window.__API_BASE__` before the app script runs.
- The default backend URL in the UI points to `https://safetyvision-guard.onrender.com`.

## Verification

1. Open the Vercel URL and confirm the dashboard loads.
2. Open the Render URL and confirm `/docs` works.
3. Trigger one API call from the frontend and confirm it reaches Render.
