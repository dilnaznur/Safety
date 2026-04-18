import asyncio
import json
import logging
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import yaml

from detector import CameraManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SafetyVisionProduct")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


def load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        raise RuntimeError("config.yaml not found in product folder")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


config = load_config()
detection_cfg = config.get("detection", {})

camera_manager = CameraManager(
    cameras=config.get("cameras", []),
    fps_per_camera=float(detection_cfg.get("fps_per_camera", 1)),
    mode=str(detection_cfg.get("mode", "all")),
)

app = FastAPI(title="SafetyVision Product", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.on_event("startup")
async def startup_event():
    camera_manager.start_all()
    logger.info("Camera workers started")


@app.on_event("shutdown")
async def shutdown_event():
    camera_manager.stop_all()
    logger.info("Camera workers stopped")


@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "cameras": camera_manager.camera_list(),
    }


@app.get("/api/config")
async def get_config():
    safe_config = dict(config)
    cameras = []
    for cam in safe_config.get("cameras", []):
        c = dict(cam)
        if "password" in c:
            c["password"] = "***"
        cameras.append(c)
    safe_config["cameras"] = cameras
    return safe_config


@app.get("/api/cameras")
async def get_cameras():
    return {"cameras": camera_manager.camera_list()}


@app.get("/api/stats/{camera_id}")
async def get_camera_stats(camera_id: str):
    payload = camera_manager.get_latest(camera_id)
    if payload is None:
        raise HTTPException(404, "Camera not found")
    return {
        "camera_id": camera_id,
        "connected": payload.get("connected", False),
        "stats": payload.get("stats", {}),
        "alerts": payload.get("alerts", []),
        "timestamp": payload.get("timestamp"),
        "error": payload.get("error"),
    }


@app.post("/api/cameras/{camera_id}/mode/{mode}")
async def set_camera_mode(camera_id: str, mode: str):
    valid_modes = {"all", "people", "ppe", "fire", "spill", "fall"}
    if mode not in valid_modes:
        raise HTTPException(400, f"Unsupported mode: {mode}")
    ok = camera_manager.set_mode(camera_id, mode)
    if not ok:
        raise HTTPException(404, "Camera not found")
    return {"success": True, "camera_id": camera_id, "mode": mode}


@app.websocket("/ws")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()

    cameras = camera_manager.camera_list()
    if not cameras:
        await websocket.send_json({"type": "error", "message": "No enabled cameras in config.yaml"})
        await websocket.close()
        return

    current_camera = cameras[0]["id"]

    try:
        while True:
            try:
                incoming = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                msg = json.loads(incoming)
                requested = msg.get("camera_id")
                if requested and camera_manager.get_latest(requested):
                    current_camera = requested
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

            payload = camera_manager.get_latest(current_camera)
            if payload is None:
                await websocket.send_json({"type": "error", "message": "Camera not found"})
                continue

            await websocket.send_json({"type": "frame", **payload})

    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    host = config.get("server", {}).get("host", "0.0.0.0")
    port = int(config.get("server", {}).get("port", 5000))
    uvicorn.run("server:app", host=host, port=port, reload=False)
