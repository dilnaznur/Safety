"""
SafetyVision AI - Industrial Safety Monitoring Backend
FastAPI + WebSocket + YOLOv8 real-time inference
Multi-mode: Demo (competitions), Production (surveillance), Development
Supports RTSP, ONVIF, GPU acceleration, and adaptive performance optimization
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging
import os
import time
import httpx
from dotenv import load_dotenv

# Import new architecture modules
from config import Config, DeploymentMode
from surveillance_connector import RTSPConnector, ONVIFConnector, CameraPool
from demo_mode import DemoManager, DemoPrecomputeGenerator
from performance import AdaptiveProcessor, PerformanceMetrics

load_dotenv()

# ==================== CONFIGURATION & INITIALIZATION ====================

# Load global configuration
config = Config()

logger = logging.getLogger("SafetyVision")
logging.basicConfig(level=logging.INFO)

# Telegram setup from config
BOT_TOKEN = config.alerts.bot_token or ""
CHAT_ID = config.alerts.chat_id or ""
TELEGRAM_ENABLED = config.alerts.telegram_enabled

# Thread pool for CPU-bound YOLO inference
_inference_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")

# ==================== APP INITIALIZATION ====================

app = FastAPI(title="SafetyVision AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SafetyVision")

# ==================== PERFORMANCE MONITORING & DEMO SETUP ====================

# Initialize performance processor
performance_processor = AdaptiveProcessor(
    target_fps=config.performance.target_fps,
    min_fps=1,
    max_fps=30,
)
logger.info(f"Performance monitoring enabled (target: {config.performance.target_fps} FPS)")

# Initialize demo manager if in demo mode
demo_manager: Optional[DemoManager] = None
if config.mode == DeploymentMode.DEMO:
    demo_manager = DemoManager(
        sample_dir=config.demo.sample_dir,
        precomputed_dir=config.demo.precomputed_dir if config.demo.use_precomputed else None,
    )
    samples = demo_manager.list_samples()
    logger.info(f"Demo mode: Found {len(samples)} samples: {samples}")

# Initialize camera pool for multi-camera support
camera_pool: Optional[CameraPool] = None
if config.mode == DeploymentMode.PRODUCTION:
    camera_pool = CameraPool(max_cameras=8)
    
    # Add RTSP camera if configured
    if config.surveillance.rtsp_url:
        camera_pool.add_camera(
            "primary",
            rtsp_url=config.surveillance.rtsp_url,
            username=config.surveillance.rtsp_username,
            password=config.surveillance.rtsp_password,
            timeout=config.surveillance.connection_timeout,
        )
    
    # Add ONVIF camera if configured
    if config.surveillance.onvif_enabled and config.surveillance.onvif_host:
        camera_pool.add_camera(
            "onvif_primary",
            onvif_host=config.surveillance.onvif_host,
            onvif_port=config.surveillance.onvif_port,
            onvif_username=config.surveillance.onvif_username,
            onvif_password=config.surveillance.onvif_password,
        )

# ==================== TELEGRAM ALERT SERVICE ====================

class TelegramAlertService:
    """Async Telegram bot alerts with per-class rate limiting."""

    def __init__(self, bot_token: str, chat_id: str, cooldown: float = 10.0):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown = cooldown
        self._last_sent: Dict[str, float] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self.enabled = bool(bot_token and chat_id)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    def _can_send(self, alert_class: str) -> bool:
        now = time.time()
        last = self._last_sent.get(alert_class, 0)
        if now - last >= self.cooldown:
            self._last_sent[alert_class] = now
            return True
        return False

    async def send_alert(self, alert: dict, snapshot_bytes: Optional[bytes] = None):
        """Send alert to Telegram (non-blocking background task)."""
        if not self.enabled:
            return
        alert_class = alert.get("type", "UNKNOWN")
        if not self._can_send(alert_class):
            return
        try:
            client = await self._get_client()
            severity = alert.get("severity", "info").upper()
            message = alert.get("message", "")
            conf = alert.get("confidence", "")
            timestamp = alert.get("timestamp", datetime.now().isoformat())

            emoji_map = {"CRITICAL": "\U0001f6a8", "HIGH": "\u26a0\ufe0f",
                         "WARNING": "\u26a1", "INFO": "\u2139\ufe0f"}
            emoji = emoji_map.get(severity, "\u2139\ufe0f")

            text = (
                f"{emoji} *SafetyVision AI Alert*\n\n"
                f"Type: `{alert_class}`\n"
                f"Severity: `{severity}`\n"
                f"Message: {message}\n"
            )
            if conf:
                text += f"Confidence: `{conf}`\n"
            text += f"Time: `{timestamp}`"

            base_url = f"https://api.telegram.org/bot{self.bot_token}"

            if snapshot_bytes:
                await client.post(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": self.chat_id, "caption": text, "parse_mode": "Markdown"},
                    files={"photo": ("alert.jpg", snapshot_bytes, "image/jpeg")},
                )
            else:
                await client.post(
                    f"{base_url}/sendMessage",
                    json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
                )
        except Exception as exc:
            logger.warning(f"Telegram send failed: {exc}")

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


telegram = TelegramAlertService(BOT_TOKEN, CHAT_ID, cooldown=10.0)


@app.on_event("shutdown")
async def shutdown_event():
    await telegram.close()


# ==================== MODEL MANAGER (Singleton) ====================

class ModelManager:
    """Loads and manages all YOLO models once."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.models: Dict[str, YOLO] = {}
        self.model_paths = {
            "people": "models/people_best.pt",
            "ppe":    "models/ppe_best.pt",
            "fire":   "models/fire_best.pt",
            "spill":  "models/spill_best.pt",
            "fall":   "models/fall_best.pt",
        }
        self._load_models()

    def _load_models(self):
        for name, path in self.model_paths.items():
            try:
                if os.path.exists(path):
                    self.models[name] = YOLO(path)
                    logger.info(f"[OK] {name} model loaded from {path}")
                else:
                    logger.error(f"[MISS] Model file not found: {path}")
            except Exception as exc:
                logger.error(f"[FAIL] Could not load {name}: {exc}")

    def get(self, name: str) -> Optional[YOLO]:
        return self.models.get(name)


model_manager = ModelManager()

# ==================== DETECTION ENGINE ====================

class DetectionEngine:
    """Runs inference across models, tracks people, and generates alerts."""

    PPE_CLASSES = ["Boots", "Ear-protection", "Glass", "Glove",
                   "Helmet", "Mask", "Person", "Vest"]
    FIRE_CLASSES = ["fire", "smoke", "other"]
    FALL_CLASSES = ["Falling", "Sitting", "Standing"]

    SPILL_SEVERITY = {
        0: ("Minor – Water / Safe liquid", "warning"),
        1: ("Minor – Safe liquid", "warning"),
        2: ("Moderate – Oil / Coolant", "high"),
        3: ("Moderate – Coolant", "high"),
        4: ("Critical – Chemical", "critical"),
        5: ("Critical – Hazardous", "critical"),
    }

    COLORS = {
        "person":          (0, 255, 0),
        "Helmet":          (0, 200, 0),
        "Vest":            (0, 200, 0),
        "Boots":           (0, 200, 0),
        "Glove":           (0, 200, 0),
        "Glass":           (0, 200, 0),
        "Mask":            (0, 200, 0),
        "Ear-protection":  (0, 200, 0),
        "Person":          (255, 180, 0),
        "fire":            (0, 0, 255),
        "smoke":           (0, 165, 255),
        "Falling":         (0, 0, 255),
        "Standing":        (0, 255, 0),
        "Sitting":         (0, 255, 255),
    }

    def __init__(self):
        self.stats = {
            "people_count": 0,
            "total_people_today": 0,
            "max_people_count": 0,
            "people_entered": 0,
            "people_exited": 0,
            "ppe_compliance": 0.0,
            "fire_risk": "Safe",
            "active_alerts": 0,
            "spill_count": 0,
            "fall_count": 0,
        }

        self._tracker: Dict[int, np.ndarray] = {}
        self._next_id = 0
        self._history: Dict[int, list] = defaultdict(list)
        self._track_dist = 80

        self._entrance_frac = 0.25
        self._exit_frac = 0.75

        self._last_alert_ts: Dict[str, float] = {}
        self._alert_cooldown = 5.0

    def _should_alert(self, key: str) -> bool:
        now = time.time()
        if now - self._last_alert_ts.get(key, 0) > self._alert_cooldown:
            self._last_alert_ts[key] = now
            return True
        return False

    @staticmethod
    def _iou_overlap(person_box, item_boxes, threshold=0.05):
        px1, py1, px2, py2 = person_box
        p_area = max((px2 - px1) * (py2 - py1), 1)
        for ib in item_boxes:
            ix1, iy1, ix2, iy2 = ib
            xx1, yy1 = max(px1, ix1), max(py1, iy1)
            xx2, yy2 = min(px2, ix2), min(py2, iy2)
            if xx2 > xx1 and yy2 > yy1:
                if (xx2 - xx1) * (yy2 - yy1) / p_area > threshold:
                    return True
        return False

    # ── per-model detection ──────────────────────────────

    def detect_people(self, frame: np.ndarray):
        model = model_manager.get("people")
        if model is None:
            return [], []
        try:
            results = model(frame, conf=0.3, imgsz=640, verbose=False)
        except Exception as exc:
            logger.warning(f"People inference error: {exc}")
            return [], []

        raw = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                raw.append({
                    "class": "person", "confidence": float(box.conf[0]),
                    "bbox": xyxy.tolist(), "center": (float(cx), float(cy)), "id": None,
                })

        tracked = self._track(raw, frame.shape[1])
        count = len(tracked)
        self.stats["people_count"] = count
        if count > self.stats["max_people_count"]:
            self.stats["max_people_count"] = count

        alerts = []
        if count > 50 and self._should_alert("occupancy"):
            alerts.append(self._mkalert("OCCUPANCY_EXCEEDED", "warning",
                                        f"High occupancy: {count} people detected"))
        return tracked, alerts

    def _track(self, detections, frame_w):
        used = set()
        for det in detections:
            c = np.array(det["center"])
            best_id, best_d = None, float("inf")
            for pid, pos in self._tracker.items():
                if pid in used:
                    continue
                d = np.linalg.norm(c - pos)
                if d < self._track_dist and d < best_d:
                    best_id, best_d = pid, d
            if best_id is not None:
                det["id"] = best_id
                self._tracker[best_id] = c
                used.add(best_id)
            else:
                det["id"] = self._next_id
                self._tracker[self._next_id] = c
                self._next_id += 1
                self.stats["total_people_today"] += 1
            self._history[det["id"]].append(c)
            if len(self._history[det["id"]]) > 10:
                self._history[det["id"]].pop(0)
            ev = self._zone_check(det["id"], frame_w)
            if ev == "entered":
                self.stats["people_entered"] += 1
            elif ev == "exited":
                self.stats["people_exited"] += 1

        cur_ids = {d["id"] for d in detections}
        for g in set(self._tracker) - cur_ids:
            del self._tracker[g]
            self._history.pop(g, None)
        return detections

    def _zone_check(self, pid, fw):
        h = self._history.get(pid, [])
        if len(h) < 2:
            return None
        prev_x, cur_x = h[-2][0], h[-1][0]
        if prev_x < fw * self._entrance_frac <= cur_x:
            return "entered"
        if prev_x > fw * self._exit_frac >= cur_x:
            return "exited"
        return None

    def detect_ppe(self, frame: np.ndarray):
        model = model_manager.get("ppe")
        if model is None:
            return [], []
        try:
            results = model(frame, conf=0.35, imgsz=640, verbose=False)
        except Exception as exc:
            logger.warning(f"PPE inference error: {exc}")
            return [], []

        detections = []
        people_boxes, item_map = [], defaultdict(list)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = self.PPE_CLASSES[cls] if cls < len(self.PPE_CLASSES) else f"cls_{cls}"
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                detections.append({"class": name, "confidence": float(box.conf[0]), "bbox": xyxy})
                if name == "Person":
                    people_boxes.append(xyxy)
                else:
                    item_map[name].append(xyxy)

        alerts = []
        total, compliant = len(people_boxes), 0
        for pb in people_boxes:
            has_h = self._iou_overlap(pb, item_map.get("Helmet", []))
            has_v = self._iou_overlap(pb, item_map.get("Vest", []))
            if has_h and has_v:
                compliant += 1
            else:
                missing = []
                if not has_h: missing.append("Helmet")
                if not has_v: missing.append("Vest")
                key = f"ppe_{'_'.join(missing)}"
                if self._should_alert(key):
                    alerts.append(self._mkalert(
                        "PPE_VIOLATION",
                        "critical" if "Helmet" in missing else "warning",
                        f"Worker missing: {', '.join(missing)}", pb,
                    ))
        self.stats["ppe_compliance"] = round(compliant / total * 100, 1) if total else 0.0
        return detections, alerts

    def detect_fire(self, frame: np.ndarray):
        model = model_manager.get("fire")
        if model is None:
            return [], []
        try:
            results = model(frame, conf=0.3, imgsz=640, verbose=False)
        except Exception as exc:
            logger.warning(f"Fire inference error: {exc}")
            return [], []

        detections, alerts = [], []
        fire_seen = smoke_seen = False
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = self.FIRE_CLASSES[cls] if cls < len(self.FIRE_CLASSES) else "other"
                if name == "other":
                    continue
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                detections.append({"class": name, "confidence": conf, "bbox": xyxy})
                if name == "fire":
                    fire_seen = True
                    if self._should_alert("fire"):
                        alerts.append(self._mkalert("FIRE_DETECTED", "critical",
                                                    "FIRE DETECTED! Immediate action required!", xyxy, conf))
                elif name == "smoke":
                    smoke_seen = True
                    if self._should_alert("smoke"):
                        alerts.append(self._mkalert("SMOKE_DETECTED", "high",
                                                    "Smoke detected – investigate immediately.", xyxy, conf))
        self.stats["fire_risk"] = "CRITICAL" if fire_seen else ("High" if smoke_seen else "Safe")
        return detections, alerts

    def detect_spills(self, frame: np.ndarray):
        model = model_manager.get("spill")
        if model is None:
            return [], []
        try:
            results = model(frame, conf=0.4, imgsz=640, verbose=False)
        except Exception as exc:
            logger.warning(f"Spill inference error: {exc}")
            return [], []

        detections, alerts = [], []
        count = 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                sev_label, sev = self.SPILL_SEVERITY.get(cls, ("Unknown", "warning"))
                detections.append({"class": f"Spill ({sev_label})", "confidence": conf,
                                   "bbox": xyxy, "severity": sev_label})
                count += 1
                if self._should_alert(f"spill_{cls}"):
                    alerts.append(self._mkalert("SPILL_DETECTED", sev,
                                                f"{sev_label} spill – cleaning required.", xyxy, conf))
        self.stats["spill_count"] = count
        return detections, alerts

    def detect_falls(self, frame: np.ndarray):
        model = model_manager.get("fall")
        if model is None:
            return [], []
        try:
            results = model(frame, conf=0.35, imgsz=640, verbose=False)
        except Exception as exc:
            logger.warning(f"Fall inference error: {exc}")
            return [], []

        detections, alerts = [], []
        falls = 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = self.FALL_CLASSES[cls] if cls < len(self.FALL_CLASSES) else f"cls_{cls}"
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                detections.append({"class": name, "confidence": conf, "bbox": xyxy})
                if name == "Falling":
                    falls += 1
                    if self._should_alert("fall"):
                        alerts.append(self._mkalert("FALL_DETECTED", "critical",
                                                    "FALL DETECTED! Emergency response required!", xyxy, conf))
        self.stats["fall_count"] = falls
        return detections, alerts

    # ── composite processing ─────────────────────────────

    MODE_MAP = {
        "all": ["people", "ppe", "fire", "spill", "fall"],
        "people": ["people"], "ppe": ["ppe"], "fire": ["fire"],
        "spill": ["spill"], "fall": ["fall"],
    }
    DETECT_FN = {
        "people": "detect_people", "ppe": "detect_ppe", "fire": "detect_fire",
        "spill": "detect_spills", "fall": "detect_falls",
    }

    def process_frame(self, frame: np.ndarray, mode: str = "all"):
        dets, alerts = [], []
        for m in self.MODE_MAP.get(mode, self.MODE_MAP["all"]):
            try:
                d, a = getattr(self, self.DETECT_FN[m])(frame)
                dets.extend(d)
                alerts.extend(a)
            except Exception as exc:
                logger.warning(f"Model '{m}' error: {exc}")
        self.stats["active_alerts"] = len(alerts)
        return dets, alerts

    def _mkalert(self, atype, sev, msg, bbox=None, confidence=None):
        a = {"type": atype, "severity": sev, "message": msg,
             "timestamp": datetime.now().isoformat()}
        if bbox:
            a["bbox"] = bbox
        if confidence is not None:
            a["confidence"] = round(confidence, 3)
        return a


engine = DetectionEngine()

# ==================== WEBSOCKET MANAGER ====================

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"Client connected ({len(self.active)} total)")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logger.info(f"Client disconnected ({len(self.active)} total)")

    async def send(self, data: dict, ws: WebSocket):
        try:
            await ws.send_json(data)
        except Exception:
            pass


ws_manager = ConnectionManager()

# ==================== VIDEO SOURCE MANAGER ====================

class VideoSource:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.source = None
        self._upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(self._upload_dir, exist_ok=True)

    def open(self, source=0):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            if isinstance(source, int) and os.name == 'nt':
                self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                self.cap = None
                return False
            if source in (0, 1):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                self.cap.set(cv2.CAP_PROP_FOCUS, 0)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.source = source
            return True
        except Exception as exc:
            logger.error(f"Video source error: {exc}")
            self.cap = None
            return False

    def read(self):
        if self.cap is None:
            return False, None
        try:
            ret, frame = self.cap.read()
            if not ret and isinstance(self.source, str):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            return ret, frame
        except Exception:
            return False, None

    def release(self):
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except Exception:
            pass

    def save_upload(self, data: bytes, filename: str) -> str:
        path = os.path.join(self._upload_dir, filename)
        with open(path, "wb") as f:
            f.write(data)
        return path


video_source = VideoSource()


def _run_inference(infer_frame, mode_str, display_shape=None):
    dets, alerts = engine.process_frame(infer_frame, mode_str)
    if display_shape is not None:
        ih, iw = infer_frame.shape[:2]
        dh, dw = display_shape[:2]
        if iw != dw or ih != dh:
            sx, sy = dw / iw, dh / ih
            for det in dets:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    return dets, alerts


async def _send_telegram_alerts(alerts: list, frame=None):
    if not telegram.enabled or not alerts:
        return
    for alert in alerts:
        if alert.get("severity") not in ("critical", "high"):
            continue
        snapshot = None
        if frame is not None:
            try:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                snapshot = buf.tobytes()
            except Exception:
                pass
        try:
            await telegram.send_alert(alert, snapshot)
        except Exception as exc:
            logger.warning(f"Telegram background error: {exc}")


# ==================== API ROUTES ====================

@app.get("/", response_class=HTMLResponse)
async def index():
    p = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(p) if os.path.exists(p) else HTMLResponse("<h1>SafetyVision AI</h1>")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "mode": config.mode.value,
        "models": {k: ("loaded" if k in model_manager.models else "missing")
                   for k in model_manager.model_paths},
        "telegram": telegram.enabled,
        "version": "2.1.0",
    }


# ==================== CONFIGURATION & SYSTEM ENDPOINTS ====================

@app.get("/api/config")
async def get_config():
    """Get current system configuration"""
    return config.to_dict()


@app.get("/api/system/stats")
async def get_system_stats():
    """Get system health and performance statistics"""
    return {
        "mode": config.mode.value,
        "performance": performance_processor.get_metrics().to_dict(),
        "profiler": performance_processor.profiler.get_stats(),
        "bottleneck": performance_processor.profiler.find_bottleneck(),
    }


# ==================== DEMO MODE ENDPOINTS ====================

@app.get("/api/demo/samples")
async def demo_list_samples():
    """List available demo samples"""
    if not demo_manager:
        return {"error": "Demo mode not enabled", "samples": []}
    
    samples = demo_manager.list_samples()
    return {
        "mode": config.mode.value,
        "samples": samples,
        "use_precomputed": config.demo.use_precomputed,
    }


@app.post("/api/demo/load")
async def demo_load_sample(sample_name: str = Query(...)):
    """Load a demo sample for playback"""
    if not demo_manager:
        raise HTTPException(400, "Demo mode not enabled")
    
    if not demo_manager.load_sample(sample_name):
        raise HTTPException(404, f"Sample not found: {sample_name}")
    
    return {
        "success": True,
        "sample": sample_name,
        "progress": demo_manager.get_current_progress(),
    }


@app.get("/api/demo/progress")
async def demo_get_progress():
    """Get current demo playback progress"""
    if not demo_manager:
        return {"error": "Demo mode not enabled"}
    
    return demo_manager.get_current_progress()


# ==================== SURVEILLANCE ENDPOINTS ====================

@app.get("/api/cameras/list")
async def cameras_list():
    """List connected surveillance cameras"""
    if not camera_pool:
        return {"error": "Camera pool not initialized", "cameras": []}
    
    cameras = camera_pool.list_cameras()
    return {
        "cameras": cameras,
        "total": len(cameras),
        "max_cameras": camera_pool.max_cameras,
    }


@app.get("/api/cameras/{camera_name}/stats")
async def camera_stats(camera_name: str):
    """Get statistics for a specific camera"""
    if not camera_pool:
        raise HTTPException(400, "Camera pool not initialized")
    
    cam = camera_pool.get_camera(camera_name)
    if not cam:
        raise HTTPException(404, f"Camera not found: {camera_name}")
    
    return {
        "camera": camera_name,
        "stats": cam.get_stats() if hasattr(cam, 'get_stats') else {},
    }


@app.get("/api/stats")
async def get_stats():
    return engine.stats


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    allowed = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    ext = os.path.splitext(file.filename or "v.mp4")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format. Use: {', '.join(allowed)}")
    try:
        data = await file.read()
        path = video_source.save_upload(data, f"upload{ext}")
        return {"success": video_source.open(path), "source": path}
    except Exception as exc:
        logger.error(f"Video upload error: {exc}")
        raise HTTPException(500, "Video upload failed")


@app.post("/api/use-webcam")
async def use_webcam():
    return {"success": video_source.open(0), "source": "webcam"}


@app.post("/api/telegram-test")
async def telegram_test():
    if not telegram.enabled:
        return {"success": False, "error": "Telegram not configured. Set BOT_TOKEN and CHAT_ID."}
    try:
        telegram._last_sent.pop("TEST", None)
        await telegram.send_alert({
            "type": "TEST", "severity": "info",
            "message": "SafetyVision AI — Telegram integration test OK!",
            "timestamp": datetime.now().isoformat(),
        })
        return {"success": True}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ==================== WEBSOCKET ENDPOINT ====================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    if video_source.cap is None:
        video_source.open(0)

    detection_mode = "all"
    running = True
    
    # Use configuration values
    JPEG_QUALITY = config.performance.jpeg_quality
    DISPLAY_MAX_W = config.performance.max_frame_width
    DISPLAY_MAX_H = config.performance.max_frame_height
    INFER_MAX = config.performance.inference_width
    
    # Infer size from config
    infer_size = config.performance.get_infer_size()

    try:
        while running:
            t0_frame = time.monotonic()

            # 1. Drain client messages
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.005)
                msg = json.loads(raw)
                if "mode" in msg:
                    detection_mode = msg["mode"]
                if msg.get("action") == "emergency_stop":
                    running = False
                    break
                if msg.get("source") == "webcam":
                    video_source.open(0)
                if msg.get("demo_sample"):
                    # Load demo sample if in demo mode
                    if demo_manager:
                        demo_manager.load_sample(msg["demo_sample"])
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

            # 2. Check if should skip frame (FPS control)
            if performance_processor.should_skip_frame():
                await asyncio.sleep(0.005)
                continue

            # 3. Read frame (from video source or demo)
            t0_read = time.monotonic()
            ret, frame, precomputed_result = False, None, None
            
            if config.mode == DeploymentMode.DEMO and demo_manager:
                # Demo mode: read from demo manager
                ret, frame, precomputed_result = demo_manager.read_frame()
            else:
                # Normal mode: read from video source
                ret, frame = video_source.read()
            
            if not ret or frame is None:
                await ws_manager.send({
                    "type": "no_source",
                    "stats": dict(engine.stats),
                    "performance": performance_processor.get_metrics().to_dict(),
                    "timestamp": datetime.now().isoformat()
                }, ws)
                await asyncio.sleep(0.5)
                continue

            read_time_ms = (time.monotonic() - t0_read) * 1000

            # 4. Prepare display frame
            h, w = frame.shape[:2]
            display_frame = frame
            if w > DISPLAY_MAX_W or h > DISPLAY_MAX_H:
                s = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h)
                display_frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

            dh, dw = display_frame.shape[:2]
            if dw > infer_size[0] or dh > infer_size[1]:
                s = min(infer_size[0] / dw, infer_size[1] / dh)
                infer_frame = cv2.resize(display_frame, (int(dw * s), int(dh * s)), interpolation=cv2.INTER_AREA)
            else:
                infer_frame = display_frame

            # 5. Inference or use precomputed results (demo mode)
            t0_infer = time.monotonic()
            detections, alerts = [], []
            
            if precomputed_result:
                # Use precomputed results from demo
                detections = precomputed_result.get("detections", [])
                alerts = precomputed_result.get("alerts", [])
            else:
                # Run live inference
                try:
                    detections, alerts = await asyncio.get_event_loop().run_in_executor(
                        _inference_pool, _run_inference, infer_frame, detection_mode, display_frame.shape)
                except Exception as exc:
                    logger.warning(f"Inference error: {exc}")
                    detections, alerts = [], []
            
            infer_time_ms = (time.monotonic() - t0_infer) * 1000

            # 6. Telegram (background) - only for production critical alerts
            if alerts and config.mode == DeploymentMode.PRODUCTION:
                asyncio.create_task(_send_telegram_alerts(alerts, display_frame))

            # 7. Encode JPEG
            t0_encode = time.monotonic()
            try:
                _, buf = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                b64 = base64.b64encode(buf).decode()
            except Exception:
                continue
            
            encode_time_ms = (time.monotonic() - t0_encode) * 1000

            # 8. Record performance metrics
            performance_processor.on_frame_processed(
                inference_time_ms=infer_time_ms,
                read_time_ms=read_time_ms,
                encode_time_ms=encode_time_ms,
            )

            # 9. Send response
            fh, fw = display_frame.shape[:2]
            await ws_manager.send({
                "type": "frame",
                "frame": b64,
                "frameWidth": fw,
                "frameHeight": fh,
                "detections": [{k: v for k, v in d.items() if k != "center"} for d in detections],
                "alerts": alerts,
                "stats": dict(engine.stats),
                "performance": performance_processor.get_metrics().to_dict(),
                "demo_progress": demo_manager.get_current_progress() if demo_manager else None,
                "timestamp": datetime.now().isoformat(),
            }, ws)

            # 10. Maintain FPS with intelligent sleep
            elapsed = time.monotonic() - t0_frame
            performance_processor.fps_controller.sleep_to_target(elapsed)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(f"WS error: {exc}")
    finally:
        ws_manager.disconnect(ws)


# ==================== IMAGE DETECTION ====================

@app.post("/api/detect-image")
async def detect_image(file: UploadFile = File(...), mode: str = "all"):
    try:
        data = await file.read()
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "Could not decode image")

        h, w = frame.shape[:2]
        infer_frame = frame
        if w > 640 or h > 640:
            s = min(640 / w, 640 / h)
            infer_frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

        detections, alerts = engine.process_frame(infer_frame, mode)

        ih, iw = infer_frame.shape[:2]
        if iw != w or ih != h:
            sx, sy = w / iw, h / ih
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

        if alerts:
            asyncio.create_task(_send_telegram_alerts(alerts, frame))

        return {
            "frameWidth": w, "frameHeight": h,
            "detections": [{k: v for k, v in d.items() if k != "center"} for d in detections],
            "alerts": alerts, "stats": dict(engine.stats),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Image detection error: {exc}")
        raise HTTPException(500, "Image analysis failed")


# ==================== ENTRYPOINT ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
