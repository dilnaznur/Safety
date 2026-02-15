"""
SafetyVision AI - Industrial Safety Monitoring Backend
FastAPI + WebSocket + YOLOv8 real-time inference
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import json
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging
import os
import time
import math

# Thread pool for CPU-bound YOLO inference — prevents blocking the async event loop
_inference_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")

# ==================== APP INITIALIZATION ====================

app = FastAPI(title="SafetyVision AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SafetyVision")

# ==================== MODEL MANAGER ====================

class ModelManager:
    """Loads and manages all YOLO models."""

    def __init__(self):
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
                self.models[name] = YOLO(path)
                logger.info(f"[OK] {name} model loaded from {path}")
            except Exception as exc:
                logger.error(f"[FAIL] Could not load {name} model: {exc}")

    def get(self, name: str) -> Optional[YOLO]:
        return self.models.get(name)


model_manager = ModelManager()

# ==================== DETECTION ENGINE ====================

class DetectionEngine:
    """Runs inference across models, tracks people, and generates alerts."""

    # PPE class names in training order
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

        # People tracking state
        self._tracker: Dict[int, np.ndarray] = {}   # id → last centre
        self._next_id = 0
        self._history: Dict[int, list] = defaultdict(list)
        self._track_dist = 80  # max pixel distance for same‑person match

        # Zone definitions (fraction of frame width)
        self._entrance_frac = 0.25   # left 25 %
        self._exit_frac = 0.75       # right 25 %

        # Alert dedup
        self._last_alert_ts: Dict[str, float] = {}
        self._alert_cooldown = 5.0   # seconds
        self._frame_idx = 0          # rotation counter for "all" mode

    # ── helpers ──────────────────────────────────────────────

    def _should_alert(self, key: str) -> bool:
        now = time.time()
        if now - self._last_alert_ts.get(key, 0) > self._alert_cooldown:
            self._last_alert_ts[key] = now
            return True
        return False

    @staticmethod
    def _iou_overlap(person_box, item_boxes, threshold=0.05):
        """Return True if any item_box overlaps person_box enough."""
        px1, py1, px2, py2 = person_box
        p_area = max((px2 - px1) * (py2 - py1), 1)
        for ib in item_boxes:
            ix1, iy1, ix2, iy2 = ib
            xx1 = max(px1, ix1); yy1 = max(py1, iy1)
            xx2 = min(px2, ix2); yy2 = min(py2, iy2)
            if xx2 > xx1 and yy2 > yy1:
                inter = (xx2 - xx1) * (yy2 - yy1)
                if inter / p_area > threshold:
                    return True
        return False

    # ── per‑model detection ─────────────────────────────────

    def detect_people(self, frame: np.ndarray):
        model = model_manager.get("people")
        if model is None:
            return [], []

        results = model(frame, conf=0.3, imgsz=640, verbose=False)

        raw = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                raw.append({
                    "class": "person",
                    "confidence": float(box.conf[0]),
                    "bbox": xyxy.tolist(),
                    "center": (float(cx), float(cy)),
                    "id": None,
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

            # zone crossing
            ev = self._zone_check(det["id"], frame_w)
            if ev == "entered":
                self.stats["people_entered"] += 1
            elif ev == "exited":
                self.stats["people_exited"] += 1

        # prune disappeared
        cur_ids = {d["id"] for d in detections}
        gone = set(self._tracker) - cur_ids
        for g in gone:
            del self._tracker[g]
            self._history.pop(g, None)
        return detections

    def _zone_check(self, pid, fw):
        h = self._history.get(pid, [])
        if len(h) < 2:
            return None
        prev_x, cur_x = h[-2][0], h[-1][0]
        ent_line = fw * self._entrance_frac
        exit_line = fw * self._exit_frac
        if prev_x < ent_line <= cur_x:
            return "entered"
        if prev_x > exit_line >= cur_x:
            return "exited"
        return None

    def detect_ppe(self, frame: np.ndarray):
        model = model_manager.get("ppe")
        if model is None:
            return [], []

        results = model(frame, conf=0.35, imgsz=640, verbose=False)
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
                if not has_h:
                    missing.append("Helmet")
                if not has_v:
                    missing.append("Vest")
                key = f"ppe_{'_'.join(missing)}"
                if self._should_alert(key):
                    alerts.append(self._mkAlert_ppe(missing, pb))

        self.stats["ppe_compliance"] = round(compliant / total * 100, 1) if total else 0.0
        return detections, alerts

    def _mkAlert_ppe(self, missing, bbox):
        crit = "Helmet" in missing
        return {
            "type": "PPE_VIOLATION",
            "severity": "critical" if crit else "warning",
            "message": f"Worker missing: {', '.join(missing)}",
            "timestamp": datetime.now().isoformat(),
            "bbox": bbox,
        }

    def detect_fire(self, frame: np.ndarray):
        model = model_manager.get("fire")
        if model is None:
            return [], []

        results = model(frame, conf=0.3, imgsz=640, verbose=False)
        detections, alerts = [], []
        fire_seen = smoke_seen = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = self.FIRE_CLASSES[cls] if cls < len(self.FIRE_CLASSES) else "other"
                if name == "other":
                    continue
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                detections.append({"class": name, "confidence": float(box.conf[0]), "bbox": xyxy})
                if name == "fire":
                    fire_seen = True
                    if self._should_alert("fire"):
                        alerts.append(self._mkalert("FIRE_DETECTED", "critical",
                                                    "FIRE DETECTED! Immediate action required!", xyxy))
                elif name == "smoke":
                    smoke_seen = True
                    if self._should_alert("smoke"):
                        alerts.append(self._mkalert("SMOKE_DETECTED", "high",
                                                    "Smoke detected – investigate immediately.", xyxy))
        if fire_seen:
            self.stats["fire_risk"] = "CRITICAL"
        elif smoke_seen:
            self.stats["fire_risk"] = "High"
        else:
            self.stats["fire_risk"] = "Safe"
        return detections, alerts

    def detect_spills(self, frame: np.ndarray):
        model = model_manager.get("spill")
        if model is None:
            return [], []

        results = model(frame, conf=0.4, imgsz=640, verbose=False)
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
                                                f"{sev_label} spill detected – cleaning required.", xyxy))
        self.stats["spill_count"] = count
        return detections, alerts

    def detect_falls(self, frame: np.ndarray):
        model = model_manager.get("fall")
        if model is None:
            return [], []

        results = model(frame, conf=0.35, imgsz=640, verbose=False)
        detections, alerts = [], []
        falls = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = self.FALL_CLASSES[cls] if cls < len(self.FALL_CLASSES) else f"cls_{cls}"
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                detections.append({"class": name, "confidence": float(box.conf[0]), "bbox": xyxy})
                if name == "Falling":
                    falls += 1
                    if self._should_alert("fall"):
                        alerts.append(self._mkalert("FALL_DETECTED", "critical",
                                                    "FALL DETECTED! Emergency response required!", xyxy))
        self.stats["fall_count"] = falls
        return detections, alerts

    # ── composite processing ────────────────────────────────

    MODE_MAP = {
        "all":    ["people", "ppe", "fire", "spill", "fall"],
        "people": ["people"],
        "ppe":    ["ppe"],
        "fire":   ["fire"],
        "spill":  ["spill"],
        "fall":   ["fall"],
    }
    DETECT_FN = {
        "people": "detect_people",
        "ppe":    "detect_ppe",
        "fire":   "detect_fire",
        "spill":  "detect_spills",
        "fall":   "detect_falls",
    }

    def process_frame(self, frame: np.ndarray, mode: str = "all"):
        dets, alerts = [], []
        models_to_run = self.MODE_MAP.get(mode, self.MODE_MAP["all"])
        for m in models_to_run:
            try:
                fn = getattr(self, self.DETECT_FN[m])
                d, a = fn(frame)
                dets.extend(d)
                alerts.extend(a)
            except Exception as exc:
                logger.warning(f"Model '{m}' error: {exc}")
        self.stats["active_alerts"] = len(alerts)
        return dets, alerts

    def draw(self, frame: np.ndarray, detections: list, mode: str = "all"):
        """Draw bounding boxes & labels onto frame."""
        h, w = frame.shape[:2]

        # entrance / exit zone indicators (only in people / all mode)
        if mode in ("all", "people"):
            ent_x = int(w * self._entrance_frac)
            ext_x = int(w * self._exit_frac)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (ent_x, h), (0, 180, 0), -1)
            cv2.rectangle(overlay, (ext_x, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
            cv2.putText(frame, "ENTRANCE", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
            cv2.putText(frame, "EXIT", (ext_x + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cls = det["class"]
            conf = det["confidence"]
            color = self.COLORS.get(cls, (255, 255, 255))

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            if cls == "person" and det.get("id") is not None:
                label = f"ID:{det['id']}  {conf:.0%}"
            else:
                label = f"{cls} {conf:.0%}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Overlay counters
        if mode in ("all", "people"):
            cv2.putText(frame, f"People: {self.stats['people_count']}",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"In: {self.stats['people_entered']}  Out: {self.stats['people_exited']}",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return frame

    # ── alert helpers ───────────────────────────────────────

    def _mkalert(self, atype, sev, msg, bbox=None):
        a = {"type": atype, "severity": sev, "message": msg,
             "timestamp": datetime.now().isoformat()}
        if bbox:
            a["bbox"] = bbox
        return a

    _mkAlert = _mkalert  # alias


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
    """Manages the current video source (webcam or uploaded file)."""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.source = None  # 0 for webcam, or file path
        self._upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(self._upload_dir, exist_ok=True)

    def open(self, source=0):
        if self.cap is not None:
            self.cap.release()
        # Prefer DirectShow on Windows for better camera quality
        if isinstance(source, int) and os.name == 'nt':
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap = None
            return False
        # Request Full HD + autofocus from webcam (no effect on files)
        if source == 0 or source == 1:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)            # trigger autofocus
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)     # auto exposure on
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # minimize buffering lag
        self.source = source
        return True

    def read(self):
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret and isinstance(self.source, str):
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def save_upload(self, data: bytes, filename: str) -> str:
        path = os.path.join(self._upload_dir, filename)
        with open(path, "wb") as f:
            f.write(data)
        return path


video_source = VideoSource()


# ── Thread-safe inference wrapper ────────────────────
def _run_inference(infer_frame, mode_str, display_shape=None):
    """Run YOLO inference on inference-sized frame, scale boxes to display resolution."""
    dets, alerts = engine.process_frame(infer_frame, mode_str)
    # Scale detection coordinates from inference frame to display frame
    if display_shape is not None:
        ih, iw = infer_frame.shape[:2]
        dh, dw = display_shape[:2]
        if iw != dw or ih != dh:
            sx, sy = dw / iw, dh / ih
            for det in dets:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    return dets, alerts


# ==================== API ROUTES ====================

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>SafetyVision AI</h1><p>Place index.html in static/</p>")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": {k: ("loaded" if k in model_manager.models else "missing")
                   for k in model_manager.model_paths},
    }


@app.get("/api/stats")
async def get_stats():
    return engine.stats


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    allowed = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format. Use: {', '.join(allowed)}")
    data = await file.read()
    path = video_source.save_upload(data, f"upload{ext}")
    ok = video_source.open(path)
    return {"success": ok, "source": path}


@app.post("/api/use-webcam")
async def use_webcam():
    ok = video_source.open(0)
    return {"success": ok, "source": "webcam"}


# ==================== WEBSOCKET ENDPOINT ====================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)

    # Try to open default source if nothing is open
    if video_source.cap is None:
        video_source.open(0)

    mode = "all"
    running = True

    # ── Quality-optimised settings ───────────────────────
    TARGET_FPS = 10
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    JPEG_QUALITY = 95        # high quality for sharp display
    DISPLAY_MAX_W, DISPLAY_MAX_H = 1920, 1080  # display resolution cap
    INFER_MAX = 640          # inference resolution cap (max dimension)

    try:
        while running:
            loop_start = time.monotonic()

            # ── 1. Drain client messages (non-blocking) ─────
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.005)
                msg = json.loads(raw)
                if "mode" in msg:
                    mode = msg["mode"]
                    logger.info(f"Mode → {mode}")
                if msg.get("action") == "emergency_stop":
                    running = False
                    break
                if "source" in msg:
                    src = msg["source"]
                    if src == "webcam":
                        video_source.open(0)
            except asyncio.TimeoutError:
                pass

            # ── 2. Read frame ───────────────────────────────
            ret, frame = video_source.read()
            if not ret or frame is None:
                await ws_manager.send({
                    "type": "no_source",
                    "stats": dict(engine.stats),
                    "timestamp": datetime.now().isoformat(),
                }, ws)
                await asyncio.sleep(0.5)
                continue

            # ── Display frame: cap at display resolution, never upscale ──
            h, w = frame.shape[:2]
            display_frame = frame
            if w > DISPLAY_MAX_W or h > DISPLAY_MAX_H:
                dscale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h)
                display_frame = cv2.resize(frame,
                    (int(w * dscale), int(h * dscale)),
                    interpolation=cv2.INTER_LANCZOS4)

            # ── Inference frame: smaller copy for fast YOLO ──
            dh, dw = display_frame.shape[:2]
            if dw > INFER_MAX or dh > INFER_MAX:
                iscale = min(INFER_MAX / dw, INFER_MAX / dh)
                infer_frame = cv2.resize(display_frame,
                    (int(dw * iscale), int(dh * iscale)),
                    interpolation=cv2.INTER_LANCZOS4)
            else:
                infer_frame = display_frame

            # ── 3. Run YOLO inference in thread pool ────────
            detections, alerts = await asyncio.get_event_loop().run_in_executor(
                _inference_pool, _run_inference, infer_frame, mode,
                display_frame.shape
            )

            # ── 4. Encode display frame as high-quality JPEG ──
            _, buf = cv2.imencode(".jpg", display_frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            b64 = base64.b64encode(buf).decode()

            # ── 5. Serialise & send ─────────────────────────
            clean_dets = [{k: v for k, v in d.items() if k != "center"}
                          for d in detections]

            fh, fw = display_frame.shape[:2]
            stats_snap = dict(engine.stats)
            await ws_manager.send({
                "type": "frame",
                "frame": b64,
                "frameWidth": fw,
                "frameHeight": fh,
                "detections": clean_dets,
                "alerts": alerts,
                "stats": stats_snap,
                "timestamp": datetime.now().isoformat(),
            }, ws)

            # ── 6. Pace to target FPS ───────────────────────
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0, FRAME_INTERVAL - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(f"WS error: {exc}")
    finally:
        ws_manager.disconnect(ws)


# ==================== IMAGE UPLOAD DETECTION ====================

@app.post("/api/detect-image")
async def detect_image(file: UploadFile = File(...), mode: str = "all"):
    """Run detection on a single uploaded image and return detection results.
    The frontend displays the original file directly — no re-encoding needed."""
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")

    h, w = frame.shape[:2]

    # Create smaller inference frame for fast YOLO — original kept for display
    infer_frame = frame
    if w > 640 or h > 640:
        iscale = min(640 / w, 640 / h)
        infer_frame = cv2.resize(frame, (int(w * iscale), int(h * iscale)),
                                 interpolation=cv2.INTER_LANCZOS4)

    detections, alerts = engine.process_frame(infer_frame, mode)

    # Scale coordinates from inference frame back to original resolution
    ih, iw = infer_frame.shape[:2]
    if iw != w or ih != h:
        sx, sy = w / iw, h / ih
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            det["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

    clean_dets = [{k: v for k, v in d.items() if k != "center"} for d in detections]

    return {
        "frameWidth": w,
        "frameHeight": h,
        "detections": clean_dets,
        "alerts": alerts,
        "stats": dict(engine.stats),
    }


# ==================== ENTRYPOINT ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
