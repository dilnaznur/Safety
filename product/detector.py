import base64
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from shared.detection_logic import DetectionEngine

logger = logging.getLogger("SafetyVisionProduct")


class CameraWorker:
    def __init__(self, camera_cfg: Dict[str, Any], fps: float, mode: str = "all"):
        self.camera_cfg = camera_cfg
        self.camera_id = camera_cfg["id"]
        self.camera_name = camera_cfg.get("name", self.camera_id)
        self.rtsp_url = camera_cfg["rtsp_url"]
        self.mode = mode
        self.interval = 1.0 / max(fps, 0.2)

        self.engine = DetectionEngine()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

        self._lock = threading.Lock()
        self.latest_payload: Dict[str, Any] = {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "connected": False,
            "frame": None,
            "detections": [],
            "alerts": [],
            "stats": {
                "people_count": 0,
                "ppe_compliance": 0.0,
                "fire_risk": "Safe",
                "active_alerts": 0,
                "spill_count": 0,
                "fall_count": 0,
            },
            "timestamp": datetime.now().isoformat(),
            "error": "Camera not started",
        }

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.camera_id}")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._release_capture()

    def set_mode(self, mode: str):
        self.mode = mode

    def get_latest(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.latest_payload))

    def _release_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _open_capture(self) -> bool:
        self._release_capture()
        self.cap = cv2.VideoCapture(self.rtsp_url)
        ok = bool(self.cap and self.cap.isOpened())
        if not ok:
            logger.warning("[%s] Could not open RTSP stream", self.camera_id)
        return ok

    def _summarize_stats(self, detections: List[Dict[str, Any]], alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        person_boxes: List[List[float]] = []
        ppe_person_boxes: List[List[float]] = []
        helmet_boxes: List[List[float]] = []
        vest_boxes: List[List[float]] = []
        fire_seen = False
        smoke_seen = False
        spill_count = 0
        fall_count = 0

        for det in detections:
            cls_name = str(det.get("class", "")).strip()
            cls_lower = cls_name.lower()
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            if cls_lower == "person":
                person_boxes.append(bbox)
            elif cls_name == "Person":
                ppe_person_boxes.append(bbox)
            elif cls_name == "Helmet":
                helmet_boxes.append(bbox)
            elif cls_name == "Vest":
                vest_boxes.append(bbox)
            elif cls_lower == "fire":
                fire_seen = True
            elif cls_lower == "smoke":
                smoke_seen = True
            elif cls_name.startswith("Spill"):
                spill_count += 1
            elif cls_name == "Falling":
                fall_count += 1

        if not person_boxes:
            person_boxes = ppe_person_boxes

        compliant = 0
        for person_box in person_boxes:
            has_h = self.engine._iou_overlap(person_box, helmet_boxes)
            has_v = self.engine._iou_overlap(person_box, vest_boxes)
            if has_h and has_v:
                compliant += 1

        people_count = len(person_boxes)
        ppe_compliance = round((compliant / people_count) * 100, 1) if people_count else 0.0
        fire_risk = "Critical" if fire_seen else ("High" if smoke_seen else "Safe")

        return {
            "people_count": people_count,
            "ppe_compliance": ppe_compliance,
            "fire_risk": fire_risk,
            "active_alerts": len(alerts),
            "spill_count": spill_count,
            "fall_count": fall_count,
        }

    def _draw_overlay(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cls_name = str(det.get("class", "object"))
            conf = float(det.get("confidence", 0.0))
            color = self.engine.COLORS.get(cls_name, (255, 255, 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(out, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return out

    def _run(self):
        logger.info("[%s] Worker started", self.camera_id)
        while not self.stop_event.is_set():
            loop_start = time.time()

            if self.cap is None and not self._open_capture():
                self._set_error("Could not connect to camera")
                time.sleep(3.0)
                continue

            assert self.cap is not None
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._set_error("Failed to read frame")
                self._release_capture()
                time.sleep(2.0)
                continue

            try:
                detections, alerts = self.engine.process_frame(frame, self.mode)
                stats = self._summarize_stats(detections, alerts)
                self.engine.stats.update(stats)

                annotated = self._draw_overlay(frame, detections)
                ok_enc, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                b64 = base64.b64encode(buf).decode("utf-8") if ok_enc else None

                payload = {
                    "camera_id": self.camera_id,
                    "camera_name": self.camera_name,
                    "connected": True,
                    "frame": b64,
                    "detections": [{k: v for k, v in d.items() if k != "center"} for d in detections],
                    "alerts": alerts,
                    "stats": stats,
                    "timestamp": datetime.now().isoformat(),
                    "error": None,
                }
                with self._lock:
                    self.latest_payload = payload
            except Exception as exc:
                logger.exception("[%s] Processing error: %s", self.camera_id, exc)
                self._set_error(f"Processing error: {exc}")

            elapsed = time.time() - loop_start
            sleep_for = self.interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        logger.info("[%s] Worker stopped", self.camera_id)

    def _set_error(self, message: str):
        payload = self.get_latest()
        payload.update(
            {
                "connected": False,
                "error": message,
                "timestamp": datetime.now().isoformat(),
            }
        )
        with self._lock:
            self.latest_payload = payload


class CameraManager:
    def __init__(self, cameras: List[Dict[str, Any]], fps_per_camera: float, mode: str = "all"):
        self.workers: Dict[str, CameraWorker] = {}
        for cam in cameras:
            if cam.get("enabled", True):
                self.workers[cam["id"]] = CameraWorker(cam, fps=fps_per_camera, mode=mode)

    def start_all(self):
        for worker in self.workers.values():
            worker.start()

    def stop_all(self):
        for worker in self.workers.values():
            worker.stop()

    def camera_list(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        for cid, worker in self.workers.items():
            latest = worker.get_latest()
            data.append(
                {
                    "id": cid,
                    "name": worker.camera_name,
                    "connected": bool(latest.get("connected")),
                    "error": latest.get("error"),
                }
            )
        return data

    def get_latest(self, camera_id: str) -> Optional[Dict[str, Any]]:
        worker = self.workers.get(camera_id)
        if not worker:
            return None
        return worker.get_latest()

    def set_mode(self, camera_id: str, mode: str) -> bool:
        worker = self.workers.get(camera_id)
        if not worker:
            return False
        worker.set_mode(mode)
        return True
