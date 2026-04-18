"""
Shared detection logic for SafetyVision products.
Extracted from app.py to be reused by demo and product apps.
"""

import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("SafetyVision")


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
