"""
Demo mode handler for competitions and showcases
Pre-recorded samples with instant results
"""

import os
import cv2
import json
import logging
from typing import Optional, List, Tuple, Dict
from pathlib import Path

logger = logging.getLogger("SafetyVision.Demo")


class DemoSample:
    """Single demo sample with precomputed results"""
    
    def __init__(self, video_path: str, results_path: Optional[str] = None):
        self.video_path = video_path
        self.results_path = results_path
        self.name = Path(video_path).stem
        
        # Precomputed data
        self.precomputed_results: List[Dict] = []
        self.frame_count = 0
        self.fps = 10
        
        # Load precomputed results if available
        self._load_precomputed()
    
    def _load_precomputed(self):
        """Load precomputed detection results"""
        if not self.results_path or not os.path.exists(self.results_path):
            return
        
        try:
            with open(self.results_path, 'r') as f:
                data = json.load(f)
                self.precomputed_results = data.get('results', [])
                self.frame_count = data.get('frame_count', 0)
                self.fps = data.get('fps', 10)
                logger.info(f"Loaded precomputed results for {self.name}: {len(self.precomputed_results)} frames")
        except Exception as e:
            logger.warning(f"Could not load precomputed results: {e}")
    
    def get_result(self, frame_idx: int) -> Optional[Dict]:
        """Get precomputed result for frame"""
        if frame_idx < len(self.precomputed_results):
            return self.precomputed_results[frame_idx]
        return None
    
    def open_video(self):
        """Open video file for reading"""
        if not os.path.exists(self.video_path):
            logger.error(f"Video file not found: {self.video_path}")
            return None
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {self.video_path}")
            return None
        
        # Get metadata
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 10
        
        return cap


class DemoManager:
    """
    Manages demo samples for showcasing
    Supports instant playback with precomputed results
    """
    
    def __init__(self, sample_dir: str = "samples", precomputed_dir: Optional[str] = None):
        self.sample_dir = sample_dir
        self.precomputed_dir = precomputed_dir or os.path.join(sample_dir, "precomputed")
        self.samples: Dict[str, DemoSample] = {}
        self.current_sample: Optional[DemoSample] = None
        self.current_video: Optional[cv2.VideoCapture] = None
        self.current_frame_idx = 0
        self.playing = False
        self.loop = True
        
        # Scan for samples
        self._discover_samples()
    
    def _discover_samples(self):
        """Discover available demo samples"""
        if not os.path.exists(self.sample_dir):
            logger.warning(f"Sample directory not found: {self.sample_dir}")
            return
        
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        
        for file in os.listdir(self.sample_dir):
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(self.sample_dir, file)
                
                # Check for precomputed results
                results_path = None
                if self.precomputed_dir and os.path.exists(self.precomputed_dir):
                    results_file = Path(file).stem + ".json"
                    results_path = os.path.join(self.precomputed_dir, results_file)
                    if not os.path.exists(results_path):
                        results_path = None
                
                sample = DemoSample(video_path, results_path)
                self.samples[sample.name] = sample
                logger.info(f"Found demo sample: {sample.name} (precomputed: {results_path is not None})")
    
    def list_samples(self) -> List[str]:
        """List available samples"""
        return list(self.samples.keys())
    
    def load_sample(self, sample_name: str) -> bool:
        """Load and prepare sample"""
        if sample_name not in self.samples:
            logger.error(f"Sample not found: {sample_name}")
            return False
        
        sample = self.samples[sample_name]
        
        # Close previous video
        if self.current_video:
            self.current_video.release()
        
        # Open new video
        video_cap = sample.open_video()
        if not video_cap:
            return False
        
        self.current_sample = sample
        self.current_video = video_cap
        self.current_frame_idx = 0
        self.playing = False
        
        logger.info(f"Loaded sample: {sample_name} ({sample.frame_count} frames @ {sample.fps:.1f} FPS)")
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat], Optional[Dict]]:
        """
        Read next frame with precomputed results
        Returns: (success, frame, precomputed_result)
        """
        if not self.current_video or not self.current_sample:
            return False, None, None
        
        ret, frame = self.current_video.read()
        
        if not ret:
            if self.loop:
                self.current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.current_video.read()
            
            if not ret:
                return False, None, None
        
        # Get precomputed result
        precomputed = self.current_sample.get_result(self.current_frame_idx)
        self.current_frame_idx += 1
        
        return True, frame, precomputed
    
    def seek(self, frame_idx: int) -> bool:
        """Seek to frame"""
        if not self.current_video:
            return False
        
        self.current_video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.current_frame_idx = frame_idx
        return True
    
    def get_current_progress(self) -> dict:
        """Get playback progress"""
        if not self.current_sample:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "sample_name": self.current_sample.name,
            "current_frame": self.current_frame_idx,
            "total_frames": self.current_sample.frame_count,
            "fps": self.current_sample.fps,
            "playing": self.playing,
            "progress_percent": round((self.current_frame_idx / max(self.current_sample.frame_count, 1)) * 100, 1),
        }


class DemoPrecomputeGenerator:
    """
    Generate precomputed results for demo samples
    Runs offline to speed up demo playback
    """
    
    def __init__(self, sample_dir: str, output_dir: str):
        self.sample_dir = sample_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_for_sample(self, video_path: str, detection_engine):
        """
        Run detection on full video and save results
        This is done offline, before demo
        """
        logger.info(f"Generating precomputed results for: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open: {video_path}")
            return False
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        
        results = []
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                detections, alerts = detection_engine.process_frame(frame, mode="all")
                
                # Store results
                result = {
                    "frame_idx": frame_idx,
                    "detections": [{
                        "class": d.get("class"),
                        "confidence": d.get("confidence"),
                        "bbox": d.get("bbox"),
                        "severity": d.get("severity"),
                    } for d in detections],
                    "alerts": alerts,
                    "stats": dict(detection_engine.stats),
                }
                results.append(result)
                
                frame_idx += 1
                if frame_idx % 10 == 0:
                    logger.info(f"  Processed {frame_idx}/{frame_count} frames")
        
        finally:
            cap.release()
        
        # Save results
        output_file = os.path.join(
            self.output_dir,
            Path(video_path).stem + ".json"
        )
        
        output_data = {
            "video_path": video_path,
            "frame_count": frame_count,
            "fps": fps,
            "results": results,
            "generated_at": __import__('datetime').datetime.now().isoformat(),
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved precomputed results to: {output_file}")
        return True
    
    def generate_all(self, detection_engine):
        """Generate precomputed results for all samples"""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        
        for file in os.listdir(self.sample_dir):
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(self.sample_dir, file)
                self.generate_for_sample(video_path, detection_engine)


def create_sample_alert_example() -> Dict:
    """Create sample alert for demo UI"""
    return {
        "type": "FIRE_DETECTED",
        "severity": "critical",
        "message": "FIRE DETECTED! Immediate action required!",
        "confidence": 0.987,
        "bbox": [100, 150, 300, 400],
        "timestamp": __import__('datetime').datetime.now().isoformat(),
    }


def create_sample_statistics() -> Dict:
    """Create sample statistics for demo dashboard"""
    return {
        "people_count": 5,
        "total_people_today": 47,
        "max_people_count": 12,
        "people_entered": 23,
        "people_exited": 18,
        "ppe_compliance": 85.5,
        "fire_risk": "Safe",
        "active_alerts": 1,
        "spill_count": 0,
        "fall_count": 1,
    }
