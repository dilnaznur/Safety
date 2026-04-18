"""
Performance optimization module
Handles frame buffering, FPS control, and inference acceleration
"""

import time
import logging
from typing import Optional, Callable
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("SafetyVision.Performance")


@dataclass
class PerformanceMetrics:
    """Track performance statistics"""
    
    fps_actual: float = 0.0
    fps_target: float = 10.0
    inference_time_ms: float = 0.0
    frame_read_time_ms: float = 0.0
    frame_encode_time_ms: float = 0.0
    total_frame_time_ms: float = 0.0
    frames_dropped: int = 0
    frames_processed: int = 0
    frame_queue_size: int = 0
    
    def to_dict(self) -> dict:
        return {
            "fps_actual": round(self.fps_actual, 1),
            "fps_target": round(self.fps_target, 1),
            "inference_ms": round(self.inference_time_ms, 2),
            "frame_read_ms": round(self.frame_read_time_ms, 2),
            "encode_ms": round(self.frame_encode_time_ms, 2),
            "total_ms": round(self.total_frame_time_ms, 2),
            "dropped_frames": self.frames_dropped,
            "processed_frames": self.frames_processed,
        }


class FPSController:
    """
    Maintains consistent target FPS
    Handles frame skipping and timing
    """
    
    def __init__(self, target_fps: float = 10.0, window_size: int = 30):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.window_size = window_size
        
        # Timing tracking
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = time.monotonic()
        self.frame_count = 0
        self.start_time = time.monotonic()
    
    def should_process(self) -> bool:
        """
        Check if enough time has passed to process next frame
        Supports frame skipping for lower FPS
        """
        current_time = time.monotonic()
        elapsed = current_time - self.last_frame_time
        
        if elapsed >= self.frame_interval:
            self.last_frame_time = current_time
            self.frame_count += 1
            self.frame_times.append(current_time)
            return True
        
        return False
    
    def sleep_to_target(self, elapsed_time: float):
        """Sleep to maintain target FPS"""
        remaining = self.frame_interval - elapsed_time
        if remaining > 0:
            time.sleep(remaining)
    
    def get_actual_fps(self) -> float:
        """Calculate actual FPS from recent frames"""
        if len(self.frame_times) < 2:
            return 0.0
        
        time_diff = self.frame_times[-1] - self.frame_times[0]
        frame_diff = len(self.frame_times) - 1
        
        if time_diff == 0:
            return 0.0
        
        return frame_diff / time_diff
    
    def reset(self):
        """Reset controller"""
        self.frame_times.clear()
        self.last_frame_time = time.monotonic()
        self.frame_count = 0
        self.start_time = time.monotonic()


class FrameOptimizer:
    """
    Optimize frame size and quality for inference
    Implements intelligent resizing based on performance
    """
    
    def __init__(self, max_inference_size: int = 640, 
                 quality_levels: Optional[dict] = None):
        self.max_inference_size = max_inference_size
        self.quality_levels = quality_levels or {
            "high": {"size": 640, "quality": 90},
            "medium": {"size": 416, "quality": 80},
            "low": {"size": 320, "quality": 70},
        }
        self.current_level = "medium"
    
    def calculate_inference_size(self, frame_height: int, frame_width: int) -> tuple:
        """
        Calculate optimal inference size maintaining aspect ratio
        """
        level = self.quality_levels.get(self.current_level, self.quality_levels["medium"])
        max_size = level["size"]
        
        scale = min(max_size / frame_width, max_size / frame_height)
        new_w = int(frame_width * scale)
        new_h = int(frame_height * scale)
        
        # Ensure dimensions are multiples of 32 (YOLO requirement)
        new_w = (new_w // 32) * 32 or 32
        new_h = (new_h // 32) * 32 or 32
        
        return (new_w, new_h)
    
    def adjust_quality_level(self, actual_fps: float, target_fps: float):
        """
        Dynamically adjust quality based on FPS performance
        """
        fps_ratio = actual_fps / target_fps if target_fps > 0 else 1.0
        
        old_level = self.current_level
        
        if fps_ratio > 1.1:  # Exceeding target, can increase quality
            if self.current_level == "low":
                self.current_level = "medium"
            elif self.current_level == "medium":
                self.current_level = "high"
        
        elif fps_ratio < 0.7:  # Can't keep up, reduce quality
            if self.current_level == "high":
                self.current_level = "medium"
            elif self.current_level == "medium":
                self.current_level = "low"
        
        if old_level != self.current_level:
            logger.info(f"Quality adjusted: {old_level} → {self.current_level} (FPS ratio: {fps_ratio:.2f})")


class BatchInferenceBuffer:
    """
    Buffer frames for batch inference
    Reduces per-frame overhead
    """
    
    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size
        self.buffer: list = []
        self.timestamps: list = []
    
    def add(self, frame, timestamp: float = None):
        """Add frame to buffer"""
        if timestamp is None:
            timestamp = time.monotonic()
        
        self.buffer.append(frame)
        self.timestamps.append(timestamp)
    
    def is_full(self) -> bool:
        """Check if buffer is ready for batch processing"""
        return len(self.buffer) >= self.batch_size
    
    def get_batch(self) -> tuple:
        """Get and clear buffer"""
        frames = self.buffer.copy()
        times = self.timestamps.copy()
        self.buffer.clear()
        self.timestamps.clear()
        return frames, times
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.timestamps.clear()


class InferenceProfiler:
    """
    Profile inference performance and identify bottlenecks
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.read_times = deque(maxlen=window_size)
        self.encode_times = deque(maxlen=window_size)
        self.total_times = deque(maxlen=window_size)
    
    def record_inference(self, time_ms: float):
        """Record inference time"""
        self.inference_times.append(time_ms)
    
    def record_read(self, time_ms: float):
        """Record frame read time"""
        self.read_times.append(time_ms)
    
    def record_encode(self, time_ms: float):
        """Record frame encoding time"""
        self.encode_times.append(time_ms)
    
    def record_total(self, time_ms: float):
        """Record total frame time"""
        self.total_times.append(time_ms)
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        return {
            "inference": {
                "avg_ms": round(sum(self.inference_times) / len(self.inference_times), 2) if self.inference_times else 0,
                "max_ms": round(max(self.inference_times), 2) if self.inference_times else 0,
                "min_ms": round(min(self.inference_times), 2) if self.inference_times else 0,
            },
            "read": {
                "avg_ms": round(sum(self.read_times) / len(self.read_times), 2) if self.read_times else 0,
                "max_ms": round(max(self.read_times), 2) if self.read_times else 0,
            },
            "encode": {
                "avg_ms": round(sum(self.encode_times) / len(self.encode_times), 2) if self.encode_times else 0,
                "max_ms": round(max(self.encode_times), 2) if self.encode_times else 0,
            },
            "total": {
                "avg_ms": round(sum(self.total_times) / len(self.total_times), 2) if self.total_times else 0,
                "max_ms": round(max(self.total_times), 2) if self.total_times else 0,
            }
        }
    
    def find_bottleneck(self) -> str:
        """Identify the slowest component"""
        if not (self.inference_times or self.read_times or self.encode_times):
            return "unknown"
        
        stats = self.get_stats()
        avg_inference = stats["inference"]["avg_ms"]
        avg_read = stats["read"]["avg_ms"]
        avg_encode = stats["encode"]["avg_ms"]
        
        times = {
            "inference": avg_inference,
            "read": avg_read,
            "encode": avg_encode,
        }
        
        bottleneck = max(times, key=times.get)
        return bottleneck


class AdaptiveProcessor:
    """
    Adaptive frame processing that adjusts based on system load
    """
    
    def __init__(self, target_fps: int = 10, min_fps: int = 1, max_fps: int = 30):
        self.target_fps = target_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        
        self.fps_controller = FPSController(target_fps)
        self.frame_optimizer = FrameOptimizer()
        self.profiler = InferenceProfiler()
        self.metrics = PerformanceMetrics(fps_target=target_fps)
        
        self.adjustment_counter = 0
        self.adjustment_interval = 100  # Adjust every N frames
    
    def should_skip_frame(self) -> bool:
        """Determine if frame should be skipped"""
        return not self.fps_controller.should_process()
    
    def on_frame_processed(self, inference_time_ms: float, read_time_ms: float, 
                          encode_time_ms: float):
        """Record frame processing metrics"""
        total_time = inference_time_ms + read_time_ms + encode_time_ms
        
        self.profiler.record_inference(inference_time_ms)
        self.profiler.record_read(read_time_ms)
        self.profiler.record_encode(encode_time_ms)
        self.profiler.record_total(total_time)
        
        self.metrics.inference_time_ms = inference_time_ms
        self.metrics.frame_read_time_ms = read_time_ms
        self.metrics.frame_encode_time_ms = encode_time_ms
        self.metrics.total_frame_time_ms = total_time
        self.metrics.fps_actual = self.fps_controller.get_actual_fps()
        self.metrics.frames_processed += 1
        
        # Periodic adjustment
        self.adjustment_counter += 1
        if self.adjustment_counter % self.adjustment_interval == 0:
            self.frame_optimizer.adjust_quality_level(
                self.metrics.fps_actual,
                self.metrics.fps_target
            )
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.metrics
    
    def get_detailed_stats(self) -> dict:
        """Get detailed performance statistics"""
        return {
            "metrics": self.metrics.to_dict(),
            "profiler": self.profiler.get_stats(),
            "bottleneck": self.profiler.find_bottleneck(),
            "quality_level": self.frame_optimizer.current_level,
        }
