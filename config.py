"""
Configuration management for SafetyVision AI
Supports multiple deployment modes and optimization settings
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class DeploymentMode(Enum):
    """Deployment modes for different use cases"""
    DEMO = "demo"  # Competition/demo mode with samples
    PRODUCTION = "production"  # Real surveillance system
    DEVELOPMENT = "development"  # Local development


class OptimizationLevel(Enum):
    """Inference optimization levels"""
    HIGH_QUALITY = "high"      # Full precision, best accuracy (16-30 FPS)
    BALANCED = "balanced"      # Medium precision (8-15 FPS)
    OPTIMIZED = "optimized"    # INT8 quantization (1-8 FPS)
    EXTREME = "extreme"        # Super lightweight, mobile-ready


@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    
    # Frame processing
    target_fps: int = 10
    max_frame_width: int = 1920
    max_frame_height: int = 1080
    inference_width: int = 640
    inference_height: int = 640
    jpeg_quality: int = 90
    
    # Model inference
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    use_gpu: bool = True
    
    # Buffer & threading
    frame_buffer_size: int = 2
    inference_threads: int = 1
    
    # Optimization mode
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    
    def get_infer_size(self) -> tuple:
        return (self.inference_width, self.inference_height)


@dataclass
class SurveillanceConfig:
    """RTSP/IP camera configuration"""
    
    # Camera connection
    rtsp_url: Optional[str] = None
    rtsp_username: Optional[str] = None
    rtsp_password: Optional[str] = None
    rtsp_stream: str = "/stream"
    connection_timeout: int = 10
    reconnect_interval: int = 5
    
    # ONVIF (IP camera protocol)
    onvif_enabled: bool = False
    onvif_host: Optional[str] = None
    onvif_port: int = 8080
    onvif_username: Optional[str] = None
    onvif_password: Optional[str] = None
    
    # Stream quality
    preferred_profile: int = 0  # ONVIF profile index


@dataclass
class AlertConfig:
    """Alert and notification settings"""
    
    # Telegram
    telegram_enabled: bool = True
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    telegram_cooldown: float = 10.0
    
    # Alert filtering
    min_severity_for_telegram: str = "high"  # "info", "warning", "high", "critical"
    send_snapshots: bool = True
    
    # Local alerting
    log_file: Optional[str] = "logs/alerts.log"
    webhook_url: Optional[str] = None


@dataclass
class DemoConfig:
    """Demo mode configuration"""
    
    # Sample videos
    sample_dir: str = "samples"
    loop_samples: bool = True
    sample_fps: int = 10
    
    # Pre-recorded results for instant demo
    use_precomputed: bool = True
    precomputed_dir: str = "samples/precomputed"
    
    # UI
    show_confidence: bool = True
    show_tracking: bool = True
    show_statistics: bool = True


class Config:
    """Main configuration manager"""
    
    def __init__(self, mode: DeploymentMode = DeploymentMode.DEVELOPMENT):
        self.mode = mode
        self.performance = PerformanceConfig()
        self.surveillance = SurveillanceConfig()
        self.alerts = AlertConfig()
        self.demo = DemoConfig()
        
        # Load from environment variables
        self._load_from_env()
        
        # Apply mode-specific defaults
        self._apply_mode_defaults()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        
        # Deployment mode
        mode_str = os.environ.get("DEPLOYMENT_MODE", "development").lower()
        try:
            self.mode = DeploymentMode[mode_str.upper()]
        except KeyError:
            self.mode = DeploymentMode.DEVELOPMENT
        
        # Performance
        self.performance.target_fps = int(os.environ.get("TARGET_FPS", "10"))
        self.performance.use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"
        opt_level = os.environ.get("OPTIMIZATION_LEVEL", "balanced").lower()
        try:
            self.performance.optimization_level = OptimizationLevel[opt_level.upper()]
        except KeyError:
            self.performance.optimization_level = OptimizationLevel.BALANCED
        
        # Surveillance (RTSP)
        self.surveillance.rtsp_url = os.environ.get("RTSP_URL")
        self.surveillance.rtsp_username = os.environ.get("RTSP_USERNAME")
        self.surveillance.rtsp_password = os.environ.get("RTSP_PASSWORD")
        
        # Alerts (Telegram)
        self.alerts.bot_token = os.environ.get("BOT_TOKEN")
        self.alerts.chat_id = os.environ.get("CHAT_ID")
        self.alerts.telegram_enabled = bool(self.alerts.bot_token and self.alerts.chat_id)
    
    def _apply_mode_defaults(self):
        """Apply defaults based on deployment mode"""
        
        if self.mode == DeploymentMode.PRODUCTION:
            # Production: Lower FPS for efficiency, higher quality detection
            self.performance.target_fps = 5
            self.performance.optimization_level = OptimizationLevel.OPTIMIZED
            self.performance.inference_width = 416
            self.performance.inference_height = 416
            self.alerts.min_severity_for_telegram = "high"
            
        elif self.mode == DeploymentMode.DEMO:
            # Demo: Higher FPS for smooth UI, best quality
            self.performance.target_fps = 15
            self.performance.optimization_level = OptimizationLevel.HIGH_QUALITY
            self.alerts.telegram_enabled = False  # Demo doesn't send real alerts
            
        elif self.mode == DeploymentMode.DEVELOPMENT:
            # Development: Balance quality and speed
            self.performance.target_fps = 10
            self.performance.optimization_level = OptimizationLevel.BALANCED
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary"""
        return {
            "mode": self.mode.value,
            "performance": {
                "target_fps": self.performance.target_fps,
                "optimization_level": self.performance.optimization_level.value,
                "use_gpu": self.performance.use_gpu,
                "inference_size": self.performance.get_infer_size(),
            },
            "surveillance": {
                "rtsp_enabled": bool(self.surveillance.rtsp_url),
                "onvif_enabled": self.surveillance.onvif_enabled,
            },
            "alerts": {
                "telegram_enabled": self.alerts.telegram_enabled,
            }
        }


# Global config instance
config = Config(mode=DeploymentMode(os.environ.get("DEPLOYMENT_MODE", "development").lower()))
