"""
Integration example showing how to use all new modules
This demonstrates the new architecture
"""

from config import Config, DeploymentMode
from surveillance_connector import RTSPConnector, ONVIFConnector, CameraPool
from demo_mode import DemoManager, DemoPrecomputeGenerator
from performance import AdaptiveProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafetyVisionApp:
    """Main application integrating all components"""
    
    def __init__(self):
        # Load configuration from environment
        self.config = Config()
        logger.info(f"Deployment mode: {self.config.mode.value}")
        
        # Initialize video source based on mode
        self.video_source = None
        self.setup_video_source()
        
        # Initialize performance monitoring
        self.processor = AdaptiveProcessor(
            target_fps=self.config.performance.target_fps
        )
        
        # Initialize demo mode if applicable
        self.demo_manager = None
        if self.config.mode == DeploymentMode.DEMO:
            self.setup_demo_mode()
    
    def setup_video_source(self):
        """Setup video source based on deployment mode"""
        
        if self.config.mode == DeploymentMode.PRODUCTION:
            if self.config.surveillance.rtsp_url:
                logger.info("Setting up RTSP connection...")
                self.video_source = RTSPConnector(
                    rtsp_url=self.config.surveillance.rtsp_url,
                    username=self.config.surveillance.rtsp_username,
                    password=self.config.surveillance.rtsp_password,
                    timeout=self.config.surveillance.connection_timeout,
                )
                self.video_source.connect()
            
            elif self.config.surveillance.onvif_enabled:
                logger.info("Setting up ONVIF connection...")
                self.video_source = ONVIFConnector(
                    host=self.config.surveillance.onvif_host,
                    port=self.config.surveillance.onvif_port,
                    username=self.config.surveillance.onvif_username,
                    password=self.config.surveillance.onvif_password,
                )
                self.video_source.connect()
        
        elif self.config.mode == DeploymentMode.DEMO:
            # Demo mode uses DemoManager
            pass
        
        else:  # DEVELOPMENT
            logger.info("Development mode - will use webcam/uploaded videos")
    
    def setup_demo_mode(self):
        """Setup demo mode for competitions"""
        logger.info("Setting up DEMO mode...")
        
        self.demo_manager = DemoManager(
            sample_dir=self.config.demo.sample_dir,
            precomputed_dir=self.config.demo.precomputed_dir if self.config.demo.use_precomputed else None,
        )
        
        samples = self.demo_manager.list_samples()
        logger.info(f"Found {len(samples)} demo samples: {samples}")
        
        # Load first sample if available
        if samples:
            self.demo_manager.load_sample(samples[0])
            logger.info(f"Loaded sample: {samples[0]}")
    
    def get_performance_config(self) -> dict:
        """Get current performance configuration"""
        return {
            "target_fps": self.config.performance.target_fps,
            "optimization_level": self.config.performance.optimization_level.value,
            "inference_size": self.config.performance.get_infer_size(),
            "use_gpu": self.config.performance.use_gpu,
        }
    
    def get_system_status(self) -> dict:
        """Get system status"""
        return {
            "mode": self.config.mode.value,
            "performance": self.processor.get_metrics().to_dict(),
            "config": self.config.to_dict(),
            "video_source_connected": self.video_source is not None and getattr(self.video_source, 'connected', False),
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Initialize app
    app = SafetyVisionApp()
    
    # Example 1: Get configuration
    print("\n=== Configuration ===")
    print(app.config.to_dict())
    
    # Example 2: Get performance metrics
    print("\n=== Performance Metrics ===")
    print(app.processor.get_detailed_stats())
    
    # Example 3: List demo samples
    if app.demo_manager:
        print("\n=== Demo Samples ===")
        samples = app.demo_manager.list_samples()
        for sample in samples:
            print(f"  - {sample}")
    
    # Example 4: Test RTSP connection
    if isinstance(app.video_source, RTSPConnector):
        print("\n=== RTSP Stats ===")
        print(app.video_source.get_stats())
    
    # Example 5: Generate precomputed results for demo
    # This is typically run once, offline
    if app.config.mode == DeploymentMode.DEMO:
        print("\n=== Generating Precomputed Results ===")
        print("This should only be done once, offline, before deployment")
        print("Command: python app_integration.py --generate-demo")
        # from app import engine  # Use your detection engine
        # gen = DemoPrecomputeGenerator('samples', 'samples/precomputed')
        # gen.generate_all(engine)
