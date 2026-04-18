"""
Surveillance system connector
Supports RTSP, ONVIF, and IP cameras
"""

import cv2
import logging
import time
from typing import Optional, Tuple
from threading import Thread, Event
from queue import Queue, Empty
import urllib.parse

logger = logging.getLogger("SafetyVision.Surveillance")


class RTSPConnector:
    """
    RTSP stream connector with auto-reconnection
    Handles authentication and stream buffering
    """
    
    def __init__(self, rtsp_url: str, username: Optional[str] = None, 
                 password: Optional[str] = None, timeout: int = 10,
                 reconnect_interval: int = 5, buffer_size: int = 2):
        self.rtsp_url = rtsp_url
        self.username = username
        self.password = password
        self.timeout = timeout
        self.reconnect_interval = reconnect_interval
        self.buffer_size = buffer_size
        
        # Auth URL
        if username and password:
            # Parse URL and add credentials
            parsed = urllib.parse.urlparse(rtsp_url)
            self.auth_url = f"{parsed.scheme}://{username}:{password}@{parsed.netloc}{parsed.path}"
        else:
            self.auth_url = rtsp_url
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue: Queue = Queue(maxsize=buffer_size)
        self.running = False
        self.reader_thread: Optional[Thread] = None
        self.connected = False
        self.last_frame: Optional[Tuple] = None
        self.dropped_frames = 0
        self.total_frames = 0
        self._stop_event = Event()
    
    def connect(self) -> bool:
        """Attempt to connect to RTSP stream"""
        try:
            logger.info(f"Connecting to RTSP: {self.rtsp_url}")
            
            # Create capture with specific backend
            self.cap = cv2.VideoCapture(self.auth_url)
            
            if not self.cap.isOpened():
                logger.error("Failed to open RTSP stream")
                self.cap = None
                return False
            
            # Configure buffer to prevent lag
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Request FPS
            
            self.connected = True
            logger.info("RTSP connection established")
            
            # Start background reader thread
            if not self.reader_thread or not self.reader_thread.is_alive():
                self.running = True
                self.reader_thread = Thread(target=self._reader_loop, daemon=True)
                self.reader_thread.start()
            
            return True
        
        except Exception as e:
            logger.error(f"RTSP connection error: {e}")
            self.connected = False
            return False
    
    def _reader_loop(self):
        """Background thread that continuously reads frames"""
        reconnect_attempts = 0
        
        while self.running and not self._stop_event.is_set():
            try:
                if not self.cap or not self.cap.isOpened():
                    if reconnect_attempts < 3:
                        logger.warning(f"Reconnecting... (attempt {reconnect_attempts + 1}/3)")
                        time.sleep(self.reconnect_interval)
                        if not self.connect():
                            reconnect_attempts += 1
                            continue
                    else:
                        logger.error("Max reconnection attempts reached")
                        break
                
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning("Failed to read frame from RTSP")
                    self.connected = False
                    continue
                
                # Add to queue with overflow handling
                try:
                    self.frame_queue.put((frame, time.time()), block=False)
                    self.total_frames += 1
                    reconnect_attempts = 0  # Reset on success
                    
                except:  # Queue full
                    try:
                        self.frame_queue.get_nowait()  # Drop oldest
                        self.frame_queue.put((frame, time.time()), block=False)
                        self.dropped_frames += 1
                    except:
                        pass
                
            except Exception as e:
                logger.error(f"Reader loop error: {e}")
                time.sleep(0.1)
    
    def read(self) -> Tuple[bool, Optional[bytes]]:
        """
        Read next frame from stream
        Returns: (success, frame)
        """
        try:
            frame, timestamp = self.frame_queue.get(timeout=2.0)
            self.last_frame = (frame, timestamp)
            return True, frame
        except Empty:
            if self.last_frame:
                return True, self.last_frame[0]
            return False, None
    
    def get_stats(self) -> dict:
        """Get connection statistics"""
        return {
            "connected": self.connected,
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "queue_size": self.frame_queue.qsize(),
            "drop_rate": round((self.dropped_frames / max(self.total_frames, 1)) * 100, 2)
        }
    
    def disconnect(self):
        """Close connection"""
        logger.info("Disconnecting RTSP stream")
        self.running = False
        self._stop_event.set()
        
        if self.reader_thread:
            self.reader_thread.join(timeout=3)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.connected = False


class ONVIFConnector:
    """
    ONVIF protocol connector for IP cameras
    Auto-discovers profiles and streams
    """
    
    def __init__(self, host: str, port: int = 8080, username: Optional[str] = None,
                 password: Optional[str] = None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.rtsp_connector: Optional[RTSPConnector] = None
        self.profile_uri: Optional[str] = None
        
        try:
            import onvif
            self.onvif_available = True
        except ImportError:
            self.onvif_available = False
            logger.warning("ONVIF library not available. Install with: pip install onvif-zeep")
    
    def connect(self, profile_index: int = 0) -> bool:
        """
        Connect to ONVIF camera
        Discovers available profiles and retrieves RTSP URL
        """
        if not self.onvif_available:
            logger.error("ONVIF not available")
            return False
        
        try:
            from onvif import ONVIFCamera
            
            logger.info(f"Connecting to ONVIF camera at {self.host}:{self.port}")
            
            cam = ONVIFCamera(
                self.host, self.port,
                self.username, self.password,
                wsdl_dir='onvif_wsdl'  # Optional WSDL cache
            )
            
            # Get media service
            media_service = cam.create_media_service()
            
            # Get profiles
            profiles = media_service.GetProfiles()
            
            if not profiles:
                logger.error("No ONVIF profiles found")
                return False
            
            profile = profiles[profile_index]
            self.profile_uri = profile.token
            
            # Get stream URI
            stream_uri = media_service.GetStreamUri({'StreamSetup': {
                'Stream': 'RTP-Unicast',
                'Transport': {'Protocol': 'RTSP'}
            }, 'ProfileToken': profile.token})
            
            rtsp_url = stream_uri.Uri
            logger.info(f"Got RTSP URL from ONVIF: {rtsp_url}")
            
            # Connect via RTSP
            self.rtsp_connector = RTSPConnector(
                rtsp_url,
                username=self.username,
                password=self.password
            )
            
            return self.rtsp_connector.connect()
        
        except Exception as e:
            logger.error(f"ONVIF connection error: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[bytes]]:
        """Read frame from ONVIF camera"""
        if self.rtsp_connector:
            return self.rtsp_connector.read()
        return False, None
    
    def get_stats(self) -> dict:
        """Get connection statistics"""
        if self.rtsp_connector:
            return self.rtsp_connector.get_stats()
        return {"connected": False}
    
    def disconnect(self):
        """Close connection"""
        if self.rtsp_connector:
            self.rtsp_connector.disconnect()


class CameraPool:
    """
    Multi-camera support
    Manages multiple RTSP/ONVIF streams
    """
    
    def __init__(self, max_cameras: int = 8):
        self.cameras: dict = {}
        self.max_cameras = max_cameras
        self.active_index = 0
    
    def add_camera(self, name: str, rtsp_url: Optional[str] = None,
                   onvif_host: Optional[str] = None, **kwargs) -> bool:
        """Add camera to pool"""
        if len(self.cameras) >= self.max_cameras:
            logger.error(f"Max cameras ({self.max_cameras}) reached")
            return False
        
        try:
            if rtsp_url:
                connector = RTSPConnector(rtsp_url, **kwargs)
            elif onvif_host:
                connector = ONVIFConnector(onvif_host, **kwargs)
            else:
                return False
            
            if connector.connect():
                self.cameras[name] = connector
                logger.info(f"Camera '{name}' added to pool")
                return True
            return False
        
        except Exception as e:
            logger.error(f"Failed to add camera '{name}': {e}")
            return False
    
    def get_camera(self, name: str) -> Optional[RTSPConnector]:
        """Get camera by name"""
        return self.cameras.get(name)
    
    def get_frame(self, name: str) -> Tuple[bool, Optional[bytes]]:
        """Get frame from camera"""
        cam = self.get_camera(name)
        if cam:
            return cam.read()
        return False, None
    
    def list_cameras(self) -> list:
        """List all cameras"""
        return list(self.cameras.keys())
    
    def remove_camera(self, name: str):
        """Remove camera"""
        if name in self.cameras:
            self.cameras[name].disconnect()
            del self.cameras[name]
            logger.info(f"Camera '{name}' removed")
    
    def disconnect_all(self):
        """Disconnect all cameras"""
        for cam in self.cameras.values():
            cam.disconnect()
        self.cameras.clear()
        logger.info("All cameras disconnected")
