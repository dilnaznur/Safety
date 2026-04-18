# SafetyVision AI - Complete Architecture Guide

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SafetyVision AI Platform                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │  DEMO MODE       │  │   PRODUCTION     │  │  DEVELOPMENT     │
│  │  (Competition)   │  │  (Surveillance)  │  │  (Local Test)    │
│  │                  │  │                  │  │                  │
│  │ • Sample videos  │  │ • RTSP streams   │  │ • Webcam         │
│  │ • Instant results│  │ • ONVIF cameras  │  │ • Video files    │
│  │ • Pre-computed   │  │ • Multi-camera   │  │ • Images         │
│  │ • Fast UI        │  │ • Real alerts    │  │ • Debug mode     │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
│           │                    │                      │
│           └────────┬───────────┴──────────┬───────────┘
│                    │                      │
│          ┌─────────▼──────────┐  ┌────────▼──────────┐
│          │ Video Source Layer │  │ Config Layer      │
│          ├────────────────────┤  ├───────────────────┤
│          │ • RTSPConnector    │  │ • DeploymentMode  │
│          │ • ONVIFConnector   │  │ • Optimization    │
│          │ • CameraPool       │  │ • Performance     │
│          │ • VideoCapture     │  │ • Alerts          │
│          └────────┬───────────┘  └──────────────────┘
│                   │
│          ┌────────▼─────────────────┐
│          │ Performance Layer         │
│          ├──────────────────────────┤
│          │ • FPSController          │
│          │ • FrameOptimizer         │
│          │ • InferenceProfiler      │
│          │ • AdaptiveProcessor      │
│          └────────┬─────────────────┘
│                   │
│          ┌────────▼──────────────────┐
│          │ Detection Engine          │
│          ├───────────────────────────┤
│          │ • YOLOv8 Models (5x)      │
│          │ • Person Detection        │
│          │ • PPE Compliance Check    │
│          │ • Fire/Smoke Detection    │
│          │ • Fall Detection          │
│          │ • Spill Detection         │
│          └────────┬──────────────────┘
│                   │
│    ┌──────────────┼──────────────┐
│    │              │              │
│ ┌──▼──────┐  ┌───▼────┐  ┌─────▼────┐
│ │ Telegram │  │Webhooks│  │  Local   │
│ │ Alerts   │  │        │  │  Logging │
│ │          │  │        │  │          │
│ └──────────┘  └────────┘  └──────────┘
│
│          ┌────────────────────────────┐
│          │ Web Frontend (FastAPI)     │
│          ├────────────────────────────┤
│          │ • Real-time WebSocket      │
│          │ • REST API Endpoints       │
│          │ • Static HTML/JS           │
│          │ • Statistics Dashboard     │
│          │ • Alert Notifications      │
│          └────────────────────────────┘
│
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Deployment Modes

### 1. DEMO MODE (For Competitions)

**Purpose**: Showcase AI capabilities with instant results

**Features**:

- Pre-recorded video samples
- Instant precomputed detection results (no latency)
- Professional dashboard for live demonstrations
- Multiple sample scenarios (fire, falls, PPE violations, spills)

**Setup**:

```bash
# Create sample directory
mkdir -p samples/precomputed

# Copy demo videos
cp your_demo_videos/*.mp4 samples/

# Set environment
export DEPLOYMENT_MODE=demo
export TARGET_FPS=15  # Smooth UI

# Generate precomputed results (offline)
python -c "
from demo_mode import DemoPrecomputeGenerator
from app import engine

gen = DemoPrecomputeGenerator('samples', 'samples/precomputed')
gen.generate_all(engine)
"

# Run
uvicorn app:app --host 0.0.0.0 --port 8000
```

**What makes it instant**:

- Precomputed JSON with detection results
- No real-time inference needed
- Only frame playback and result display
- Perfect for live demos

### 2. PRODUCTION MODE (For Surveillance Systems)

**Purpose**: Real-time monitoring with actual surveillance cameras

**Features**:

- RTSP/ONVIF camera support
- Multi-camera monitoring
- Real critical alerts
- Optimized for continuous operation
- 5 FPS default (reduce CPU load)

**Setup - Option A: RTSP Direct**

```bash
export DEPLOYMENT_MODE=production
export RTSP_URL="rtsp://username:password@192.168.1.100:554/stream"
export TARGET_FPS=5
export OPTIMIZATION_LEVEL=optimized
```

**Setup - Option B: ONVIF Auto-discovery**

```bash
export DEPLOYMENT_MODE=production
export ONVIF_ENABLED=true
export ONVIF_HOST="192.168.1.100"
export ONVIF_PORT=8080
export ONVIF_USERNAME="admin"
export ONVIF_PASSWORD="password"
```

**Setup - Multi-camera**

```python
# config_cameras.py
from surveillance_connector import CameraPool

pool = CameraPool(max_cameras=8)

# Add cameras
pool.add_camera(
    "entrance",
    rtsp_url="rtsp://user:pass@192.168.1.100:554/stream1",
)

pool.add_camera(
    "warehouse",
    onvif_host="192.168.1.101",
    onvif_username="admin",
    onvif_password="password"
)

pool.add_camera(
    "office",
    rtsp_url="rtsp://192.168.1.102:554/stream",
)
```

### 3. DEVELOPMENT MODE (For Testing)

**Purpose**: Local development with webcam/video files

```bash
export DEPLOYMENT_MODE=development
export TARGET_FPS=10
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## ⚡ Performance Optimization

### Key Optimization Strategies

#### 1. **Inference Optimization**

```python
# config.py
performance = PerformanceConfig(
    target_fps=1,  # Process 1 frame per second
    optimization_level=OptimizationLevel.OPTIMIZED,  # INT8 quantization
    inference_width=416,  # Smaller input
    inference_height=416,
)
```

**Impact on Speed**:

- **HIGH_QUALITY** (640x640): 16-30 FPS (best accuracy)
- **BALANCED** (512x512): 8-15 FPS (default)
- **OPTIMIZED** (416x416): 3-8 FPS (INT8 quantized)
- **EXTREME** (320x320): 1-3 FPS (mobile-ready)

#### 2. **Frame Skipping**

```python
# Only process every Nth frame
target_fps = 1  # Process 1 frame per second
frame_interval = 1.0 / target_fps  # Skip frames between processing
```

#### 3. **Adaptive Quality**

```python
from performance import AdaptiveProcessor

processor = AdaptiveProcessor(target_fps=5, min_fps=1, max_fps=15)

# System automatically adjusts quality based on FPS performance
# If can't maintain 5 FPS → reduce to BALANCED mode
# If easily exceeds target → boost to HIGH_QUALITY
```

#### 4. **GPU Acceleration**

```python
# models/app.py
results = model(frame, conf=0.35, imgsz=640, device=0)  # device=0 for GPU
```

#### 5. **Batch Processing** (Future)

```python
from performance import BatchInferenceBuffer

buffer = BatchInferenceBuffer(batch_size=4)
# Process 4 frames at once for better GPU utilization
```

### Optimization Tips for 1 FPS

If you truly need only 1 frame per second:

```python
# Option 1: Skip frames aggressively
config.performance.target_fps = 1
config.performance.inference_width = 320  # Very small
config.performance.optimization_level = OptimizationLevel.EXTREME

# Option 2: Process every Nth frame
# Capture at 30 FPS, process every 30th frame = 1 FPS inference

# Option 3: Use smaller models
# Consider YOLOv8n (nano) instead of full models
```

## 📊 API Endpoints

### REST Endpoints

```
GET  /                      - Main UI
GET  /health                - System health
GET  /api/stats             - Detection statistics
GET  /api/config            - Current configuration

POST /api/detect-image      - Single image detection
POST /api/upload-video      - Upload video file
POST /api/use-webcam        - Use webcam source
POST /api/telegram-test     - Test Telegram alerts

# Demo endpoints
GET  /api/demo/samples      - List demo samples
POST /api/demo/load         - Load demo sample
GET  /api/demo/progress     - Get playback progress
```

### WebSocket Endpoint

```javascript
// ws://localhost:8000/ws
const ws = new WebSocket("ws://localhost:8000/ws");

// Send commands
ws.send(
  JSON.stringify({
    mode: "all", // or "people", "fire", "ppe", etc.
    source: "webcam",
  }),
);

// Receive frames
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.frame - base64 encoded JPEG
  // data.detections - detection results
  // data.alerts - active alerts
  // data.stats - current statistics
};
```

## 🐳 Docker Deployment

### Production Docker Compose

```yaml
# docker-compose.yml
version: "3.8"

services:
  safetyvision:
    build: .
    ports:
      - "8000:8000"
    environment:
      DEPLOYMENT_MODE: production
      RTSP_URL: rtsp://camera:554/stream
      BOT_TOKEN: ${BOT_TOKEN}
      CHAT_ID: ${CHAT_ID}
      TARGET_FPS: 5
      OPTIMIZATION_LEVEL: optimized
    volumes:
      - ./logs:/app/logs
      - ./samples:/app/samples
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 2G
```

## 📈 Monitoring & Debugging

### Performance Monitoring

```python
# Get detailed metrics
stats = adaptive_processor.get_detailed_stats()

# {
#   "metrics": {
#     "fps_actual": 8.5,
#     "fps_target": 10.0,
#     "inference_ms": 85.3,
#     "frame_read_ms": 5.2,
#     "encode_ms": 12.1,
#     "total_ms": 102.6,
#   },
#   "profiler": {...},
#   "bottleneck": "inference",
#   "quality_level": "medium"
# }
```

### Bottleneck Analysis

The system automatically identifies what's slowing you down:

- **inference** → Reduce model size or use GPU
- **read** → Check RTSP connection, use buffering
- **encode** → Lower JPEG quality or use hardware encoder

## 🔧 Configuration Examples

### Example 1: 1 FPS Surveillance

```env
DEPLOYMENT_MODE=production
TARGET_FPS=1
OPTIMIZATION_LEVEL=optimized
RTSP_URL=rtsp://camera/stream
```

### Example 2: Multi-Camera Demo

```env
DEPLOYMENT_MODE=demo
TARGET_FPS=15
# Load multiple samples in rotation
```

### Example 3: GPU-Accelerated

```env
DEPLOYMENT_MODE=production
USE_GPU=true
TARGET_FPS=20
OPTIMIZATION_LEVEL=high
```

## 🔐 Security Considerations

1. **RTSP Credentials**: Use environment variables, never hardcode
2. **Telegram Bot**: Keep token secret, use .env file
3. **ONVIF**: Disable if not needed
4. **Network**: Run behind firewall, use HTTPS in production
5. **Logs**: Regularly rotate and secure alert logs

## 📝 Troubleshooting

| Issue                  | Solution                                                   |
| ---------------------- | ---------------------------------------------------------- |
| Low FPS                | Reduce inference_width, use OptimizationLevel.OPTIMIZED    |
| RTSP Connection Failed | Check URL format, verify credentials, test with ffplay     |
| High CPU Usage         | Lower target_fps, reduce frame resolution                  |
| GPU Not Used           | Verify CUDA installation, check device=0                   |
| Memory Leak            | Clear frame buffers, check VideoCapture release            |
| Telegram Not Sending   | Verify BOT_TOKEN and CHAT_ID, test with /api/telegram-test |

## 🚀 Quick Start Commands

```bash
# Demo mode (for competition)
export DEPLOYMENT_MODE=demo
export TARGET_FPS=15
uvicorn app:app --host 0.0.0.0 --port 8000

# Production mode (for surveillance)
export DEPLOYMENT_MODE=production
export RTSP_URL="rtsp://camera/stream"
export BOT_TOKEN="your_token"
export CHAT_ID="your_chat_id"
uvicorn app:app --host 0.0.0.0 --port 8000

# Development mode
export DEPLOYMENT_MODE=development
python app.py

# Install with GPU support
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Next Steps

1. **Set up demo samples** → Copy videos to `samples/` folder
2. **Configure cameras** → Add RTSP URLs or ONVIF devices
3. **Test locally** → Run in development mode with webcam
4. **Generate precomputed** → For demo mode, pre-process samples
5. **Deploy to production** → Use Docker Compose
6. **Monitor performance** → Check metrics dashboard
7. **Set up alerts** → Configure Telegram notifications
