# 🏭 SafetyVision AI - Industrial Safety Monitoring Platform

**Advanced AI-powered real-time safety monitoring for industrial facilities**

Transform your surveillance system with intelligent detection of critical safety issues - falls, fires, spills, PPE violations, and personnel tracking.

## ✨ Features

### 🎯 Multi-Mode Architecture

- **DEMO Mode**: Pre-recorded samples with instant results for competitions
- **PRODUCTION Mode**: Real-time monitoring with RTSP/ONVIF camera support
- **DEVELOPMENT Mode**: Local testing with webcam

### 🤖 5 Specialized AI Models

| Model      | Purpose                                        | Accuracy |
| ---------- | ---------------------------------------------- | -------- |
| **People** | Headcount & tracking                           | 94.4%    |
| **PPE**    | Equipment compliance (helmet, vest, gloves)    | 97.7%    |
| **Fire**   | Fire & smoke detection                         | 90.2%    |
| **Spill**  | Hazardous liquid detection (6 severity levels) | 98.7%    |
| **Fall**   | Fall & posture detection                       | 97.7%    |

### ⚡ Performance Optimization

- Configurable FPS (1-30 fps)
- Adaptive quality levels
- INT8 quantization support
- GPU acceleration capable
- Frame skipping for bandwidth optimization
- Performance profiling & bottleneck analysis

### 🔔 Real-Time Alerts

- **Telegram Notifications** with snapshots
- **Webhook Support** for custom integrations
- **Local Logging** of all events
- **Smart Rate Limiting** to avoid alert spam

### 📊 Live Dashboard

- Real-time video stream with detections
- Statistics tracking
- Performance metrics
- Alert notifications
- Multi-detection mode selector

## 🚀 Quick Start

### Installation (2 minutes)

```bash
# Clone repository
git clone <repo> guard
cd guard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Deploy in 3 Modes

#### 🎬 Demo Mode (For Competitions)

```bash
# Copy demo videos
mkdir -p samples
cp demo_*.mp4 samples/

# Generate instant results (precomputed)
python generate_precomputed_demo.py --all

# Start
export DEPLOYMENT_MODE=demo
export TARGET_FPS=15
uvicorn app:app --host 0.0.0.0 --port 8000

# Open: http://localhost:8000
```

#### 🏢 Production Mode (Surveillance)

```bash
# RTSP Camera
export DEPLOYMENT_MODE=production
export RTSP_URL="rtsp://user:pass@192.168.1.100:554/stream"
export TARGET_FPS=5
export BOT_TOKEN="your_telegram_token"
export CHAT_ID="your_chat_id"
uvicorn app:app --host 0.0.0.0 --port 8000

# Or ONVIF Camera
export ONVIF_ENABLED=true
export ONVIF_HOST="192.168.1.100"
```

#### 🔧 Development Mode (Testing)

```bash
export DEPLOYMENT_MODE=development
uvicorn app:app --reload
```

## 🏗️ Architecture

### System Components

```
Input Layer
    ├─ RTSP Camera Streams
    ├─ ONVIF Cameras (IP)
    └─ Local/Demo Videos

        ↓

Performance Layer
    ├─ FPS Controller (1-30 fps)
    ├─ Frame Optimizer (adaptive quality)
    └─ Inference Profiler (bottleneck analysis)

        ↓

Detection Engine
    ├─ People Detection + Tracking
    ├─ PPE Compliance Check
    ├─ Fire/Smoke Detection
    ├─ Fall Detection
    └─ Spill Classification

        ↓

Alert & Output
    ├─ Telegram Notifications
    ├─ Webhook Events
    ├─ WebSocket Streaming
    └─ Local Logging
```

### Configuration System

All configuration via environment variables:

```env
# Deployment
DEPLOYMENT_MODE=demo|production|development

# Performance
TARGET_FPS=5              # 1-30 fps
OPTIMIZATION_LEVEL=balanced  # high|balanced|optimized|extreme
USE_GPU=true|false

# Cameras (Production)
RTSP_URL=rtsp://user:pass@host/stream
ONVIF_ENABLED=true
ONVIF_HOST=192.168.1.100

# Alerts
BOT_TOKEN=telegram_token
CHAT_ID=chat_id
```

See [.env.example](.env.example) for all options.

## 📊 API Reference

### REST Endpoints

```bash
# Health & Config
GET    /health                    → System status
GET    /api/config               → Configuration
GET    /api/stats                → Detection statistics
GET    /api/system/stats         → Performance metrics

# Detection
POST   /api/detect-image         → Detect in single image
POST   /api/upload-video         → Upload video file
POST   /api/use-webcam           → Use webcam

# Demo Mode
GET    /api/demo/samples         → List samples
POST   /api/demo/load            → Load sample
GET    /api/demo/progress        → Playback progress

# Surveillance
GET    /api/cameras/list         → List cameras
GET    /api/cameras/{name}/stats → Camera stats

# Alerts
POST   /api/telegram-test        → Test Telegram
```

### WebSocket

```javascript
// Connect
const ws = new WebSocket("ws://localhost:8000/ws");

// Send command
ws.send(
  JSON.stringify({
    mode: "all", // or "fire", "people", "ppe", etc.
    source: "webcam",
  }),
);

// Receive frame
ws.onmessage = (event) => {
  const { frame, detections, alerts, stats } = JSON.parse(event.data);
  // frame: base64 JPEG
  // detections: [{class, confidence, bbox}, ...]
  // alerts: [{type, severity, message}, ...]
  // stats: {people_count, ppe_compliance, ...}
};
```

## 🔧 Performance Tuning

### Optimize for Speed (Real-time)

```env
TARGET_FPS=15
OPTIMIZATION_LEVEL=high
USE_GPU=true
```

Result: 15+ FPS with good accuracy

### Optimize for Bandwidth (1 FPS)

```env
TARGET_FPS=1
OPTIMIZATION_LEVEL=optimized
```

Result: 1 FPS, minimal bandwidth/CPU

### Optimize for Accuracy

```env
TARGET_FPS=5
OPTIMIZATION_LEVEL=high
INFERENCE_WIDTH=640
INFERENCE_HEIGHT=640
```

Result: Best accuracy, moderate speed

## 📈 Monitoring

Check system health:

```bash
# Real-time performance metrics
curl http://localhost:8000/api/system/stats

# Response:
{
  "metrics": {
    "fps_actual": 8.5,
    "fps_target": 10.0,
    "inference_ms": 85.3,
    "bottleneck": "inference"
  },
  "quality_level": "medium",
  "profiler": {
    "inference": {"avg_ms": 85, "max_ms": 120},
    "read": {"avg_ms": 5.2},
    "encode": {"avg_ms": 12.1}
  }
}
```

## 🐳 Docker Deployment

### Production Docker Compose

```bash
docker-compose -f docker-compose.prod.yml up -d

# Monitor
docker-compose -f docker-compose.prod.yml logs -f

# Stop
docker-compose -f docker-compose.prod.yml down
```

Configuration via `.env` file:

```env
DEPLOYMENT_MODE=production
RTSP_URL=rtsp://camera/stream
BOT_TOKEN=your_token
CHAT_ID=your_chat_id
TARGET_FPS=5
CPU_LIMIT=2
MEMORY_LIMIT=2G
```

## 🔐 Security

- **Credentials**: Use environment variables (never hardcode)
- **Network**: Restrict port 8000 to trusted networks
- **HTTPS**: Use reverse proxy (nginx) in production
- **Secrets**: Store `.env` securely, never commit to git
- **Tokens**: Rotate Telegram tokens quarterly

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed system architecture
- **[QUICK_START.md](QUICK_START.md)** - Step-by-step setup guide
- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** - Pre-deployment checklist
- **[.env.example](.env.example)** - Configuration template

## 🛠️ Troubleshooting

| Issue                     | Solution                                                    |
| ------------------------- | ----------------------------------------------------------- |
| **Low FPS**               | Lower `TARGET_FPS`, increase `OPTIMIZATION_LEVEL=optimized` |
| **High CPU**              | Use `OPTIMIZATION_LEVEL=optimized`, reduce frame size       |
| **No Detections**         | Verify models in `models/`, test with sample image          |
| **Telegram Not Working**  | Verify `BOT_TOKEN` and `CHAT_ID`, use `/api/telegram-test`  |
| **Camera Not Connecting** | Check RTSP URL, test with `ffplay`, verify network          |
| **Memory Issues**         | Lower `TARGET_FPS`, reduce resolution, restart service      |

See [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) for full troubleshooting guide.

## 📦 Models

All models are included:

- `models/people_best.pt` (94.4% mAP)
- `models/ppe_best.pt` (97.7% mAP)
- `models/fire_best.pt` (90.2% mAP)
- `models/spill_best.pt` (98.7% mAP)
- `models/fall_best.pt` (97.7% mAP)

Models automatically load on startup. Verify with:

```bash
curl http://localhost:8000/health
```

## 🚀 Advanced Features

### Multi-Camera Setup

```python
# Add multiple cameras for monitoring
camera_pool.add_camera("entrance", rtsp_url="rtsp://...")
camera_pool.add_camera("warehouse", onvif_host="192.168.1.101")
camera_pool.add_camera("office", rtsp_url="rtsp://...")
```

### Precomputed Demo Results

For instant playback without inference lag:

```bash
python generate_precomputed_demo.py --all
```

Generates JSON files with pre-processed detections for smooth demos.

### Custom Webhooks

Forward alerts to your system:

```python
# Configure webhook_url in config.alerts
# Receives POST with: {type, severity, message, bbox, confidence}
```

## 📊 Performance Stats

Typical performance on Intel i7 / RTX 3060:

| Setting      | FPS   | Latency    | CPU | GPU |
| ------------ | ----- | ---------- | --- | --- |
| High Quality | 20-30 | 30-50ms    | 60% | 80% |
| Balanced     | 10-15 | 65-100ms   | 40% | 60% |
| Optimized    | 5-10  | 100-200ms  | 25% | 40% |
| Extreme      | 1-5   | 200-1000ms | 10% | 20% |

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] More detection models
- [ ] Mobile app
- [ ] Cloud deployment
- [ ] Multi-language support
- [ ] Extended analytics

## 📝 License

[Add your license here]

## 📞 Support

- Check [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical info
- Review [QUICK_START.md](QUICK_START.md) for setup help
- Check logs for error messages
- Verify configuration with `/api/config` endpoint

## 🎯 Next Steps

1. ✅ Review [QUICK_START.md](QUICK_START.md)
2. ✅ Configure [.env](.env.example)
3. ✅ Choose deployment mode (demo/production/dev)
4. ✅ Follow [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)
5. ✅ Deploy and monitor

---

**SafetyVision AI - Making workplaces safer with AI** 🏭🤖
