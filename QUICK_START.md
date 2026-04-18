# SafetyVision AI - Quick Start Guide

## 🚀 Installation & Setup

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd guard

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Optional: Install ONVIF support
pip install onvif-zeep
```

### Step 2: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings
# - Set DEPLOYMENT_MODE (development, demo, production)
# - Add RTSP URL if needed
# - Configure Telegram alerts
```

## 🎯 Three Deployment Modes

### Mode 1: DEMO (For Competitions) ⚡

**Best for**: Live demonstrations, showcases, competitions

```bash
# 1. Create samples directory
mkdir -p samples/precomputed

# 2. Copy your demo videos
cp /path/to/demo/*.mp4 samples/

# 3. Set configuration
export DEPLOYMENT_MODE=demo
export TARGET_FPS=15

# 4. Run the app
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Advanced: Pre-generate results for instant playback**

```bash
# This makes the demo super fast (no inference lag)
python -c "
from demo_mode import DemoPrecomputeGenerator
from app import engine

gen = DemoPrecomputeGenerator('samples', 'samples/precomputed')
print('Generating precomputed results...')
gen.generate_all(engine)
print('Done! Now your demos will play instantly with results.')
"
```

**Access demo**: Open http://localhost:8000 → Select sample from dropdown

---

### Mode 2: PRODUCTION (For Surveillance) 🏢

**Best for**: Real-time monitoring, surveillance systems, continuous operation

#### Option A: RTSP Camera

```bash
# Set configuration
export DEPLOYMENT_MODE=production
export RTSP_URL="rtsp://username:password@192.168.1.100:554/stream"
export TARGET_FPS=5
export OPTIMIZATION_LEVEL=optimized
export BOT_TOKEN="your_telegram_bot_token"
export CHAT_ID="your_telegram_chat_id"

# Run
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### Option B: ONVIF Camera

```bash
# Set configuration
export DEPLOYMENT_MODE=production
export ONVIF_ENABLED=true
export ONVIF_HOST="192.168.1.100"
export ONVIF_USERNAME="admin"
export ONVIF_PASSWORD="password"
export TARGET_FPS=5

# Run
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### Option C: Multi-Camera Setup

```bash
# Edit app.py to add cameras:
# camera_pool.add_camera(
#     "entrance",
#     rtsp_url="rtsp://192.168.1.100/stream"
# )
# camera_pool.add_camera(
#     "warehouse",
#     onvif_host="192.168.1.101",
#     onvif_username="admin",
#     onvif_password="password"
# )
```

**Check camera health**: GET http://localhost:8000/api/cameras/list

---

### Mode 3: DEVELOPMENT (For Testing) 🔧

**Best for**: Local testing, debugging, development

```bash
# Set configuration (or use defaults)
export DEPLOYMENT_MODE=development

# Run with reload for development
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Use webcam:
# POST http://localhost:8000/api/use-webcam

# Or upload video:
# POST http://localhost:8000/api/upload-video (with file)
```

## 📊 Performance Optimization

### Optimize for 1 FPS (Very Low Bandwidth)

```bash
export TARGET_FPS=1
export OPTIMIZATION_LEVEL=optimized
export RTSP_URL="your_camera_url"
```

### Optimize for Real-Time (15+ FPS)

```bash
export TARGET_FPS=15
export OPTIMIZATION_LEVEL=high
export USE_GPU=true
```

### Optimize for Balance (5-10 FPS)

```bash
export TARGET_FPS=10
export OPTIMIZATION_LEVEL=balanced
```

## 🐳 Docker Deployment (Production)

### Quick Start

```bash
# Build and run
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f safetyvision

# Stop
docker-compose -f docker-compose.prod.yml down
```

### With Environment File

```bash
# Create .env for docker-compose
cat > .env.docker << EOF
DEPLOYMENT_MODE=production
RTSP_URL=rtsp://camera/stream
BOT_TOKEN=your_token
CHAT_ID=your_chat_id
TARGET_FPS=5
OPTIMIZATION_LEVEL=optimized
CPU_LIMIT=2
MEMORY_LIMIT=2G
EOF

# Run with config
docker-compose -f docker-compose.prod.yml up -d --env-file .env.docker
```

## 🧪 Testing

### Test Telegram Alerts

```bash
# Check if Telegram is configured
curl http://localhost:8000/api/health

# Test Telegram
curl -X POST http://localhost:8000/api/telegram-test
```

### Test Detection

```bash
# Detect in image
curl -F "file=@test.jpg" "http://localhost:8000/api/detect-image?mode=all"

# Response includes detections and statistics
```

### Test API Endpoints

```bash
# Get configuration
curl http://localhost:8000/api/config

# Get system stats
curl http://localhost:8000/api/system/stats

# List demo samples (demo mode)
curl http://localhost:8000/api/demo/samples

# List cameras (production mode)
curl http://localhost:8000/api/cameras/list
```

## 📱 Web Interface

Open your browser and navigate to:

```
http://localhost:8000
```

Features:
- ✅ Real-time video stream
- ✅ Live detections with boxes
- ✅ Statistics dashboard
- ✅ Alert notifications
- ✅ Mode selector (people, fire, PPE, spill, fall)
- ✅ FPS and performance metrics

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"No camera found"** | Check RTSP URL format, test with ffplay |
| **Low FPS** | Reduce `TARGET_FPS` or increase `OPTIMIZATION_LEVEL=optimized` |
| **High CPU** | Lower `TARGET_FPS` or use GPU with `USE_GPU=true` |
| **Telegram not working** | Verify `BOT_TOKEN` and `CHAT_ID`, test with `/api/telegram-test` |
| **Models not loading** | Check `models/` directory has `.pt` files |
| **Memory issues** | Reduce frame resolution, lower `TARGET_FPS` |

## 📈 Monitoring Performance

Check real-time metrics at:

```
GET http://localhost:8000/api/system/stats
```

Response includes:
- FPS (actual vs target)
- Inference time
- Bottleneck analysis
- Quality level (high/balanced/optimized)

## 🔐 Security

1. **Never commit `.env` file** with secrets
2. **Use strong Telegram bot token** - rotate periodically
3. **RTSP credentials** - use environment variables only
4. **Firewall** - restrict port 8000 to trusted networks
5. **HTTPS** - use reverse proxy (nginx) in production

## 📞 Support

For issues or questions:
1. Check [ARCHITECTURE.md](ARCHITECTURE.md) for detailed guide
2. Review logs: `docker-compose logs` or check terminal output
3. Test individual endpoints with curl

## 🎬 Example Workflow

### For Competition Demo

```bash
# 1. Prepare samples
mkdir samples
cp fire_demo.mp4 samples/
cp fall_demo.mp4 samples/
cp ppe_demo.mp4 samples/

# 2. Pre-generate results (make it instant!)
python generate_precomputed.py

# 3. Start in demo mode
export DEPLOYMENT_MODE=demo
export TARGET_FPS=15
uvicorn app:app --host 0.0.0.0 --port 8000

# 4. Open browser and showcase!
# http://localhost:8000
```

### For Surveillance Deployment

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with camera URLs and Telegram settings

# 2. Start in production mode
export DEPLOYMENT_MODE=production
uvicorn app:app --host 0.0.0.0 --port 8000

# 3. Monitor via web interface
# http://your-server:8000

# 4. Receive Telegram alerts when issues detected
```

## ✨ Next Steps

1. ✅ Read [ARCHITECTURE.md](ARCHITECTURE.md) for detailed info
2. ✅ Set up [.env](.env.example) configuration
3. ✅ Test with demo samples or webcam
4. ✅ Configure real cameras for production
5. ✅ Deploy with Docker Compose
6. ✅ Set up monitoring and alerting

Enjoy SafetyVision AI! 🎉
