# SafetyVision AI - Deployment Checklist

## Pre-Deployment Checklist

### ✅ Code & Dependencies

- [ ] Clone/download repository
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux)
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify models exist in `models/` directory:
  - [ ] `people_best.pt`
  - [ ] `ppe_best.pt`
  - [ ] `fire_best.pt`
  - [ ] `spill_best.pt`
  - [ ] `fall_best.pt`
- [ ] Verify static files in `static/` directory:
  - [ ] `index.html`
  - [ ] `js/i18n.js`
  - [ ] `vendor/` files

### 📋 Configuration (.env file)

Choose your deployment mode and complete relevant section:

#### DEMO MODE (Competition)

- [ ] Copy `.env.example` to `.env`
- [ ] Set `DEPLOYMENT_MODE=demo`
- [ ] Set `TARGET_FPS=15` (smooth UI)
- [ ] Create `samples/` directory
- [ ] Copy demo videos to `samples/`
- [ ] Run: `python generate_precomputed_demo.py --all`
- [ ] Verify: `samples/precomputed/` has JSON files

#### PRODUCTION MODE (Surveillance - RTSP)

- [ ] Copy `.env.example` to `.env`
- [ ] Set `DEPLOYMENT_MODE=production`
- [ ] Set `RTSP_URL=rtsp://user:pass@camera:554/stream`
- [ ] Set `TARGET_FPS=5` (efficient for surveillance)
- [ ] Set `OPTIMIZATION_LEVEL=optimized`
- [ ] Test RTSP connection: `ffplay "rtsp_url"` or `cvlc "rtsp_url"`
- [ ] (Optional) Configure Telegram alerts:
  - [ ] `BOT_TOKEN=` (get from @BotFather)
  - [ ] `CHAT_ID=` (forward message to @userinfobot)
  - [ ] Test: `curl -X POST http://localhost:8000/api/telegram-test`

#### PRODUCTION MODE (Surveillance - ONVIF)

- [ ] Copy `.env.example` to `.env`
- [ ] Set `DEPLOYMENT_MODE=production`
- [ ] Set `ONVIF_ENABLED=true`
- [ ] Set `ONVIF_HOST=192.168.1.100` (camera IP)
- [ ] Set `ONVIF_USERNAME=admin`
- [ ] Set `ONVIF_PASSWORD=password`
- [ ] Install ONVIF: `pip install onvif-zeep`
- [ ] Test connection from app logs

#### DEVELOPMENT MODE (Testing)

- [ ] Copy `.env.example` to `.env`
- [ ] Set `DEPLOYMENT_MODE=development`
- [ ] No special config needed - uses webcam by default

### 🧪 Local Testing

- [ ] Start app: `uvicorn app:app --host 0.0.0.0 --port 8000`
- [ ] Open browser: `http://localhost:8000`
- [ ] Verify UI loads
- [ ] Check console for errors
- [ ] Test detection with sample image: `POST /api/detect-image`
- [ ] Monitor performance: `GET /api/system/stats`
- [ ] (Demo only) Load sample and verify instant results
- [ ] (Surveillance only) Verify camera connection: `GET /api/cameras/list`

### 🐳 Docker Deployment (Optional)

- [ ] Install Docker & Docker Compose
- [ ] Review `docker-compose.prod.yml`
- [ ] Update environment variables in docker-compose file or .env
- [ ] Build: `docker-compose -f docker-compose.prod.yml build`
- [ ] Start: `docker-compose -f docker-compose.prod.yml up -d`
- [ ] Check health: `docker-compose -f docker-compose.prod.yml logs -f`
- [ ] Access: `http://localhost:8000`

### 🚨 Alerts & Monitoring

- [ ] (Telegram) Test bot works: `curl -X POST http://localhost:8000/api/telegram-test`
- [ ] (Telegram) Verify chat ID receives test message
- [ ] Configure alert severity threshold
- [ ] Test alert creation with test detection
- [ ] Verify logs directory: `logs/alerts.log`
- [ ] Set up log rotation (optional)

### 📊 Performance Tuning

- [ ] Monitor FPS via `/api/system/stats`
- [ ] Check bottleneck (inference, read, encode)
- [ ] Adjust quality if needed:
  - Low FPS? → Reduce `TARGET_FPS` or increase `OPTIMIZATION_LEVEL`
  - High CPU? → Use `OPTIMIZATION_LEVEL=optimized` or GPU
  - Lag? → Lower inference resolution

### 🔐 Security

- [ ] **Never** commit `.env` file with secrets
- [ ] Use environment variables for all credentials
- [ ] Restrict port 8000 to trusted networks only
- [ ] (Production) Use HTTPS with reverse proxy (nginx/traefik)
- [ ] (Production) Enable firewall rules
- [ ] (Production) Rotate Telegram bot token periodically
- [ ] (Production) Secure RTSP URLs (use strong passwords)

### 📝 Documentation

- [ ] Review [ARCHITECTURE.md](ARCHITECTURE.md) for detailed info
- [ ] Keep [QUICK_START.md](QUICK_START.md) for reference
- [ ] Document any custom configuration
- [ ] Document camera setup and credentials (secure location)

### ✨ Final Verification

- [ ] Health check: `curl http://localhost:8000/health`
- [ ] Models loaded: Check health response
- [ ] Telegram working: `curl -X POST http://localhost:8000/api/telegram-test`
- [ ] Performance acceptable: `curl http://localhost:8000/api/system/stats`
- [ ] UI responsive: Open in browser and interact
- [ ] Detection working: Test with sample image/video
- [ ] No errors in logs

## Deployment Environments

### Local Development

```bash
export DEPLOYMENT_MODE=development
uvicorn app:app --reload
# Access: http://localhost:8000
```

### Local Testing (All Modes)

```bash
# Demo mode
export DEPLOYMENT_MODE=demo
python generate_precomputed_demo.py --all
uvicorn app:app

# Surveillance mode with test camera
export DEPLOYMENT_MODE=production
export RTSP_URL=rtsp://test_camera
uvicorn app:app
```

### Production Linux Server

```bash
# Copy files
scp -r guard/ user@server:/opt/

# SSH into server
ssh user@server

# Navigate to directory
cd /opt/guard

# Create systemd service
sudo tee /etc/systemd/system/safetyvision.service > /dev/null <<EOF
[Unit]
Description=SafetyVision AI Service
After=network.target

[Service]
Type=simple
User=safetyvision
WorkingDirectory=/opt/guard
ExecStart=/opt/guard/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable safetyvision
sudo systemctl start safetyvision
sudo systemctl status safetyvision
```

### Docker Container

```bash
# Production Docker
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs -f

# Stop
docker-compose -f docker-compose.prod.yml down
```

### Kubernetes (Advanced)

For large-scale deployments, create Kubernetes manifests based on docker-compose.

## Maintenance

### Regular Tasks

- [ ] Monitor system resources (CPU, memory)
- [ ] Check for new model updates
- [ ] Rotate Telegram bot tokens (quarterly)
- [ ] Review and archive logs (weekly)
- [ ] Update dependencies (monthly)
  ```bash
  pip list --outdated
  pip install --upgrade package_name
  ```
- [ ] Test Telegram alerts (weekly)

### Troubleshooting

| Issue                | Check                                                                 |
| -------------------- | --------------------------------------------------------------------- |
| App won't start      | Check logs, verify Python version (3.9+), check port 8000             |
| No detections        | Verify models loaded, check camera connection, test with sample image |
| High CPU             | Lower FPS, reduce resolution, use GPU, check inference time           |
| Memory leak          | Monitor with `top`, restart service periodically                      |
| Telegram not working | Verify token and chat ID, test endpoint                               |
| Camera disconnects   | Check network, verify RTSP URL, increase timeout                      |

### Rollback Procedure

```bash
# If deployment fails:
1. Stop service
   systemctl stop safetyvision
   # or
   docker-compose -f docker-compose.prod.yml down

2. Revert to previous version
   cd /opt/guard
   git checkout previous_commit

3. Restart
   systemctl start safetyvision
   # or
   docker-compose -f docker-compose.prod.yml up -d
```

## Monitoring & Observability

### Metrics Endpoints

```
GET /health                      → System health
GET /api/stats                   → Detection statistics
GET /api/config                  → Current configuration
GET /api/system/stats            → Performance metrics
GET /api/cameras/list            → Connected cameras
GET /api/demo/progress           → Demo playback progress
```

### Log Files

- `logs/alerts.log` - All detection alerts
- Console output - Real-time processing info
- Docker logs - `docker-compose logs -f`

### Performance Monitoring

1. Open web UI: `http://localhost:8000`
2. Watch FPS and statistics
3. Check `/api/system/stats` for bottleneck analysis
4. Adjust configuration if needed

## Success Criteria

✅ Your deployment is successful when:

- [ ] Web interface loads without errors
- [ ] Models detected and loaded (health check shows "loaded")
- [ ] Video stream displays on main page
- [ ] Detections appear with bounding boxes
- [ ] Statistics update in real-time
- [ ] Alert notifications received (if Telegram configured)
- [ ] FPS is stable at target rate
- [ ] CPU usage is acceptable
- [ ] No errors in logs

## Support Resources

- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Quick Start**: [QUICK_START.md](QUICK_START.md)
- **Configuration**: [.env.example](.env.example)
- **Logs**: Check console output or `logs/alerts.log`

---

**Ready to deploy?** Start with the Quick Start guide and follow this checklist step by step!
