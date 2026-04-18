# SafetyVision Product (Client Install)

## 1) Prepare config

Edit `config.yaml` and set real RTSP URLs:

- `rtsp://admin:password@192.168.1.100:554/stream1`

Set `enabled: true` only for cameras that should be monitored.

## 2) Install

Run:

- `install.bat`

## 3) Start system

Run:

- `start.bat`

Open dashboard:

- `http://localhost:5000`

## 4) Stop system

Run:

- `stop.bat`

## Notes

- Works fully offline on local network.
- Default processing rate is 1 frame per second per camera.
- Alerts include PPE violations, fire/smoke, spills, and falls.
