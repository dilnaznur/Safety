@echo off
setlocal
cd /d %~dp0

if not exist venv\Scripts\activate.bat (
  echo Virtual environment is missing. Run install.bat first.
  pause
  exit /b 1
)

call venv\Scripts\activate.bat
uvicorn server:app --host 0.0.0.0 --port 5000
