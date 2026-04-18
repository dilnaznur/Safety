@echo off
setlocal
cd /d %~dp0

echo ========================================
echo SafetyVision Product Installer
echo ========================================

echo [1/4] Checking Python...
where py >nul 2>nul
if %errorlevel% neq 0 (
  echo Python launcher not found. Install Python 3.10+ and retry.
  pause
  exit /b 1
)

echo [2/4] Creating virtual environment...
if not exist venv (
  py -3 -m venv venv
  if %errorlevel% neq 0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
  )
)

echo [3/4] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
  echo Dependency installation failed.
  pause
  exit /b 1
)

echo [4/4] Validating models...
if not exist models\people_best.pt echo Missing models\people_best.pt
if not exist models\ppe_best.pt echo Missing models\ppe_best.pt
if not exist models\fire_best.pt echo Missing models\fire_best.pt
if not exist models\spill_best.pt echo Missing models\spill_best.pt
if not exist models\fall_best.pt echo Missing models\fall_best.pt

echo.
echo Installation complete.
echo Start system: start.bat
echo Open dashboard: http://localhost:5000
pause
