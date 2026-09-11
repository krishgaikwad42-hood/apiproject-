@echo off
title Smart Healthcare Diagnosis API — v1.0.0
cd /d "%~dp0"

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║         SMART HEALTHCARE DIAGNOSIS API  v1.0.0              ║
echo  ║         ML-Powered Clinical Risk Prediction System          ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.
echo  [*] Starting FastAPI server with Uvicorn...
echo  [*] Dashboard  ^>  http://127.0.0.1:8000/
echo  [*] Swagger UI ^>  http://127.0.0.1:8000/docs
echo  [*] ReDoc      ^>  http://127.0.0.1:8000/redoc
echo  [*] Health     ^>  http://127.0.0.1:8000/health
echo  [*] Metrics    ^>  http://127.0.0.1:8000/metrics
echo.
echo  [!] Press Ctrl+C to stop the server.
echo.

REM Start the server in background, wait 3 seconds for it to be ready, then open browser
start "" python main.py
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8000/"

REM Now wait for the Python process (keeps window alive)
python main.py
pause
