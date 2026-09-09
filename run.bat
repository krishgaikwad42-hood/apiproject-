@echo off
title Smart Healthcare Diagnosis API Server
cd /d "%~dp0"

echo ======================================================================
echo           SMART HEALTHCARE DIAGNOSIS API (DISEASE PREDICTION)
echo ======================================================================
echo.
echo Starting FastAPI application with Uvicorn...
echo Dashboard URL : http://127.0.0.1:8000/
echo Swagger UI    : http://127.0.0.1:8000/docs
echo Health Check  : http://127.0.0.1:8000/health
echo.
echo Opening interactive dashboard in your browser in 2 seconds...
start "" "http://127.0.0.1:8000/"

echo.
echo Press Ctrl+C in this window to stop the server at any time.
echo.
python main.py
pause
