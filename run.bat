@echo off
cd /d "%~dp0"
echo Installing dependencies (first run only)...
py -m pip install -q -r requirements.txt
echo.
echo Starting AI Storyteller at http://localhost:8000  (Ctrl+C to stop)
py backend\main.py
