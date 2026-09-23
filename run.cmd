@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.cmd first.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8001 or http://127.0.0.1:8001/docs
.venv\Scripts\python.exe main.py
pause
