@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto error
)
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
if errorlevel 1 goto error
echo Setup complete. Run run.cmd or test.cmd.
pause
exit /b 0
:error
echo Setup failed. Install Python 3.12 or newer and check network access.
pause
exit /b 1
