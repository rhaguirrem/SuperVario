Do it,@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Install the environment first. See README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m supervario gui %*
if errorlevel 1 pause
