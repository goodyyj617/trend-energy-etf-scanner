@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Run the research lab launcher once first to set up Python.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\lab_link.py register
echo.
echo Done. Add a Chrome bookmark with this URL:  trendlab://saved
echo (trendlab://research opens the research page.)
pause
