@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Run the asset-allocation lab launcher once first to set up Python.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\aa_link.py register
echo.
echo Done. Add a Chrome bookmark with this URL:  aalab://scorecard
echo (aalab://research opens the strategy page, aalab://notes the findings page.)
pause
