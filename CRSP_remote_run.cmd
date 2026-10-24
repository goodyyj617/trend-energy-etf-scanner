@echo off
setlocal
cd /d "%~dp0"
rem Run a CRSP study from Google Drive on this computer; no CRSP data is left here afterwards.
if "%~1"=="" (
  echo Usage: CRSP_remote_run.cmd "G:\My Drive\CRSP" [screens^|families] [top]
  exit /b 1
)
echo [update] Getting the latest code.
git pull --ff-only
if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating the Python environment.
  py -3.13 -m venv .venv 2>nul || python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo Python was not found. Install Python 3.11 or newer from python.org ^(tick "Add python.exe to PATH"^) and run this again.
  exit /b 1
)
".venv\Scripts\python.exe" -c "import pandas, pyarrow, psutil, yfinance, yaml" 2>nul
if errorlevel 1 (
  echo [setup] Installing packages. This takes a few minutes; please wait until it finishes.
  ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt -r requirements-lab.txt psutil || exit /b 1
)
set "STUDY=%~2"
if "%STUDY%"=="" set "STUDY=screens"
set "TOP=%~3"
if "%TOP%"=="" set "TOP=3000"
".venv\Scripts\python.exe" scripts\crsp_remote.py --drive "%~1" --study %STUDY% --top %TOP%
pause
