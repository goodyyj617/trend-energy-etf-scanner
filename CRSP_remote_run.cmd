@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [setup] First run: creating the Python environment and installing packages.
  py -3.13 -m venv .venv || python -m venv .venv
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt -r requirements-lab.txt psutil
)
if "%~1"=="" (
  echo Usage: CRSP_remote_run.cmd "G:\My Drive\CRSP" [screens^|families] [top]
  exit /b 1
)
set "STUDY=%~2"
if "%STUDY%"=="" set "STUDY=screens"
set "TOP=%~3"
if "%TOP%"=="" set "TOP=3000"
".venv\Scripts\python.exe" scripts\crsp_remote.py --drive "%~1" --study %STUDY% --top %TOP%
pause
