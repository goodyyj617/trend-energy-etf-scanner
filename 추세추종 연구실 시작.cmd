@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [setup] First run: creating the Python environment and installing packages. This takes a few minutes.
  py -3.13 -m venv .venv || python -m venv .venv
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt -r requirements-lab.txt
)
rem The update may rewrite this file. cmd reads a ( ... ) block whole before running it,
rem so the update and the start below sit in one block that ends with exit.
(
  ".venv\Scripts\python.exe" scripts\lab_update.py
  echo Starting the research lab. Your browser opens automatically.
  echo To stop it, close this window or press Ctrl+C.
  ".venv\Scripts\python.exe" -m streamlit run lab\app.py --server.headless false
  pause
  exit /b
)
