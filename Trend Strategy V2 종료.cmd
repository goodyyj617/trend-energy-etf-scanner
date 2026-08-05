@echo off
setlocal
set "ROOT=%~dp0"
for %%I in ("%ROOT%.") do set "ROOT=%%~fI"
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" exit /b 1
"%PYTHON%" "%ROOT%\scripts\trend_v2_windows_launcher.py" stop --root "%ROOT%"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" if not "%TREND_V2_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
