@echo off
chcp 65001 >nul
setlocal
set "ROOT=%~dp0"
if not defined TREND_V2_STORE set "TREND_V2_STORE=%ROOT%.trend_v2_store"
if not defined TREND_V2_PYTHON set "TREND_V2_PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%TREND_V2_PYTHON%" (
  if not exist "%TREND_V2_STORE%\launcher" mkdir "%TREND_V2_STORE%\launcher"
  echo [오류] 저장소 가상환경을 찾을 수 없습니다: %TREND_V2_PYTHON%
  echo [런처 오류] 저장소 가상환경을 찾을 수 없습니다: %TREND_V2_PYTHON%>>"%TREND_V2_STORE%\launcher\launcher.log"
  echo 상세 로그: %TREND_V2_STORE%\launcher\launcher.log
  if not "%TREND_V2_NO_PAUSE%"=="1" pause
  exit /b 1
)
"%TREND_V2_PYTHON%" "%ROOT%scripts\trend_v2_windows_launcher.py" stop --root "%ROOT%"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" if not "%TREND_V2_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
