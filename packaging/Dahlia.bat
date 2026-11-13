@echo off
rem Dahlia portable launcher (no install, no registry).
rem This window stays open while Dahlia runs and keeps a log,
rem so if anything goes wrong the cause stays on screen.
setlocal
cd /d %~dp0
set DAHLIA_PYTHON=%~dp0python\python.exe
set DAHLIA_WORKERS_DIR=%~dp0workers
if not exist "%~dp0studio.exe" (
  echo [ERROR] studio.exe not found in this folder.
  pause
  exit /b 1
)
python\python.exe check_env.py
if not exist "%ProgramFiles(x86)%\Microsoft\EdgeWebView\Application" (
  echo [WARN] WebView2 runtime folder was not found.
  echo Dahlia needs the Microsoft Edge WebView2 runtime.
  echo It is built into Windows 11 and recent Windows 10.
)
echo.
echo Starting Dahlia... (close the Dahlia window to quit)
echo Log file: dahlia-run.log
"%~dp0studio.exe" > dahlia-run.log 2>&1
echo.
echo Dahlia exited. Code: %ERRORLEVEL%
echo ---- dahlia-run.log ----
type dahlia-run.log
echo ------------------------
pause
endlocal
