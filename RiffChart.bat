@echo off
rem Starts RiffChart and opens it in your browser. Close this window to stop RiffChart.
title RiffChart (close this window to stop it)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo RiffChart isn't installed in this folder yet: double-click install.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m riffchart.ui.server %*
if errorlevel 1 pause
