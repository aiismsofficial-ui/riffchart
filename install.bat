@echo off
rem RiffChart installer: double-click this file. It runs install.ps1 (PowerShell) from this folder.
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
