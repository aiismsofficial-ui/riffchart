@echo off
rem RiffChart uninstaller: double-click this file. It runs uninstall.ps1 (PowerShell) from this folder.
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1" %*
echo.
pause
