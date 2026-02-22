@echo off
REM Run Cipher (backend + frontend) when PowerShell script execution is disabled.
REM Uses -ExecutionPolicy Bypass for this run only.
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -NoProfile -File ".\run_drone_full.ps1"
pause
