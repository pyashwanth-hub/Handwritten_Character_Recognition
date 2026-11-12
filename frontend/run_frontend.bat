@echo off
title AI Handwritten Character Recognition - Frontend Web App
cd /d "%~dp0"
echo ========================================================
echo   Starting Standalone Frontend Web Server
echo   Port: 3000 (http://127.0.0.1:3000)
echo   Connects to Backend REST API at http://127.0.0.1:5000
echo ========================================================
echo.

echo Opening Frontend in browser in 2 seconds...
start "" timeout /t 2 >nul & start http://127.0.0.1:3000
python -m http.server 3000
pause
