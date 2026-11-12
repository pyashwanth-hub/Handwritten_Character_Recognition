@echo off
title AI Handwritten Character Recognition - Fullstack Launcher
cd /d "%~dp0"
echo ======================================================================
echo    AI Handwritten Character Recognition - Fullstack Launcher
echo    Starting Frontend (Port 3000) + Backend REST API (Port 5000)
echo ======================================================================
echo.

:: 1. Start Backend in separate window
echo [1/2] Starting Python Flask Deep Learning Backend on port 5000...
start "AI Backend Server (Port 5000)" cmd /k "cd /d "%~dp0backend" && run_backend.bat"

:: 2. Wait 2 seconds for backend to initialize
timeout /t 2 /nobreak >nul

:: 3. Start Frontend and open browser
echo [2/2] Starting Frontend Web Server on port 3000...
start "" timeout /t 2 >nul & start http://127.0.0.1:3000
start "AI Frontend Web App (Port 3000)" cmd /k "cd /d "%~dp0frontend" && run_frontend.bat"

echo.
echo ======================================================================
echo  Both Frontend and Backend are now running!
echo  - Frontend UI:  http://127.0.0.1:3000
echo  - Backend API:  http://127.0.0.1:5000
echo ======================================================================
