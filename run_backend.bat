@echo off
title AI Handwritten Character Recognition - Backend REST API
cd /d "%~dp0"
echo ========================================================
echo   Starting Python Flask Backend REST API Server
echo   Port: 5000 (http://127.0.0.1:5000)
echo   CORS Enabled
echo ========================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please ensure Python is in your PATH.
    pause
    exit /b 1
)

python app.py
pause
