@echo off
title AI Handwritten Character Recognition System
echo ========================================================
echo   AI Handwritten Character Recognition Web App
echo   Powered by Python, Flask, CNN and Deep Learning
echo ========================================================
echo.

:: Check python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ and make sure to check "Add Python to PATH".
    pause
    exit /b 1
)

echo Starting Flask Application Server...
echo Server URL: http://127.0.0.1:5000
echo.
echo Opening browser in 3 seconds...
start "" timeout /t 3 >nul & start http://127.0.0.1:5000
python app.py

pause
