@echo off
chcp 65001 >nul
title 首都师范大学生存助手 - Web服务

echo ========================================
echo   首都师范大学在校生生存助手 - Web界面
echo ========================================
echo.

cd /d "%~dp0"

echo 正在启动服务 (端口 8080)...
echo.

REM 激活虚拟环境并启动
call .venv\Scripts\activate.bat
python run_web.py

pause
