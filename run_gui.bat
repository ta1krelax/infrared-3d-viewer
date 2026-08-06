@echo off
chcp 65001 >nul
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    set "PYTHON_CMD=python"
)

%PYTHON_CMD% -c "import numpy, matplotlib, PIL" >nul 2>nul
if not %errorlevel%==0 (
    echo 首次运行，正在安装所需组件……
    %PYTHON_CMD% -m pip install -r requirements.txt
    if not %errorlevel%==0 (
        echo.
        echo 依赖安装失败。请检查 Python 和网络连接。
        pause
        exit /b 1
    )
)

%PYTHON_CMD% app.py
if not %errorlevel%==0 pause
