@echo off
chcp 65001 >nul
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    set "PYTHON_CMD=python"
)

%PYTHON_CMD% -m PyInstaller --version >nul 2>nul
if not %errorlevel%==0 (
    echo 正在安装 PyInstaller……
    %PYTHON_CMD% -m pip install pyinstaller
    if not %errorlevel%==0 (
        echo PyInstaller 安装失败。
        pause
        exit /b 1
    )
)

echo 正在构建单文件 EXE……
%PYTHON_CMD% -m PyInstaller --noconfirm --clean Infrared3DViewer.spec
if not %errorlevel%==0 (
    echo 打包失败。
    pause
    exit /b 1
)

if not exist "release" mkdir "release"
copy /y "dist\Infrared3DViewer.exe" "release\Infrared3DViewer_v1.7.exe" >nul
copy /y "使用说明.txt" "release\使用说明.txt" >nul

powershell -NoProfile -Command "Compress-Archive -LiteralPath 'release\Infrared3DViewer_v1.7.exe','release\使用说明.txt' -DestinationPath 'release\Infrared3DViewer_v1.7_Windows_x64.zip' -Force"
copy /y "release\Infrared3DViewer_v1.7_Windows_x64.zip" "release\Infrared3DViewer_Windows_x64.zip" >nul

echo.
echo 构建完成：
echo   release\Infrared3DViewer_v1.7.exe
echo   release\Infrared3DViewer_v1.7_Windows_x64.zip
pause
