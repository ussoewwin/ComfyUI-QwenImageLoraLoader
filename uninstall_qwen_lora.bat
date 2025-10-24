@echo off
echo ComfyUI-QwenImageLoraLoader Uninstallation Script
echo =================================================

REM ComfyUI\custom_nodes を検索
for /f "delims=" %%i in ('dir /s /b /ad "*ComfyUI\custom_nodes" 2^>nul') do (
    set "CUSTOM_NODES_PATH=%%i"
    goto :found
)

:found
if "%CUSTOM_NODES_PATH%"=="" (
    echo ERROR: ComfyUI\custom_nodes not found.
    echo Please ensure ComfyUI is installed.
    pause
    exit /b 1
)

REM ComfyUIのルートパスを取得
for %%i in ("%CUSTOM_NODES_PATH%\..") do set "COMFYUI_ROOT=%%~fi"
echo Found ComfyUI at: %COMFYUI_ROOT%

set "NUNCHAKU_PATH=%COMFYUI_ROOT%\custom_nodes\ComfyUI-nunchaku"

echo.
echo Checking ComfyUI-nunchaku installation...
if not exist "%NUNCHAKU_PATH%" (
    echo ERROR: ComfyUI-nunchaku not found at %NUNCHAKU_PATH%
    pause
    exit /b 1
)

echo Restoring original __init__.py...
if exist "%NUNCHAKU_PATH%\__init__.py.backup" (
    copy "%NUNCHAKU_PATH%\__init__.py.backup" "%NUNCHAKU_PATH%\__init__.py"
    echo Original __init__.py restored from backup.
    echo Backup file kept: __init__.py.backup
) else (
    echo WARNING: No backup found. Manual restoration may be required.
    echo Please check if the integration code was added manually.
)

echo.
echo Uninstallation completed!
echo Please restart ComfyUI.
echo.
pause
