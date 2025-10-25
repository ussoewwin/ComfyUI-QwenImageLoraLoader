@echo off
echo ComfyUI-QwenImageLoraLoader Uninstallation Script
echo =================================================

REM Assuming the script is located at ComfyUI/custom_nodes/ComfyUI-QwenImageLoraLoader/
set "SCRIPT_DIR=%~dp0"
for %%i in ("%SCRIPT_DIR%..\..") do set "COMFYUI_ROOT=%%~fi"

set "CUSTOM_NODES_PATH=%COMFYUI_ROOT%\custom_nodes"
set "NUNCHAKU_PATH=%CUSTOM_NODES_PATH%\ComfyUI-nunchaku"

if not exist "%CUSTOM_NODES_PATH%" (
    echo ERROR: custom_nodes folder not found at %CUSTOM_NODES_PATH%
    echo Please run this script from the ComfyUI-QwenImageLoraLoader directory inside custom_nodes.
    pause
    exit /b 1
)

echo Found ComfyUI at: %COMFYUI_ROOT%

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
