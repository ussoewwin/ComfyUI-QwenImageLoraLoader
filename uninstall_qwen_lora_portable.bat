@echo off
echo ComfyUI-QwenImageLoraLoader Uninstallation Script (Portable Version)
echo ==================================================================
echo This script is for portable ComfyUI installations with embedded Python.
echo For regular installations, use uninstall_qwen_lora.bat instead.
echo.

REM Assuming the script is located at ComfyUI/custom_nodes/ComfyUI-QwenImageLoraLoader/
set "SCRIPT_DIR=%~dp0"
for %%i in ("%SCRIPT_DIR%..\..") do set "COMFYUI_ROOT=%%~fi"

REM For portable ComfyUI, python_embeded is usually in the parent directory of ComfyUI
for %%i in ("%COMFYUI_ROOT%\..") do set "COMFYUI_PARENT=%%~fi"

set "CUSTOM_NODES_PATH=%COMFYUI_ROOT%\custom_nodes"
set "NUNCHAKU_PATH=%CUSTOM_NODES_PATH%\ComfyUI-nunchaku"

if not exist "%CUSTOM_NODES_PATH%" (
    echo [ERROR] custom_nodes not found under: %CUSTOM_NODES_PATH%
    pause
    exit /b 1
)

if not exist "%NUNCHAKU_PATH%" (
    echo [ERROR] ComfyUI-nunchaku not found under: %NUNCHAKU_PATH%
    pause
    exit /b 1
)

REM Check for embedded Python in multiple possible locations
if exist "%COMFYUI_ROOT%\python_embeded\python.exe" (
    set "PYTHON_CMD=%COMFYUI_ROOT%\python_embeded\python.exe"
) else if exist "%COMFYUI_PARENT%\python_embeded\python.exe" (
    set "PYTHON_CMD=%COMFYUI_PARENT%\python_embeded\python.exe"
) else (
    echo [ERROR] Embedded Python not found
    echo Searched locations:
    echo - %COMFYUI_ROOT%\python_embeded\python.exe
    echo - %COMFYUI_PARENT%\python_embeded\python.exe
    pause
    exit /b 1
)

echo Found embedded Python at: %PYTHON_CMD%
echo.

echo Checking for backups...

REM Try new-style backup first
if exist "%NUNCHAKU_PATH%\__init__.py.qwen_image_backup" (
    copy "%NUNCHAKU_PATH%\__init__.py.qwen_image_backup" "%NUNCHAKU_PATH%\__init__.py" >nul
    echo Restored from backup: __init__.py.qwen_image_backup
    echo [OK] Qwen Image LoRA Loader integration removed.
    pause
    exit /b 0
)

REM No backup found - remove integration blocks using Python script
echo No backup found. Removing integration blocks...
"%PYTHON_CMD%" "%SCRIPT_DIR%remove_integration.py"

echo.
echo [OK] Qwen Image LoRA Loader integration removed.
pause
exit /b 0

