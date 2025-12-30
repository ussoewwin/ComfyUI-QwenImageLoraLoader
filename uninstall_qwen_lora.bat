@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Assuming the script is located at ComfyUI/custom_nodes/ComfyUI-QwenImageLoraLoader/
set "SCRIPT_DIR=%~dp0"
for %%i in ("%SCRIPT_DIR%..\..") do set "COMFYUI_ROOT=%%~fi"

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
where py >nul 2>&1
if not errorlevel 1 (
    call py -3 "%SCRIPT_DIR%remove_integration.py"
)

echo [OK] Qwen Image LoRA Loader integration removed.
pause
exit /b 0
