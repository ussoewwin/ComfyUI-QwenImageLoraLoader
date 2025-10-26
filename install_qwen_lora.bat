@echo off
echo ComfyUI-QwenImageLoraLoader Installation Script
echo ================================================

REM Assuming the script is located at ComfyUI/custom_nodes/ComfyUI-QwenImageLoraLoader/
set "SCRIPT_DIR=%~dp0"
for %%i in ("%SCRIPT_DIR%..\..") do set "COMFYUI_ROOT=%%~fi"

set "CUSTOM_NODES_PATH=%COMFYUI_ROOT%\custom_nodes"
set "NUNCHAKU_PATH=%CUSTOM_NODES_PATH%\ComfyUI-nunchaku"
set "LORA_LOADER_PATH=%CUSTOM_NODES_PATH%\ComfyUI-QwenImageLoraLoader"

if not exist "%CUSTOM_NODES_PATH%" (
    echo ERROR: custom_nodes folder not found at %CUSTOM_NODES_PATH%
    echo Please run this script from the ComfyUI-QwenImageLoraLoader directory inside custom_nodes.
    echo.
    echo Expected directory structure:
    echo ComfyUI/
    echo   custom_nodes/
    echo     ComfyUI-QwenImageLoraLoader/
    echo       install_qwen_lora.bat
    pause
    exit /b 1
)

echo Found ComfyUI at: %COMFYUI_ROOT%


echo.
echo Checking ComfyUI-nunchaku installation...
if not exist "%NUNCHAKU_PATH%" (
    echo ERROR: ComfyUI-nunchaku not found at %NUNCHAKU_PATH%
    echo Please install ComfyUI-nunchaku first.
    echo Download from: https://github.com/nunchaku-tech/ComfyUI-nunchaku
    pause
    exit /b 1
)

echo Checking ComfyUI-QwenImageLoraLoader installation...
if not exist "%LORA_LOADER_PATH%" (
    echo ERROR: ComfyUI-QwenImageLoraLoader not found at %LORA_LOADER_PATH%
    echo Please install ComfyUI-QwenImageLoraLoader first.
    echo Download from: https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader
    pause
    exit /b 1
)

echo Backing up original __init__.py...
if exist "%NUNCHAKU_PATH%\__init__.py" (
    copy "%NUNCHAKU_PATH%\__init__.py" "%NUNCHAKU_PATH%\__init__.py.backup"
    echo Backup created: __init__.py.backup
)

echo Adding LoRA loader integration code...

REM Check if already installed
findstr /C:"ComfyUI-QwenImageLoraLoader Integration" "%NUNCHAKU_PATH%\__init__.py" >nul 2>&1
if %errorlevel% equ 0 (
    echo Already installed. Integration code already exists in __init__.py
    pause
    exit /b 0
)

REM Append integration code using Python script
python "%LORA_LOADER_PATH%\append_integration.py" "%NUNCHAKU_PATH%\__init__.py"
if errorlevel 1 (
    echo ERROR: Failed to append integration code
    pause
    exit /b 1
)

echo.
echo Installation completed successfully!
echo.
echo ComfyUI Root: %COMFYUI_ROOT%
echo Nunchaku Path: %NUNCHAKU_PATH%
echo LoRA Loader Path: %LORA_LOADER_PATH%
echo.
echo The following nodes have been added:
echo - NunchakuQwenImageLoraLoader
echo - NunchakuQwenImageLoraStack
echo.
echo Please restart ComfyUI to use the new nodes.
echo.
pause
