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

REM Create temporary file with UTF-8 encoding
set "TEMP_FILE=%TEMP%\qwen_lora_integration.txt"
(
echo.
echo # ComfyUI-QwenImageLoraLoader Integration
echo try:
echo     # Import from the independent ComfyUI-QwenImageLoraLoader
echo     import sys
echo     import os
echo     qwen_lora_path = os.path.join^(os.path.dirname^(__file__^), "..", "ComfyUI-QwenImageLoraLoader"^)
echo     if qwen_lora_path not in sys.path:
echo         sys.path.insert^(0, qwen_lora_path^)
echo     
echo     # Import directly from the file path
echo     import importlib.util
echo     spec = importlib.util.spec_from_file_location^("qwenimage", os.path.join^(qwen_lora_path, "nodes", "lora", "qwenimage.py"^)^)
echo     qwenimage_module = importlib.util.module_from_spec^(spec^)
echo     spec.loader.exec_module^(qwenimage_module^)
echo     
echo     NunchakuQwenImageLoraLoader = qwenimage_module.NunchakuQwenImageLoraLoader
echo     NunchakuQwenImageLoraStack = qwenimage_module.NunchakuQwenImageLoraStack
echo.
echo     NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraLoader"] = NunchakuQwenImageLoraLoader
echo     NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraStack"] = NunchakuQwenImageLoraStack
echo     logger.info^("Successfully imported Qwen Image LoRA loaders from ComfyUI-QwenImageLoraLoader"^)
echo except ImportError:
echo     logger.exception^("Nodes `NunchakuQwenImageLoraLoader` and `NunchakuQwenImageLoraStack` import failed:"^)
) > "%TEMP_FILE%"

REM Append UTF-8 encoded content to __init__.py using PowerShell
powershell -Command "Get-Content '%TEMP_FILE%' -Encoding UTF8 | Add-Content '%NUNCHAKU_PATH%\__init__.py' -Encoding UTF8"

REM Clean up temporary file
del "%TEMP_FILE%"

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
