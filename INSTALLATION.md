# ComfyUI-QwenImageLoraLoader Installation Guide

## Overview

ComfyUI-QwenImageLoraLoader is a LoRA loader for Nunchaku Qwen Image models. This guide provides detailed installation instructions for all platforms.

## Prerequisites

### 1. Required Software
- **ComfyUI** (official version)
- **ComfyUI-nunchaku** (official version)
- **Python 3.10-3.13**

### 2. Pre-installation
Follow these steps in order:

1. Install **ComfyUI**
2. Install **ComfyUI-nunchaku**
   ```bash
   git clone https://github.com/nunchaku-tech/ComfyUI-nunchaku.git ComfyUI/custom_nodes/ComfyUI-nunchaku
   ```
3. Install **ComfyUI-QwenImageLoraLoader**
   ```bash
   git clone https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader.git ComfyUI/custom_nodes/ComfyUI-QwenImageLoraLoader
   ```

## Automated Installation (Windows Only)

> **Note**: The automated installation batch files (`.bat`) are **Windows-only**. For macOS and Linux users, please use the [Manual Installation](#manual-installation) method below.

### 1. Choose the Appropriate Installation Script

#### For ComfyUI Installations with Global Python Environment (Windows)
- **Script**: `install_qwen_lora.bat`
- **Platform**: Windows only
- **Python Environment**: Uses global Python environment (system-installed Python)
- **Requirements**: Python must be installed and accessible from command line (`python` command)
- **Usage**: Double-click `install_qwen_lora.bat`
- **When to use**: ComfyUI installations where Python is installed globally on the system

#### For Portable ComfyUI Installations with Embedded Python (Windows)
- **Script**: `install_qwen_lora_portable.bat`
- **Platform**: Windows only
- **Python Environment**: Uses embedded Python (`python_embeded` folder)
- **Requirements**: ComfyUI installation with `python_embeded` folder containing `python.exe`
- **Usage**: Double-click `install_qwen_lora_portable.bat`
- **When to use**: Portable ComfyUI installations that include embedded Python
- **Automatic Detection**: Script automatically searches for `python_embeded` in multiple locations:
  - `[ComfyUIFolder]/ComfyUI/python_embeded/python.exe`
  - `[ComfyUIFolder]/python_embeded/python.exe` (most common)
- **Folder Name Independence**: Works with any ComfyUI folder name (not limited to "ComfyUI")

### 2. Installation Steps
1. Double-click the appropriate `.bat` file
2. The script will automatically:
   - Detect your ComfyUI installation path
   - Back up the original `__init__.py` file (creates `__init__.py.backup`)
   - Add the integration code to ComfyUI-nunchaku's `__init__.py`
   - Verify all required files exist
3. Restart ComfyUI

### 3. Batch File Features
- **Auto-detection**: Automatically detects ComfyUI location using relative path calculation
- **Backup**: Automatically backs up original files before modification
- **Error checking**: Verifies required files exist and provides clear error messages
- **Easy operation**: Double-click to run
- **Portable Support**: `install_qwen_lora_portable.bat` includes intelligent embedded Python detection

### 4. Uninstallation (Windows)
1. Double-click `uninstall_qwen_lora.bat` to run
2. Original `__init__.py` will be restored from backup
3. Restart ComfyUI

## Manual Installation

### 1. Edit ComfyUI-nunchaku's `__init__.py`

Add the following code to the end of `ComfyUI/custom_nodes/ComfyUI-nunchaku/__init__.py`:

```python
# ComfyUI-QwenImageLoraLoader Integration
try:
    # Import from the independent ComfyUI-QwenImageLoraLoader
    import sys
    import os
    qwen_lora_path = os.path.join(os.path.dirname(__file__), "..", "ComfyUI-QwenImageLoraLoader")
    if qwen_lora_path not in sys.path:
        sys.path.insert(0, qwen_lora_path)
    
    # Import directly from the file path
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwenimage", os.path.join(qwen_lora_path, "nodes", "lora", "qwenimage.py"))
    qwenimage_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwenimage_module)
    
    NunchakuQwenImageLoraLoader = qwenimage_module.NunchakuQwenImageLoraLoader
    NunchakuQwenImageLoraStack = qwenimage_module.NunchakuQwenImageLoraStack

    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraLoader"] = NunchakuQwenImageLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraStack"] = NunchakuQwenImageLoraStack
    logger.info("Successfully imported Qwen Image LoRA loaders from ComfyUI-QwenImageLoraLoader")
except ImportError:
    logger.exception("Nodes `NunchakuQwenImageLoraLoader` and `NunchakuQwenImageLoraStack` import failed:")
```

### 2. Why This Modification is Required

- The independent LoRA loader exists as a separate custom node
- The official ComfyUI-nunchaku plugin needs to reference the independent LoRA loader
- Standard `from ... import ...` fails due to different package structures
- `importlib.util` allows direct module loading from file paths

### 3. Restart ComfyUI

## Usage

### 1. Available Nodes
- **NunchakuQwenImageLoraLoader**: Single LoRA loader
- **NunchakuQwenImageLoraStack**: Multiple LoRA stacker (dynamic UI)

### 2. Basic Usage
1. Create a workflow in ComfyUI
2. Load model with `Nunchaku Qwen Image DiT Loader`
3. Add `NunchakuQwenImageLoraLoader` or `NunchakuQwenImageLoraStack`
4. Set LoRA file and strength
5. Run

## Troubleshooting

### 1. Common Issues

#### Nodes not appearing
- Restart ComfyUI
- Re-run the appropriate installation script
- Check ComfyUI console for error messages

#### Import errors
- Verify ComfyUI-nunchaku is correctly installed
- Verify ComfyUI-QwenImageLoraLoader is in the correct location (`custom_nodes/ComfyUI-QwenImageLoraLoader`)
- Check that the integration code was added to `__init__.py`

#### Backup file not found
- If you manually edited `__init__.py`, restore the original content
- Re-run the installation script to create a proper backup

#### Embedded Python not found (Portable ComfyUI)
- Verify that `python_embeded` folder exists in your ComfyUI installation
- Check the error message for searched locations
- If using portable ComfyUI but don't have `python_embeded`, use `install_qwen_lora.bat` instead

### 2. Check Logs
Verify the following message in ComfyUI console:
- `Successfully imported Qwen Image LoRA loaders from ComfyUI-QwenImageLoraLoader`
- If error messages appear, check dependencies and file locations

## Uninstallation

### 1. Automated Uninstallation (Windows Only)
Run `uninstall_qwen_lora.bat`

### 2. Manual Uninstallation (All Platforms)
1. Remove the added code from `ComfyUI/custom_nodes/ComfyUI-nunchaku/__init__.py`
2. Restart ComfyUI
3. Optionally, delete the `ComfyUI-QwenImageLoraLoader` folder from `custom_nodes`

## Cross-Platform Notes

- **Windows**: Use automated batch file installers (`.bat`) for easiest installation
- **macOS/Linux**: Use manual installation method by editing `__init__.py` directly
- **All Platforms**: Manual installation method works universally

## Support

If you encounter issues, please report with the following information:
- **Operating System**: Windows/macOS/Linux
- **Installation Method**: Automated (which script) or Manual
- **Error Messages**: Full error text from ComfyUI console
- **ComfyUI Version**: Check Help > About in ComfyUI
- **Python Version**: Output of `python --version`
- **Log File Contents**: Relevant portions of ComfyUI log

## Special Thanks

- **Issue #10 (Portable ComfyUI Support)**: Special thanks to [@vvhitevvizard](https://github.com/vvhitevvizard) for suggesting the portable ComfyUI installation script feature. This crucial suggestion made it possible for portable ComfyUI users (with embedded Python) to easily install this LoRA loader without manual Python environment configuration.

## License

This project is licensed under the MIT License.
