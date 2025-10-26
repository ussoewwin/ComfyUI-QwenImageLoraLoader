# ComfyUI-QwenImageLoraLoader Installation Guide

## Overview

ComfyUI-QwenImageLoraLoader is a LoRA loader for Nunchaku Qwen Image models. This guide provides installation instructions.

## Prerequisites

### 1. Required Software
- **ComfyUI** (official version)
- **ComfyUI-nunchaku** (official version)
- **Python 3.10-3.12**

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

## Automated Installation (Recommended)

### 1. Using Batch File

#### Installation
1. Double-click `install_qwen_lora.bat` to run
2. ComfyUI-nunchaku's `__init__.py` will be automatically modified
3. Restart ComfyUI

#### Uninstallation
1. Double-click `uninstall_qwen_lora.bat` to run
2. Original `__init__.py` will be restored
3. Restart ComfyUI

### 2. Batch File Features
- **Auto-detection**: Automatically detects ComfyUI location
- **Backup**: Automatically backs up original files
- **Error checking**: Verifies required files exist
- **Easy operation**: Double-click to run

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

### 2. Restart ComfyUI

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
- Re-run `install_qwen_lora.bat`

#### Import errors
- Verify ComfyUI-nunchaku is correctly installed
- Verify ComfyUI-QwenImageLoraLoader is in the correct location

#### Backup file not found
- If you manually edited `__init__.py`, restore the original content

### 2. Check Logs
Verify the following message in ComfyUI console:
- `Successfully imported Qwen Image LoRA loaders from ComfyUI-QwenImageLoraLoader`
- If error messages appear, check dependencies

## Uninstallation

### 1. Automated Uninstallation (Recommended)
Run `uninstall_qwen_lora.bat`

### 2. Manual Uninstallation
1. Remove the added code from `ComfyUI/custom_nodes/ComfyUI-nunchaku/__init__.py`
2. Restart ComfyUI

## Support

If you encounter issues, please report with the following information:
- Error messages
- ComfyUI version
- Installation method (automated/manual)
- Log file contents

## License

This project is licensed under the MIT License.
