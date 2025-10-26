# ComfyUI-QwenImageLoraLoader

A ComfyUI custom node for loading and applying LoRA (Low-Rank Adaptation) to Nunchaku Qwen Image models.

**This project is based on the fork version of ComfyUI-nunchaku-qwen-lora-suport-standalone.**

## Source

This LoRA loader was extracted and modified from the fork version:
- **Original Fork**: [ComfyUI-nunchaku-qwen-lora-suport-standalone](https://github.com/ussoewwin/ComfyUI-nunchaku-qwen-lora-suport-standalone)
- **Extraction**: LoRA functionality was extracted from the full fork to create an independent custom node
- **Integration**: Modified to work with the official ComfyUI-nunchaku plugin

## Features

- **NunchakuQwenImageLoraLoader**: Load and apply single LoRA to Qwen Image models
- **NunchakuQwenImageLoraStack**: Apply multiple LoRAs with dynamic UI control
- **Dynamic VRAM Management**: Automatic CPU offloading based on available VRAM
- **LoRA Composition**: Efficient LoRA stacking and composition
- **ComfyUI Integration**: Seamless integration with ComfyUI workflows

## Installation

### Quick Installation (Recommended)

1. Clone this repository to your ComfyUI custom_nodes directory:
```bash
git clone https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader.git
```

2. Ensure you have the official ComfyUI-nunchaku plugin installed (which includes nunchaku as a dependency).

3. **Run the installation script:**
   - Double-click `install_qwen_lora.bat` to automatically configure the integration
   - This script will automatically modify the ComfyUI-nunchaku `__init__.py` file
   - Restart ComfyUI

### Manual Installation

If you prefer manual installation, see [INSTALLATION.md](INSTALLATION.md) for detailed instructions.

## Integration with ComfyUI-nunchaku

**IMPORTANT**: This node requires modification of the official ComfyUI-nunchaku plugin to function properly.

### Required Modification

You must modify the `__init__.py` file in your ComfyUI-nunchaku plugin to import this LoRA loader:

```python
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

### Why This Modification is Required

- The independent LoRA loader exists as a separate custom node
- The official ComfyUI-nunchaku plugin needs to reference the independent LoRA loader
- Standard `from ... import ...` fails due to different package structures
- `importlib.util` allows direct module loading from file paths

### Automated Installation (Recommended)

**The batch file `install_qwen_lora.bat` automatically adds the above code to your ComfyUI-nunchaku `__init__.py` file:**

1. **What the batch file does:**
   - Automatically detects your ComfyUI installation path
   - Backs up the original `__init__.py` file (creates `__init__.py.backup`)
   - Adds the exact import code shown above to the end of ComfyUI-nunchaku's `__init__.py`
   - Performs error checking to ensure all required files exist
   - Provides user feedback throughout the process

2. **Run the installation script:**
   - Double-click `install_qwen_lora.bat` to automatically configure the integration
   - No manual code editing required - the script handles everything
   - Restart ComfyUI after installation

3. **Uninstall if needed:**
   - Double-click `uninstall_qwen_lora.bat` to restore the original configuration
   - This will restore the backup and remove the integration code

## Usage

### Available Nodes
- **NunchakuQwenImageLoraLoader**: Single LoRA loader
- **NunchakuQwenImageLoraStack**: Multi LoRA stacker with dynamic UI

### Basic Usage
1. Load your Nunchaku Qwen Image model using `Nunchaku Qwen Image DiT Loader`
2. Add either `NunchakuQwenImageLoraLoader` or `NunchakuQwenImageLoraStack` node
3. Select your LoRA file and set the strength
4. Connect to your workflow

### Dynamic UI Control
The `NunchakuQwenImageLoraStack` node automatically adjusts the number of visible LoRA slots based on the `lora_count` parameter (1-10).

## Features

- **Easy Installation**: Automated installation script included
- **Automatic Integration**: No manual code editing required
- **Backup & Restore**: Automatic backup of original files
- **Cross-Platform**: Works on Windows with batch files
- **Error Handling**: Comprehensive error checking and user feedback
- **Issue #1 Fixed**: Resolved [ComfyUI\custom_nodes not found error](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/1) with improved path detection (thanks to @mcv1234's solution)
- **Issue #2 Fixed**: Fixed UTF-8 encoding error causing `SyntaxError: (unicode error)` by using PowerShell for proper UTF-8 encoding when writing to `__init__.py` (thanks to @AHEKOT's bug report)

## Requirements

- ComfyUI
- ComfyUI-nunchaku plugin (with required modification)
- PyTorch
- Python 3.10+

## Compatibility

This node is designed to work with:
- ComfyUI-nunchaku plugin (modified)
- Nunchaku Qwen Image models
- Standard ComfyUI workflows

## Changelog

### v1.3.0 (Latest)
- **Fixed Critical Bug**: Resolved `SyntaxError: invalid character '' (U+FFFD)` error when running installation script
- **Problem**: PowerShell output was being written directly into Python files, causing syntax errors
- **Root Cause**: Using PowerShell commands with piping (`Get-Content | Add-Content`) caused PowerShell status messages and metadata to be included in the output, which were then written into `__init__.py`
- **Solution**: Replaced PowerShell-based approach with a dedicated Python script (`append_integration.py`)
- **Technical Details**:
  - Created standalone Python script `append_integration.py` that handles UTF-8 file writing
  - Batch file now calls the Python script instead of using inline PowerShell commands
  - Python script uses proper UTF-8 encoding when writing to `__init__.py`
  - Eliminates any possibility of output artifacts contaminating Python files
- **Benefits**:
  - More reliable and maintainable solution
  - No risk of shell output pollution
  - Cleaner separation of concerns between batch file and Python code
  - Better error handling and user feedback

### v1.2.0
- **Fixed [Issue #2](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/2)**: Resolved UTF-8 encoding error that caused `SyntaxError: (unicode error) 'utf-8' codec can't decode byte 0x90 in position 0`
- **Reported by**: @AHEKOT (GitHub Issue #2)
- **Special Thanks**: This critical bug was discovered and reported by @AHEKOT, who provided detailed error traces and screenshots
- **Technical Fix**: Changed the batch file to use PowerShell for UTF-8 encoding when writing Python code to `__init__.py`
- **Root Cause**: Windows batch files write in Shift-JIS encoding by default, which causes Python to fail when reading files as UTF-8
- **Solution**: Temporary file creation + PowerShell UTF-8 encoding ensures proper file encoding
- **Technical Details**:
  - Changed from direct append (`>>`) to temp file method
  - Uses PowerShell `Get-Content` and `Add-Content` with `-Encoding UTF8` parameters
  - Ensures all Python code is written with proper UTF-8 encoding
- **Impact**: Installation script now works correctly without encoding errors
- **Community Contribution**: This fix was made possible by @AHEKOT's thorough bug reporting and error documentation

### v1.1.0
- **Fixed [Issue #1](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/1)**: Resolved "ComfyUI\custom_nodes not found" error reported by @mcv1234
- **Special Thanks**: This fix was implemented based on the excellent solution provided by @mcv1234 in the GitHub issue
- **Improved Path Detection**: Replaced unreliable wildcard search with relative path detection using script directory (solution by @mcv1234)
- **Enhanced Error Messages**: Added clear directory structure guidance and expected folder layout
- **Better User Experience**: More reliable installation process with comprehensive error checking
- **Technical Details**: 
  - Changed from `dir /s /b /ad "*ComfyUI\custom_nodes"` to relative path calculation (as suggested by @mcv1234)
  - Uses `%~dp0` (script directory) to calculate ComfyUI root with `..\..` navigation
  - Added validation for `custom_nodes` folder existence
  - Improved error messages with expected directory structure display
- **Community Contribution**: This improvement was made possible by the community feedback and solution provided by @mcv1234

### v1.0.0
- Initial release with LoRA loading functionality
- Automated installation scripts
- Integration with ComfyUI-nunchaku

## License

This project is licensed under the MIT License.
