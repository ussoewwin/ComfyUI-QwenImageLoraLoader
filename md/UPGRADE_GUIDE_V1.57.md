# Upgrade Guide for v1.57 and Earlier Users

If you have v1.57 or earlier installed with integration code in ComfyUI-nunchaku's `__init__.py`, you have three options:

## Option 1: Keep Everything As-Is (Recommended for Most Users)

**The integration code will continue to work without any issues. You don't need to do anything:**

1. Update ComfyUI-QwenImageLoraLoader to v1.60:
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-QwenImageLoraLoader
   git pull origin main
   ```

2. Restart ComfyUI

**That's it!** The old integration code will be safely ignored, and the new standalone mechanism will take over. All your existing workflows and LoRA files continue to work exactly as before.

## Option 2: Clean Up Old Integration Code (For Cleaner Repository)

**⚠️ Windows Only - Batch Scripts Not Available for macOS/Linux**

If you want to remove the old integration code from ComfyUI-nunchaku's `__init__.py`:

**This option ONLY works on Windows.** If you are using macOS or Linux, please use Option 3 (Manual Cleanup) below.

### For Windows Users:

1. Update to v1.60:
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-QwenImageLoraLoader
   git pull origin main
   ```

2. Run the uninstaller to remove the old integration code:

   **For Global Python Environment:**
   ```cmd
   cd ComfyUI\custom_nodes\ComfyUI-QwenImageLoraLoader
   uninstall_qwen_lora.bat
   ```

   **For Portable ComfyUI with Embedded Python:**
   ```cmd
   cd ComfyUI\custom_nodes\ComfyUI-QwenImageLoraLoader
   uninstall_qwen_lora_portable.bat
   ```

3. The uninstaller will restore your ComfyUI-nunchaku `__init__.py` to its original state

4. Restart ComfyUI

**After uninstalling the old integration code, the node will still work perfectly** because v1.60 uses the standalone loading mechanism.

## Option 3: Manual Cleanup (For macOS/Linux, or Users Who Prefer Manual Editing)

**For macOS/Linux users** (batch scripts are not available), or if you prefer to manually edit files, you have two choices:

### Method A: Manually Delete the Integration Code Block

1. Open `ComfyUI/custom_nodes/ComfyUI-nunchaku/__init__.py` in a text editor

2. Find and delete the entire following block (it will be at the end of the file):

```python
# BEGIN ComfyUI-QwenImageLoraLoader Integration
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
    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraStack"] = qwenimage_module.NunchakuQwenImageLoraStack
    logger.info("Successfully imported Qwen Image LoRA loaders from ComfyUI-QwenImageLoraLoader")
except ImportError:
    logger.exception("Nodes `NunchakuQwenImageLoraLoader` and `NunchakuQwenImageLoraStack` import failed:")
# END ComfyUI-QwenImageLoraLoader Integration
```

3. Delete this entire block from the file (from `# BEGIN` to `# END` markers inclusive)

4. Save the file

5. Restart ComfyUI

**Important:** Look for the BEGIN and END markers. Delete everything from `# BEGIN ComfyUI-QwenImageLoraLoader Integration` to `# END ComfyUI-QwenImageLoraLoader Integration` (inclusive).

## Option 4: Restore Official ComfyUI-nunchaku `__init__.py` (Emergency Recovery)

**If your ComfyUI-nunchaku `__init__.py` becomes corrupted, broken, or unrecoverable**, you can restore it from the official Nunchaku repository:

1. Download the official `__init__.py` from the [ComfyUI-nunchaku repository](https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/__init__.py)

2. Copy the downloaded file to: `ComfyUI/custom_nodes/ComfyUI-nunchaku/__init__.py`

3. Restart ComfyUI

The official `__init__.py` will not have any ComfyUI-QwenImageLoraLoader integration code. v1.60 will still work perfectly because it uses the standalone loading mechanism.

## Why the Integration Code is No Longer Needed

Starting with v1.60, ComfyUI-QwenImageLoraLoader operates as a **completely independent custom node**. Here's why integration is no longer necessary:

1. **ComfyUI's Automatic Node Discovery**: ComfyUI automatically scans the `custom_nodes/` directory and loads all `__init__.py` files at startup

2. **Automatic NODE_CLASS_MAPPINGS Merging**: All `NODE_CLASS_MAPPINGS` from different plugins are automatically merged into a single registry

3. **Direct Type Imports**: The loader imports `NunchakuQwenImageTransformer2DModel` directly from the nunchaku package, without needing main body integration

4. **Model-Agnostic LoRA Composition**: The `compose_loras_v2()` function works with any model that has `_lora_slots`, independent of the main body

5. **Wrapper-Based Architecture**: All LoRA logic is handled by `ComfyQwenImageWrapper`, which is completely self-contained

**For a complete technical explanation with 7 detailed chapters, see [v1.60 Release Notes on GitHub](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v1.60)**

## Backward Compatibility

✅ **All existing setups continue to work:**

- Workflows created with v1.57 or earlier work without modification
- LoRA files work without any changes
- Old integration code in ComfyUI-nunchaku `__init__.py` is safely ignored
- No breaking changes to node inputs/outputs

