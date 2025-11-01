# v1.60 — MAJOR UPDATE: Simplified Installation (No Integration Required!)

## Summary

**As of v1.60, ComfyUI-QwenImageLoraLoader is now a fully independent custom node.**

- ✅ **Removed ComfyUI-nunchaku integration requirement** — No manual modification of `__init__.py` needed
- ✅ **Simplified installation** — Just `git clone` and restart ComfyUI
- ✅ **No batch scripts** — No installer/uninstaller scripts required
- ✅ **Automatic node registration** — ComfyUI's built-in mechanism handles everything
- ✅ **Backward compatible** — All existing LoRA files and workflows continue to work

## What Changed

### Installation (Before → After)

**Before v1.60:**
```
1. Clone repository
2. Choose installation script (global or portable Python)
3. Run batch file (modifies ComfyUI-nunchaku/__init__.py)
4. Restart ComfyUI
```

**After v1.60:**
```
1. Clone repository
2. Restart ComfyUI
Done!
```

### Architecture

The node now operates as a completely independent plugin through ComfyUI's automatic node loading mechanism:

1. ComfyUI scans `custom_nodes/` directory on startup
2. Finds `ComfyUI-QwenImageLoraLoader/__init__.py`
3. Automatically registers `NunchakuQwenImageLoraLoader` and `NunchakuQwenImageLoraStack` nodes
4. No external integration code needed

### Technical Details

**Why integration is no longer needed:**

1. **ComfyUI's automatic node discovery** — ComfyUI automatically scans all `custom_nodes/*/` directories and loads their `__init__.py` files
2. **Independent NODE_CLASS_MAPPINGS** — Each plugin's `NODE_CLASS_MAPPINGS` is merged automatically by ComfyUI
3. **Direct type imports** — `NunchakuQwenImageTransformer2DModel` is imported directly from the nunchaku package
4. **Model-agnostic LoRA composition** — `compose_loras_v2()` function works with any model that has `_lora_slots` attribute
5. **Wrapper-based integration** — `ComfyQwenImageWrapper` handles all LoRA logic independently

For a complete technical explanation, see [WHY_NO_NUNCHAKU_INTEGRATION.md](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/WHY_NO_NUNCHAKU_INTEGRATION.md)

## Breaking Changes

**None** — This is a pure architectural simplification. All workflows and LoRA files from v1.57 continue to work without modification.

## Files Removed

The following installer/uninstaller scripts are no longer included (or can be deleted if present):
- `install_qwen_lora.bat`
- `install_qwen_lora_portable.bat`
- `uninstall_qwen_lora.bat`
- `uninstall_qwen_lora_portable.bat`
- `append_integration.py`
- `remove_integration.py`
- `INSTALLATION.md` (old installation guide)

## Installation Instructions

### Quick Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader.git
cd ComfyUI-QwenImageLoraLoader
# Optional: pip install -r requirements.txt
# Restart ComfyUI
```

### Requirements

- Python 3.11+
- ComfyUI (latest version)
- ComfyUI-nunchaku plugin (for Qwen Image support)
- CUDA-capable GPU (optional, recommended for performance)

## Upgrade from v1.57 or Earlier

If you have v1.57 or earlier installed:

1. **If you used the installer scripts before:**
   - The integration code is already in your ComfyUI-nunchaku `__init__.py`
   - It's safe to leave it there (it will simply be ignored)
   - You can optionally remove the integration code (see troubleshooting below)

2. **To upgrade to v1.60:**
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-QwenImageLoraLoader
   git pull origin main
   # No restart needed if you just pulled - already running standalone
   ```

3. **Optional: Clean up old integration code**
   - Edit `ComfyUI-nunchaku/__init__.py`
   - Search for "ComfyUI-QwenImageLoraLoader Integration"
   - Delete that entire try/except block
   - Restart ComfyUI

## Troubleshooting

### Nodes Not Appearing

1. Ensure `ComfyUI-QwenImageLoraLoader` is in `custom_nodes/` directory
2. Check folder name is exactly `ComfyUI-QwenImageLoraLoader` (case-sensitive on Linux/Mac)
3. Verify `__init__.py` exists in the root directory
4. Restart ComfyUI completely
5. Check ComfyUI console for error messages

### Old Integration Code

If you have old integration code in ComfyUI-nunchaku `__init__.py`:

1. It won't cause conflicts (the plugin ignores it)
2. To remove it:
   - Backup ComfyUI-nunchaku `__init__.py`
   - Find the section with "ComfyUI-QwenImageLoraLoader Integration"
   - Delete the entire try/except block
   - Save and restart ComfyUI

## Known Issues

- **RES4LYF Sampler**: Not supported due to device mismatch issues (Issue #7, #8)
  - Workaround: Use other samplers
- **LoRA Stack UI**: 10th row always visible (Issue #9)
  - Visual only; does not affect functionality

## Testing

This release has been verified to work with:
- ✅ Single LoRA loading (`NunchakuQwenImageLoraLoader`)
- ✅ Multiple LoRA stacking (`NunchakuQwenImageLoraStack`)
- ✅ Dynamic VRAM management
- ✅ CPU offload transitions
- ✅ ComfyUI workflows

## Backward Compatibility

- ✅ All v1.57 and earlier LoRA files work without modification
- ✅ All existing workflows work without modification
- ✅ Old integration code in ComfyUI-nunchaku `__init__.py` is safely ignored

## Special Thanks

- GavChap for the original LoRA composition implementation
- Nunchaku team for the underlying model and infrastructure
- Community members for testing and feedback

## Full Changelog

For detailed technical information about this release, see:
- [README.md#Changelog](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/README.md#changelog)
- [WHY_NO_NUNCHAKU_INTEGRATION.md](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/WHY_NO_NUNCHAKU_INTEGRATION.md)
