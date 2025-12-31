# v2.2.5 Release Notes - Repository Recovery

## Summary

**Critical Recovery Release**: This release restores all functionality after a catastrophic repository corruption incident. All updates after v2.0.8 were completely broken due to a `git force push` operation that corrupted the repository history and deleted critical files.

## Background: The Repository Corruption Incident

### What Happened

A `git force push` operation was performed that resulted in:
- **Complete loss of commit history** after v2.0.8
- **Mass deletion of critical files and directories**:
  - `images/` directory
  - `__init__.py` files
  - `nodes/` directory
  - `wrappers/` directory
  - `nunchaku_code/` directory
  - `js/` directory
  - `md/` directory
  - `LICENSE` file
  - `pyproject.toml` file

### Impact

This corruption meant that:
- All features added after v2.0.8 were non-functional
- Critical infrastructure files required for the node to function were missing
- Users could not use the repository in its broken state
- The repository was effectively unusable

## Recovery Work Performed

### 1. File Restoration

All deleted files and directories were restored from local backups:
- ✅ Restored `images/` directory with all UI assets
- ✅ Restored `__init__.py` files for proper node registration
- ✅ Restored `nodes/` directory with all LoRA loader nodes
- ✅ Restored `wrappers/` directory with model wrapper classes
- ✅ Restored `nunchaku_code/` directory with core LoRA composition logic
- ✅ Restored `js/` directory with dynamic UI extensions
- ✅ Restored `md/` directory with documentation
- ✅ Restored `LICENSE` file
- ✅ Restored `pyproject.toml` with project metadata

### 2. Feature Verification and Restoration

All features from v2.0.8 through v2.2.4 were verified and restored:

#### v2.2.4 Features (AWQ Support)
- ✅ **AWQ modulation layer detection and skip logic**: `img_mod.1` and `txt_mod.1` layers are detected and LoRA application is skipped by default to prevent noise
- ✅ **Environment variable override**: `QWENIMAGE_LORA_APPLY_AWQ_MOD=1` to override skip behavior
- ✅ **Detailed logging**: AWQ layer detection and skip status logged during LoRA application
- ✅ **AWQW4A16Linear forward patching**: Proper LoRA application to AWQ layers via forward method patching

#### v2.2.3 Features (Toggle Buttons)
- ✅ **Toggle buttons**: Enable/disable individual LoRA slots and all LoRAs at once
- ✅ **LoRA format detection**: Detailed logging of LoRA format detection (Standard LoRA, LoKR, LoHa, IA3)
- ✅ **Dynamic UI updates**: JavaScript extensions updated to reflect toggle button state
- ✅ **IS_CHANGED hash calculation**: Toggle states included in change detection

#### v2.2.0 Features (Z-ImageTurbo Support)
- ✅ **NextDiT forward signature detection**: Automatic detection of NextDiT models requiring `context` and `num_tokens` parameters
- ✅ **Automatic key mapping switching**: Dynamic switching between standard and NextDiT-specific key mappings
- ✅ **Z-ImageTurbo LoRA stacking**: Full support for Z-ImageTurbo LoRA composition
- ✅ **Module name resolution**: Improved module resolution for NextDiT's `attention` and `feed_forward` layers

#### v2.1.1 Features (ComfyUI v0.6.0+ Compatibility)
- ✅ **additional_t_cond migration**: Complete migration from `guidance` to `additional_t_cond` parameter
- ✅ **Both path support**: Both `customized_forward` and direct `self.model` call paths support new API
- ✅ **Backward compatibility**: Automatic conversion from `guidance` to `additional_t_cond` when needed

#### v2.1.0 Features (None Checks)
- ✅ **to_safely None check**: Prevents `AttributeError: 'NoneType' object has no attribute 'to'` when `self.model` is `None`
- ✅ **forward None check**: Raises `RuntimeError` when model is unloaded or garbage collected

### 3. Code Verification

All critical code paths were verified:
- ✅ `wrappers/qwenimage.py`: ComfyQwenImageWrapper with all fixes
- ✅ `wrappers/zimageturbo.py`: ComfyZImageTurboWrapper with NextDiT support
- ✅ `nunchaku_code/lora_qwen.py`: Core LoRA composition with AWQ support
- ✅ `nodes/lora/`: All LoRA loader nodes (V2, V3, Z-ImageTurbo V3)
- ✅ `js/`: All JavaScript extensions for dynamic UI
- ✅ `__init__.py`: Proper node registration

### 4. Version Update

- ✅ Updated `__version__` to "2.2.5" in `__init__.py`
- ✅ Updated `version` to "2.2.5" in `pyproject.toml`
- ✅ Updated `DisplayName` and `description` in `pyproject.toml` to include "Z-ImageTurbo"

## Technical Details

### NextDiT Forward Signature Detection

The recovery included proper implementation of NextDiT forward signature detection in `wrappers/zimageturbo.py`:

```python
import inspect
forward_sig = inspect.signature(self.model.forward)
forward_params = set(forward_sig.parameters.keys())

if "context" in forward_params and "num_tokens" in forward_params:
    # NextDiT specific processing
    return self.model(
        x_tensor,
        timestep,
        context=context,
        num_tokens=num_tokens_value,
        **forward_kwargs,
        **kwargs,
    )
```

This prevents `TypeError: NextDiT.forward() missing 2 required positional arguments: 'context' and 'num_tokens'` errors.

### AWQ Modulation Layer Skip Logic

The recovery included full implementation of AWQ modulation layer detection and skip logic in `nunchaku_code/lora_qwen.py`:

- Detection of `AWQW4A16Linear` modules
- Skip logic for `img_mod.1` and `txt_mod.1` layers
- Forward method patching for AWQ layers when override is enabled
- Proper reset logic to restore original forward methods

### LoRA Format Detection

The recovery included detailed LoRA format detection logging:
- Standard LoRA (Rank-Decomposed)
- LoKR (LoRA with Kronecker Product)
- LoHa (Low-Rank Hadamard Product)
- IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

All formats are detected and logged during LoRA loading.

## Related Issues

This recovery addresses multiple issues that arose from the repository corruption:

- **Repository corruption**: Complete loss of files and commit history after v2.0.8
- **Missing features**: All features from v2.0.8 through v2.2.4 were non-functional
- **NextDiT support**: NextDiT forward signature detection was missing
- **AWQ support**: AWQ modulation layer detection and skip logic was incomplete
- **Toggle buttons**: Toggle button functionality was not properly implemented

## Impact on Users

### For Users Updating from v2.0.8 or Earlier

✅ **All functionality restored**: You can now use all features from v2.0.8 through v2.2.4
✅ **No breaking changes**: All existing workflows continue to work
✅ **Improved stability**: All critical bugs fixed during recovery

### For Users on Broken Versions (v2.0.9 - v2.2.4)

⚠️ **Immediate update required**: If you were using a version between v2.0.9 and v2.2.4, you were likely experiencing:
- Missing nodes
- Import errors
- Runtime errors
- Missing features

✅ **Update to v2.2.5**: All issues are resolved in this release.

## Apology

We sincerely apologize for the inconvenience caused by the repository corruption incident. The recovery work has been thorough, and we have taken measures to prevent similar incidents in the future.

## Acknowledgments

Special thanks to all users who:
- Reported issues during the broken period
- Provided local backups for file restoration
- Tested the recovery builds
- Provided feedback on missing features

## Next Steps

- ✅ All critical files restored
- ✅ All features verified and working
- ✅ Version updated to 2.2.5
- ✅ Documentation updated
- 🔄 Continue monitoring for any remaining issues

If you encounter any issues after updating to v2.2.5, please report them immediately.

## Related Links

- [Issue #38](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/38) - ModuleNotFoundError: No module named 'diffusers.models.transformers.transformer_z_image'
- [Issue #37](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/37) - NunchakuZImageTurboLoraStackV2 removal
- [Issue #36](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/36) - Toggle button feature request
- [Issue #33](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/33) - None checks in to_safely and forward methods
- [v2.2.4 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.4)
- [v2.2.3 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.3)
- [v2.2.0 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.0)
- [v2.1.1 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.1.1)

---

**Release Date**: 2025-01-XX  
**Version**: 2.2.5  
**Status**: ✅ Stable - All functionality restored

