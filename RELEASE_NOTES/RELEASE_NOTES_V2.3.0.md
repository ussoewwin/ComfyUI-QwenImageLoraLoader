# v2.3.0 - Z-ImageTurbo Loader v4: Standard Format Compliance with Perfect Mapping

## ⚠️ **REQUIREMENTS**

**To use Z-ImageTurbo Loader v4, you must update to:**
- **Nunchaku 1.2.0 or later**
- **ComfyUI-Nunchaku 1.2.0 or later**

v4 is designed for the official Nunchaku Z-Image loader (NextDiT) introduced in ComfyUI-Nunchaku v1.2.0. If you are using an older version, please update both packages before using v4.

## Overview

This release updates the Z-ImageTurbo Loader to v4, which conforms to the standard ComfyUI LoRA loader format (CLIP input/output, no CPU offload parameter) while maintaining the same perfect mapping functionality as v3. Additionally, Z-ImageTurbo Loader v3 registration has been removed from ComfyUI node registration.

## Changes

### Removed
- **Z-ImageTurbo Loader v3 registration**: Removed from ComfyUI node registration. The node file remains in the repository but is no longer registered. Users should use `NunchakuZImageTurboLoraStackV4` instead.
- **Diffsynth ControlNet support**: Removed `NunchakuQI&ZITDiffsynthControlnet` node registration and all related documentation. ComfyUI-Nunchaku now has native support for ZIT (Z-Image-Turbo) Diffsynth ControlNet, so this custom node is no longer needed.

### Updated
- **Z-ImageTurbo Loader v4**: Updated to conform to standard ComfyUI LoRA loader format while maintaining perfect mapping functionality
  - **CLIP Input/Output**: Added CLIP input and output (v3 had MODEL only)
  - **CPU Offload Parameter Removal**: Removed `cpu_offload` parameter to conform to standard format
  - **Standard Format Compliance**: Now matches standard ComfyUI LoRA loader interface `(MODEL, CLIP)` input/output
  - **Perfect Mapping Maintained**: Uses the same `compose_loras_v2` as v3 for perfect mapping
  - **Enhanced Fallback Functionality**: Three-point fallback system (import, model type, application) for maximum compatibility and robustness

## Technical Details

### Standard Format Compliance

v4 conforms to the standard ComfyUI LoRA loader format by:

1. **CLIP Input Addition**
   - v3: `(MODEL,)` input/output only
   - v4: `(MODEL, CLIP)` input/output (compatibility with standard ComfyUI LoRA loader format)

2. **CPU Offload Parameter Removal**
   - v3: `cpu_offload` parameter required
   - v4: No CPU offload parameter (conforming to standard format)

3. **CLIP Return**
   - CLIP is returned unchanged (LoRAs are only applied to the model, not to CLIP)
   - Maintains compatibility with standard ComfyUI LoRA loader interface

### Perfect Mapping Functionality

v4 maintains the same perfect mapping functionality as v3:

- Uses `compose_loras_v2` directly to achieve perfect mapping
- Avoids issues with the official implementation using the same mapping logic as v3
- Maintains detailed debug logging functionality equivalent to v3

### Enhanced Fallback Functionality

v4 implements a three-point fallback system for maximum compatibility:

1. **Import Fallback**: If `compose_loras_v2` cannot be imported, falls back to standard `load_lora_for_models`
2. **Model Type Fallback**: If model type is unsupported, falls back to standard `load_lora_for_models`
3. **Application Fallback**: If `compose_loras_v2` raises exception during application, falls back to standard `load_lora_for_models`

This ensures the node continues to work even if the custom mapping function is unavailable or encounters errors.

## Code Implementation Details

### Class Structure

```python
class NunchakuZImageTurboLoraStackV4:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku Z-Image-Turbo model with dynamic UI.
    V4 uses compose_loras_v2 for perfect mapping (same as v3).
    V4 is for official Nunchaku Z-Image loader only.
    """
```

### Key Method Signature Changes

**v3**:
```python
def load_lora_stack(self, model, lora_count, cpu_offload="disable", toggle_all=True, **kwargs):
    return (ret_model,)
```

**v4**:
```python
def load_lora_stack(self, model, clip, lora_count, toggle_all=True, **kwargs):
    return (model, clip)  # CLIP unchanged
```

### IS_CHANGED Method

v4 includes CLIP in the hash calculation (unlike v3), ensuring the node re-executes when CLIP changes (for consistency with standard ComfyUI behavior):

```python
@classmethod
def IS_CHANGED(cls, model, clip, lora_count, toggle_all=True, **kwargs):
    import hashlib
    m = hashlib.sha256()
    m.update(str(model).encode())      # Hash model reference
    m.update(str(clip).encode())       # Hash CLIP reference (NEW in v4)
    # ... hash other parameters ...
    return m.digest().hex()
```

### INPUT_TYPES and RETURN_TYPES

**v3**:
```python
INPUT_TYPES = {
    "required": {
        "model": ("MODEL", ...),
        "cpu_offload": ("STRING", ...),  # Required in v3
        # ...
    }
}
RETURN_TYPES = ("MODEL",)  # MODEL only
```

**v4**:
```python
INPUT_TYPES = {
    "required": {
        "model": ("MODEL", ...),
        "clip": ("CLIP", ...),  # NEW in v4
        # ... (no cpu_offload parameter)
    }
}
RETURN_TYPES = ("MODEL", "CLIP")  # MODEL and CLIP
```

## JavaScript Extension Updates

The JavaScript extension (`js/zimageturbo_lora_dynamic_v4.js`) has been updated to:

- Remove CPU offload widget (v4 has no such parameter)
- Use extension name `"nunchaku.zimageturbo_lora_dynamic_v4"` to avoid conflicts with v2/v3 extensions
- Filter for `NunchakuZImageTurboLoraStackV4` node class only

## Migration Guide

### For Users Currently Using v3

If you are currently using `NunchakuZImageTurboLoraStackV3`:

1. **Update Your Workflows**: Replace `NunchakuZImageTurboLoraStackV3` nodes with `NunchakuZImageTurboLoraStackV4` nodes
2. **Add CLIP Input**: Connect CLIP input to the v4 node (v3 had MODEL only)
3. **Remove CPU Offload Parameter**: v4 does not have a CPU offload parameter (it's removed from the interface)
4. **CLIP Output**: The v4 node returns both MODEL and CLIP (CLIP unchanged)

### Compatibility

- **Mapping Quality**: v4 uses the same `compose_loras_v2` as v3, so mapping quality is equivalent
- **Standard Format**: v4 is recommended for new workflows due to standard format compliance
- **Backward Compatibility**: v3 node file remains in the repository but is no longer registered

## Comparison with v3

| Feature | v3 | v4 |
|---------|-----|-----|
| **Input** | MODEL only | MODEL + CLIP |
| **Output** | MODEL only | MODEL + CLIP |
| **CPU Offload Parameter** | Required | Removed |
| **Mapping Mechanism** | compose_loras_v2 | compose_loras_v2 (same) |
| **Standard Format Compliance** | Partial | Complete |
| **Fallback Functionality** | None | Three-point fallback |

## Files Modified

- `nodes/lora/zimageturbo_v4.py`: Updated to standard format (CLIP input/output, no CPU offload)
- `js/zimageturbo_lora_dynamic_v4.js`: Updated to remove CPU offload widget
- `__init__.py`: v3 registration removed, v4 registration maintained

## Related Documentation

For complete technical documentation, see:
- [V4 Development Explanation](md/V4_DEVELOPMENT_EXPLANATION.md) - Complete technical documentation with detailed code explanations

## Summary

v2.3.0 updates the Z-ImageTurbo Loader to v4, which conforms to the standard ComfyUI LoRA loader format while maintaining perfect mapping functionality. The v3 node registration has been removed, and users are encouraged to migrate to v4 for better compatibility with standard ComfyUI workflows.
