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

### Perfect Mapping Functionality - Core Technical Explanation

v4 uses `compose_loras_v2` to achieve **perfect mapping** between LoRA keys and the NextDiT model structure. This is the core technical innovation that makes v4 work correctly.

#### The Problem: Standard LoRA Loader's Incomplete Mapping

The standard ComfyUI LoRA loader has incomplete key mapping for Z-Image-Turbo (NextDiT) models. LoRA keys from standard training (e.g., `layers.0.attention.to_qkv`) don't match the NextDiT structure (`layers.0.attention.qkv`), causing LoRAs to be silently skipped or only partially applied when using the standard loader.

#### The Solution: Dynamic Key Mapping with compose_loras_v2

v4 uses `compose_loras_v2` which implements a sophisticated key mapping system:

**1. Automatic Model Structure Detection**

`compose_loras_v2` automatically detects the model structure and switches mapping rules:

```python
# From nunchaku_code/lora_qwen.py (lines 1179-1198)
global _ACTIVE_KEY_MAPPING
nextdit_markers = (
    "layers.0.attention.qkv",
    "layers.0.attention.out",
    "layers.0.feed_forward.w1",
    "layers.0.feed_forward.w2",
    "layers.0.feed_forward.w3",
    "layers.0.feed_forward.w13",
)
is_nextdit_style = any(_get_module_by_name(model, p) is not None for p in nextdit_markers)
if is_nextdit_style:
    has_w13 = _get_module_by_name(model, "layers.0.feed_forward.w13") is not None
    if has_w13:
        _ACTIVE_KEY_MAPPING = ZIMAGE_NEXTDIT_NUNCHAKU_PATCHED_KEY_MAPPING + KEY_MAPPING
    else:
        _ACTIVE_KEY_MAPPING = ZIMAGE_NEXTDIT_UNPATCHED_KEY_MAPPING + KEY_MAPPING
```

**2. NextDiT-Specific Key Mappings**

For NextDiT models, special mappings are used to convert LoRA keys to match the actual model structure:

```python
# ZIMAGE_NEXTDIT_UNPATCHED_KEY_MAPPING (for standard NextDiT)
# Maps: layers.0.attention.to_qkv -> layers.0.attention.qkv
# Maps: layers.0.feed_forward.w1/w2/w3 -> layers.0.feed_forward.w1/w2/w3

# ZIMAGE_NEXTDIT_NUNCHAKU_PATCHED_KEY_MAPPING (for nunchaku-patched NextDiT)
# Maps: layers.0.feed_forward.w1/w3 -> layers.0.feed_forward.w13 (fused GLU)
# Maps: layers.0.feed_forward.w2 -> layers.0.feed_forward.w2
```

**3. Key Classification and Mapping Process**

Each LoRA key is classified and mapped using `_classify_and_map_key`:

```python
# From nunchaku_code/lora_qwen.py (lines 326-380)
def _classify_and_map_key(key: str) -> Optional[Tuple[str, str, Optional[str], str]]:
    """
    Classifies a LoRA key and maps it to the model structure.
    Returns: (group, base_key, component, ab) where:
    - group: "qkv", "glu", "regular", etc.
    - base_key: Mapped model key (e.g., "layers.0.attention.qkv")
    - component: Component identifier (e.g., "Q", "K", "V" for QKV)
    - ab: LoRA matrix type ("A", "B", "lokr_w1", "lokr_w2")
    """
    # Extract base key and LoRA suffix (A/B or lokr_w1/lokr_w2)
    # Apply regex patterns from _ACTIVE_KEY_MAPPING or KEY_MAPPING
    # Return mapped key that matches actual model structure
```

**4. Why It's Called "Perfect Mapping" (Standard LoRA Format Only)**

**Important**: "Perfect mapping" refers specifically to **Standard LoRA format** (A/B matrices). For other formats like LoKR, LoHa, IA3, etc., perfect mapping is **not possible** - these formats are fundamentally incompatible with the perfect mapping mechanism.

For Standard LoRA format:
- **Complete Coverage**: All Standard LoRA keys (A/B matrices) are correctly mapped to the NextDiT model structure
- **No Silent Failures**: Unlike the standard loader, no Standard LoRA keys are silently skipped
- **Automatic Detection**: Detects nunchaku-patched vs unpatched NextDiT automatically
- **QKV Fusion**: Correctly handles fused QKV attention layers (maps `to_q`, `to_k`, `to_v` → `qkv`)
- **GLU Fusion**: Correctly handles fused GLU feed-forward layers (maps `w1`, `w3` → `w13` for patched models)

**Note on Other Formats**: LoKR (lokr_w1/lokr_w2) and other formats are supported but **cannot achieve perfect mapping** - they fall back to standard loader behavior which may have incomplete mapping.

**5. Code Flow in v4**

```python
# From nodes/lora/zimageturbo_v4.py (lines 336-405)
# Import compose_loras_v2
from nunchaku_code.lora_qwen import compose_loras_v2

# Prepare LoRA configs: [(lora_path, strength), ...]
lora_configs = []
for lora_name, lora_strength in loras_to_apply:
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora_configs.append((lora_path, lora_strength))

# Apply using compose_loras_v2 (perfect mapping happens inside)
compose_loras_v2(transformer, lora_configs)
```

**6. Comparison with Official Implementation**

| Aspect | Official Implementation | v4 (compose_loras_v2) |
|--------|------------------------|------------------------|
| **Key Mapping (Standard LoRA)** | Incomplete, many keys skipped | Complete, all Standard LoRA keys mapped (perfect mapping) |
| **Model Detection** | Static, assumes one structure | Dynamic, auto-detects NextDiT structure |
| **QKV Handling** | May fail on fused QKV | Correctly handles fused QKV |
| **GLU Handling** | May fail on fused GLU (w13) | Correctly handles both w1/w3 and w13 |
| **LoKR Support** | Limited | Supported but not "perfect mapping" |
| **Debug Logging** | Minimal | Comprehensive key-by-key logging |

**7. Technical Details: Key Mapping Patterns**

The mapping uses regex patterns to transform LoRA keys:

```python
# Example: QKV Attention Mapping
# LoRA key: "layers.0.attention.to_qkv.lora_A"
# Pattern: r"^(layers)[._](\d+)[._]attention[._]to[._]([qkv])$"
# Template: r"\1.\2.attention.qkv"
# Result: "layers.0.attention.qkv" (matches NextDiT structure)
# Component: "Q" (extracted from match group 3)

# Example: GLU Feed-Forward Mapping (Nunchaku-Patched)
# LoRA key: "layers.0.feed_forward.w1.lora_A"
# Pattern: r"^(layers)[._](\d+)[._]feed_forward[._](w1|w3)$"
# Template: r"\1.\2.feed_forward.w13"
# Result: "layers.0.feed_forward.w13" (fused GLU in patched NextDiT)
```

This perfect mapping (for Standard LoRA format) ensures that **all Standard LoRA weights (A/B matrices) are correctly applied to the model**, achieving the same quality as v3 while conforming to the standard ComfyUI LoRA loader format.

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
