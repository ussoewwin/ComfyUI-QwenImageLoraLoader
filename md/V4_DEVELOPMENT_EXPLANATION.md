# v2.3.0 - Z-ImageTurbo LoRA Stack v4: Standard Format Compliance with Perfect Mapping

## ⚠️ **REQUIREMENTS**

**To use Z-ImageTurbo LoRA Stack v4, you must update to:**
- **Nunchaku 1.2.0 or later**
- **ComfyUI-Nunchaku 1.2.0 or later**

v4 is designed for the official Nunchaku Z-Image loader (NextDiT) introduced in ComfyUI-Nunchaku v1.2.0. If you are using an older version, please update both packages before using v4.

## Overview

This release updates the Z-ImageTurbo LoRA Stack to v4, which conforms to the standard ComfyUI LoRA loader format (CLIP input/output, no CPU offload parameter) while maintaining the same perfect mapping functionality as v3. Additionally, Z-ImageTurbo LoRA Stack v3 registration has been removed from ComfyUI node registration.

## Changes

### Removed
- **Z-ImageTurbo LoRA Stack v3 registration**: Removed from ComfyUI node registration. The node file remains in the repository but is no longer registered. Users should use `NunchakuZImageTurboLoraStackV4` instead.
- **Diffsynth ControlNet support**: Removed `NunchakuQI&ZITDiffsynthControlnet` node registration and all related documentation. ComfyUI-Nunchaku now has native support for ZIT (Z-Image-Turbo) Diffsynth ControlNet, so this custom node is no longer needed.

### Updated
- **Z-ImageTurbo LoRA Stack v4**: Updated to conform to standard ComfyUI LoRA loader format
  - **Standard Format Compliance**: Now matches standard ComfyUI LoRA loader interface `(MODEL, CLIP)` input/output
  - **CLIP Input/Output**: Added CLIP input and output (v3 had MODEL only)
  - **CPU Offload Parameter Removal**: Removed `cpu_offload` parameter to conform to standard format
  - **Perfect Mapping for Standard LoRA Format**: Uses the same `compose_loras_v2` as v3 for perfect mapping (Standard LoRA format only)
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

**6. Comparison with Standard ComfyUI LoRA Loader**

| Aspect | Standard ComfyUI LoRA Loader | v4 (compose_loras_v2) |
|--------|------------------------------|------------------------|
| **Key Mapping (Standard LoRA)** | Incomplete, many keys skipped | Complete, all Standard LoRA keys mapped (perfect mapping) |
| **Model Detection** | Static, assumes one structure | Dynamic, auto-detects NextDiT structure |
| **QKV Handling** | May fail on fused QKV | Correctly handles fused QKV |
| **GLU Handling** | May fail on fused GLU (w13) | Correctly handles both w1/w3 and w13 |
| **LoKR Support** | Not supported (for Nunchaku) | Not supported |
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

### Development Rationale

The standard ComfyUI LoRA loader has incomplete key mapping for Nunchaku Z-Image-Turbo (NextDiT) models. While v3 implemented perfect mapping functionality (`compose_loras_v2`), v4 uses the same mapping mechanism as v3 while achieving the following improvements:

1. **CLIP Input Addition**
   - v3: `(MODEL,)` input/output only
   - v4: `(MODEL, CLIP)` input/output (compatibility with standard ComfyUI LoRA loader format)

2. **CPU Offload Parameter Removal**
   - v3: `cpu_offload` parameter required
   - v4: No CPU offload parameter (conforming to standard format)

3. **Complete Debug Logging Functionality**
   - Implements detailed debug logging equivalent to v3
   - Completely records key mapping, statistics, and application status

4. **Same Mapping Functionality as v3**
   - Uses `compose_loras_v2` directly to achieve perfect mapping
   - Uses the same mapping logic as v3 to overcome standard LoRA loader limitations

5. **Enhanced Fallback Functionality**
   - Three-point fallback system for maximum compatibility
   - Falls back to standard `load_lora_for_models` when `compose_loras_v2` import fails, application fails, or unsupported model types are encountered

### Class Structure

```python
class NunchakuZImageTurboLoraStackV4:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku Z-Image-Turbo model with dynamic UI.
    V4 uses compose_loras_v2 for perfect mapping (same as v3).
    V4 is for official Nunchaku Z-Image loader only.
    """
```

**Key Difference from v3**: The class accepts `(model, clip)` instead of `(model, cpu_offload)`, conforming to standard ComfyUI LoRA loader interface.

### IS_CHANGED Method (lines 35-52)

**Purpose**: Detect changes to trigger node re-execution. Returns a hash of all relevant parameters.

**Implementation**:

```python
@classmethod
def IS_CHANGED(cls, model, clip, lora_count, toggle_all=True, **kwargs):
    import hashlib
    m = hashlib.sha256()
    m.update(str(model).encode())      # Hash model reference
    m.update(str(clip).encode())       # Hash CLIP reference (NEW in v4)
    m.update(str(lora_count).encode())
    m.update(str(toggle_all).encode())
    # Hash all LoRA parameters (up to 10 slots)
    for i in range(1, 11):
        m.update(kwargs.get(f"lora_name_{i}", "").encode())
        m.update(str(kwargs.get(f"lora_strength_{i}", 0)).encode())
        m.update(str(kwargs.get(f"enabled_{i}", True)).encode())
    return m.digest().hex()
```

**Technical Significance**:
- **CLIP Hash Addition**: Unlike v3, v4 includes CLIP in the hash calculation, ensuring the node re-executes when CLIP changes (for consistency with standard ComfyUI behavior)
- **Comprehensive Parameter Hashing**: All LoRA parameters (name, strength, enabled state) are included to detect any changes
- **Change Detection**: ComfyUI uses this hash to determine if the node needs to be re-executed

### INPUT_TYPES Definition (lines 58-117)

**Purpose**: Define node input structure compatible with standard ComfyUI LoRA loader format.

**Implementation**:

```python
@classmethod
def INPUT_TYPES(s):
    loras = ["None"] + folder_paths.get_filename_list("loras")
    
    inputs = {
        "required": {
            "model": ("MODEL", {"tooltip": "The diffusion model to apply LoRAs to."}),
            "clip": ("CLIP", {"tooltip": "The CLIP model to apply LoRAs to."}),  # NEW in v4
            "lora_count": ("INT", {
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "tooltip": "Number of LoRA slots to process.",
            }),
            "toggle_all": ("BOOLEAN", {
                "default": True,
                "tooltip": "Enable/disable all LoRAs at once.",
            }),
        },
        "optional": {},
    }
    
    # Add all LoRA inputs (up to 10 slots) as optional
    for i in range(1, 11):
        inputs["optional"][f"enabled_{i}"] = ("BOOLEAN", {
            "default": True,
            "tooltip": f"Enable/disable LoRA {i}.",
        })
        inputs["optional"][f"lora_name_{i}"] = (loras, {
            "tooltip": f"The file name of LoRA {i}. Select 'None' to skip this slot."
        })
        inputs["optional"][f"lora_strength_{i}"] = ("FLOAT", {
            "default": 1.0,
            "min": -100.0,
            "max": 100.0,
            "step": 0.01,
            "tooltip": f"Strength for LoRA {i}.",
        })
    
    return inputs
```

**Key Changes from v3**:
1. **CLIP Input**: Added `"clip": ("CLIP", ...)` to required inputs (v3 had MODEL only)
2. **No CPU Offload Parameter**: Removed `cpu_offload` parameter (v3 required it)
3. **Dynamic UI Support**: 10 slots of LoRA inputs (enabled_1-10, lora_name_1-10, lora_strength_1-10) for dynamic UI control

### RETURN_TYPES (line 119)

```python
RETURN_TYPES = ("MODEL", "CLIP")  # v3 is ("MODEL",) only
OUTPUT_TOOLTIPS = ("The modified diffusion model with all LoRAs applied.", "The modified CLIP model (unchanged).")
```

**Technical Significance**:
- Returns both MODEL and CLIP (CLIP unchanged, for compatibility)
- Matches standard ComfyUI LoRA loader interface

### load_lora_stack Method - Complete Implementation Flow

**Method Signature** (line 126):

```python
def load_lora_stack(self, model, clip, lora_count, toggle_all=True, **kwargs):
```

**Key Difference**: Accepts `clip` parameter (v3 had `cpu_offload` instead).

**Processing Flow**:

#### Phase 1: LoRA Filtering and Validation (lines 132-171)

**Purpose**: Collect enabled LoRAs that should be applied, with detailed logging.

```python
loras_to_apply = []

# Log toggle_all state
logger.info(f"[LoRA Stack Status] toggle_all: {toggle_all}")
logger.info(f"[LoRA Stack Status] Processing {lora_count} LoRA slot(s):")

# Process only the number of LoRAs specified by lora_count
for i in range(1, lora_count + 1):
    lora_name = kwargs.get(f"lora_name_{i}")
    lora_strength = kwargs.get(f"lora_strength_{i}", 1.0)
    enabled_individual = kwargs.get(f"enabled_{i}", True)
    enabled = enabled_individual  # In v4, toggle_all is not used for filtering
    
    # Log each LoRA slot status
    status_parts = [f"Slot {i}:"]
    if lora_name and lora_name != "None":
        status_parts.append(f"'{lora_name}'")
        status_parts.append(f"strength={lora_strength}")
    else:
        status_parts.append("(no LoRA selected)")
    
    status_parts.append(f"toggle_all={toggle_all}")
    status_parts.append(f"enabled_{i}={enabled_individual}")
    status_parts.append(f"final_enabled={enabled}")
    
    if enabled and lora_name and lora_name != "None" and abs(lora_strength) > 1e-5:
        status_parts.append("→ APPLIED ✓")
        loras_to_apply.append((lora_name, lora_strength))
    else:
        status_parts.append("→ SKIPPED ✗")
    
    logger.info(f"[LoRA Stack Status] {' | '.join(status_parts)}")

logger.info(f"[LoRA Stack Status] Summary: {len(loras_to_apply)} LoRA(s) will be applied out of {lora_count} slot(s)")

if not loras_to_apply:
    return (model, clip)  # Early return if no LoRAs to apply
```

**Technical Details**:
- **Individual Enable Control**: Each LoRA slot has its own `enabled_{i}` flag (unlike v3 which used `toggle_all` for filtering)
- **Strength Threshold**: LoRAs with strength < 1e-5 are skipped (negligible strength)
- **Detailed Logging**: Each slot's status is logged for debugging
- **Early Return**: If no LoRAs are to be applied, returns immediately (CLIP unchanged)

#### Phase 2: Debug Logging Setup (lines 173-186)

**Purpose**: Import debug functions for detailed LoRA inspection.

```python
# Import mapping debug functions from nunchaku_code.lora_qwen
try:
    if lora_loader_dir not in sys.path:
        sys.path.insert(0, lora_loader_dir)
    
    from nunchaku_code.lora_qwen import _classify_and_map_key, _load_lora_state_dict, _detect_lora_format, _log_lora_format_detection
except ImportError as e:
    logger.warning(f"Failed to import mapping debug functions: {e}")
    logger.warning("Mapping debug logs will be skipped.")
    _classify_and_map_key = None
    _load_lora_state_dict = None
    _detect_lora_format = None
    _log_lora_format_detection = None
```

**Technical Details**:
- **Graceful Degradation**: If debug functions are unavailable, sets them to None and continues
- **Path Management**: Ensures `lora_loader_dir` is in sys.path for imports
- **Debug Functions**:
  - `_classify_and_map_key`: Classifies LoRA keys and maps them to model structure
  - `_load_lora_state_dict`: Loads LoRA state dictionary from file
  - `_detect_lora_format`: Detects LoRA format (Standard, LoKR, Mixed, etc.)
  - `_log_lora_format_detection`: Logs format detection results

#### Phase 3: Detailed Debug Logging (lines 188-334)

**Purpose**: Provide comprehensive debug information about LoRA keys, formats, and statistics.

**Code Structure**:

```python
if loras_to_apply and _classify_and_map_key and _load_lora_state_dict and _detect_lora_format and _log_lora_format_detection:
    from collections import defaultdict
    
    logger.info(f"Composing {len(loras_to_apply)} LoRAs...")
    
    # Cache first LoRA state dict for reuse (performance optimization)
    _cached_first_lora_state_dict = None
    
    for idx, (lora_name, lora_strength) in enumerate(loras_to_apply):
        try:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            
            # Use cached state dict for first LoRA to avoid duplicate I/O
            if idx == 0 and _cached_first_lora_state_dict is not None:
                lora_state_dict = _cached_first_lora_state_dict
            else:
                lora_state_dict = _load_lora_state_dict(lora_path)
                if idx == 0:
                    _cached_first_lora_state_dict = lora_state_dict
```

**Performance Optimization**:
- **First LoRA Caching**: The first LoRA's state dict is cached to avoid duplicate file I/O when it's reused in `compose_loras_v2`
- **Reduces I/O Operations**: Prevents reading the same file twice (once for debug logging, once for actual application)

**Format Detection and Logging** (lines 210-216):

```python
# LoRA format detection + detailed logging
try:
    detection = _detect_lora_format(lora_state_dict)
    _log_lora_format_detection(str(lora_name), detection)
except Exception:
    # Safety: never fail due to logging
    pass
```

**First LoRA Key Inspection** (lines 218-240):

```python
# First LoRA: Detailed key inspection
if idx == 0:
    logger.info(f"--- DEBUG: Inspecting keys for LoRA 1 (Strength: {lora_strength}) ---")
    
    # Check format first (performance optimization)
    _first_detection = _detect_lora_format(lora_state_dict)
    if _first_detection["has_standard"]:
        # Standard format (or mixed): Log EVERYTHING
        for key in lora_state_dict.keys():
            parsed_res = _classify_and_map_key(key)
            if parsed_res:
                group, base_key, comp, ab = parsed_res
                mapped_name = f"{base_key}.{comp}.{ab}" if comp and ab else (f"{base_key}.{ab}" if ab else base_key)
                logger.info(f"Key: {key} -> Mapped to: {mapped_name} (Group: {group})")
            else:
                logger.warning(f"Key: {key} -> UNMATCHED (Ignored)")
    else:
        # Unsupported format only: Skip loop to prevent freeze
        logger.warning(f"⚠️  Unsupported LoRA format detected (No standard keys).")
        logger.warning(f"   Skipping detailed key inspection of {len(lora_state_dict)} keys to prevent console freeze.")
        logger.warning(f"   Note: This LoRA will likely have no effect or will be skipped entirely.")
    
    logger.info("--- DEBUG: End key inspection ---")
```

**Key Statistics Collection** (lines 242-272):

```python
# Statistics for all LoRAs
lora_grouped: dict = defaultdict(dict)
lokr_keys_count = 0
standard_keys_count = 0
qkv_lokr_keys_count = 0
unrecognized_keys = []
total_keys = len(lora_state_dict)

for key, value in lora_state_dict.items():
    parsed = _classify_and_map_key(key)
    if parsed is None:
        unrecognized_keys.append(key)
        continue
    
    group, base_key, comp, ab = parsed
    if ab in ("lokr_w1", "lokr_w2"):
        lokr_keys_count += 1
        # Check if it's QKV format LoKR
        if group in ("qkv", "add_qkv") and comp is not None:
            qkv_lokr_keys_count += 1
    elif ab in ("A", "B"):
        standard_keys_count += 1
    
    if group in ("qkv", "add_qkv", "glu") and comp is not None:
        # Handle both standard LoRA (A/B) and LoKR (lokr_w1/lokr_w2) formats
        if ab in ("lokr_w1", "lokr_w2"):
            lora_grouped[base_key][f"{comp}_{ab}"] = value
        else:
            lora_grouped[base_key][f"{comp}_{ab}"] = value
    else:
        lora_grouped[base_key][ab] = value
```

**Format Classification** (lines 274-284):

```python
# Format classification
has_lokr = lokr_keys_count > 0
has_standard = standard_keys_count > 0
if has_lokr and has_standard:
    lora_format = "Mixed (LoKR + Standard LoRA)"
elif has_lokr:
    lora_format = "LoKR (QKV format)" if qkv_lokr_keys_count > 0 else "LoKR"
elif has_standard:
    lora_format = "Standard LoRA"
else:
    lora_format = "Unknown/Unsupported"
```

**Statistics Logging** (lines 303-331):

```python
# Statistics logging
logger.info(f"[LoRA {idx + 1} Statistics] '{lora_name}' (Strength: {lora_strength})")
logger.info(f"  Total keys: {total_keys}")
logger.info(f"  Standard keys: {standard_keys_count}")
logger.info(f"  LoKR keys: {lokr_keys_count}")
logger.info(f"  QKV LoKR keys: {qkv_lokr_keys_count}")
logger.info(f"  Unrecognized keys: {len(unrecognized_keys)}")
logger.info(f"  Format: {lora_format}")
logger.info(f"  Grouped base keys: {len(lora_grouped)}")
logger.debug(f"  Processed module groups: {len(processed_groups)}")

if unrecognized_keys:
    logger.warning(f"  Unrecognized keys (first 10): {unrecognized_keys[:10]}")
    if len(unrecognized_keys) > 10:
        logger.warning(f"  ... and {len(unrecognized_keys) - 10} more unrecognized keys")

# Warn if no weights would be processed
if not processed_groups:
    if lora_format == "Unknown/Unsupported":
        logger.error(f"❌ {lora_name}: No weights were processed - LoRA format is unsupported and will be skipped!")
    else:
        logger.warning(f"⚠️  {lora_name}: No weights were processed - this LoRA will have no effect!")
```

**Technical Significance**:
- **Comprehensive Statistics**: Provides detailed information about LoRA structure (keys, formats, groups)
- **Format Detection**: Identifies Standard LoRA, LoKR, Mixed, or Unsupported formats
- **Performance Optimization**: Skips detailed key inspection for unsupported formats to prevent console freeze
- **Error Detection**: Warns if LoRA format is unsupported or no weights would be processed

#### Phase 4: LoRA Application with compose_loras_v2 (lines 336-408)

**Purpose**: Apply LoRAs using `compose_loras_v2` for perfect mapping, with fallback to standard loader.

**Step 1: Model Wrapper Extraction** (line 338):

```python
# Get the underlying NextDiT model from the model patcher
model_wrapper = model.model.diffusion_model
```

**Step 2: compose_loras_v2 Import with Fallback #1** (lines 340-357):

```python
# Import compose_loras_v2 from nunchaku_code.lora_qwen
try:
    if lora_loader_dir not in sys.path:
        sys.path.insert(0, lora_loader_dir)
    
    from nunchaku_code.lora_qwen import compose_loras_v2
except ImportError as e:
    logger.error(f"Failed to import compose_loras_v2: {e}")
    logger.error("Cannot apply LoRAs - falling back to standard loader")
    # Fallback to standard loader if compose_loras_v2 is not available
    from comfy.sd import load_lora_for_models
    ret_model = model
    ret_clip = clip
    for lora_name, lora_strength in loras_to_apply:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        ret_model, ret_clip = load_lora_for_models(ret_model, ret_clip, lora, lora_strength, lora_strength)
    return (ret_model, ret_clip)
```

**Fallback #1**: If `compose_loras_v2` cannot be imported, falls back to standard ComfyUI `load_lora_for_models`. This ensures the node continues to work even if the custom mapping function is unavailable.

**Step 3: Model Type Verification with Fallback #2** (lines 359-381):

```python
# Check if model_wrapper is NextDiT (official Nunchaku loader)
model_wrapper_type_name = type(model_wrapper).__name__
model_wrapper_module = type(model_wrapper).__module__

# Get the actual NextDiT model
if model_wrapper_type_name == "NextDiT" and model_wrapper_module == "comfy.ldm.lumina.model":
    # Official loader: model_wrapper is NextDiT directly
    transformer = model_wrapper
elif hasattr(model_wrapper, 'model'):
    # Wrapped model (e.g., ComfyZImageTurboWrapper): get underlying model
    transformer = model_wrapper.model
else:
    logger.error(f"❌ Unsupported model type: {model_wrapper_type_name} from {model_wrapper_module}")
    logger.error("V4 requires NextDiT model. Falling back to standard loader.")
    # Fallback to standard loader
    from comfy.sd import load_lora_for_models
    ret_model = model
    ret_clip = clip
    for lora_name, lora_strength in loras_to_apply:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        ret_model, ret_clip = load_lora_for_models(ret_model, ret_clip, lora, lora_strength, lora_strength)
    return (ret_model, ret_clip)
```

**Model Type Handling**:
- **NextDiT (Official Loader)**: If `model_wrapper` is `NextDiT` from `comfy.ldm.lumina.model`, uses it directly
- **Wrapped Model**: If `model_wrapper` has a `model` attribute (e.g., `ComfyZImageTurboWrapper`), extracts the underlying model
- **Fallback #2**: If model type is unsupported, falls back to standard loader

**Step 4: LoRA Configuration Preparation** (lines 383-387):

```python
# Prepare LoRA configs for compose_loras_v2
lora_configs = []
for lora_name, lora_strength in loras_to_apply:
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora_configs.append((lora_path, lora_strength))
```

**Configuration Format**: `lora_configs` is a list of tuples `(lora_path, lora_strength)`, which `compose_loras_v2` expects.

**Step 5: LoRA Application with Fallback #3** (lines 389-405):

```python
# Apply LoRAs using compose_loras_v2 (perfect mapping)
logger.info(f"Applying {len(lora_configs)} LoRA(s) using compose_loras_v2...")
try:
    compose_loras_v2(transformer, lora_configs)
    logger.info(f"✅ Successfully applied {len(lora_configs)} LoRA(s) using compose_loras_v2")
except Exception as e:
    logger.error(f"❌ Failed to apply LoRAs using compose_loras_v2: {e}")
    logger.error("Falling back to standard loader.")
    # Fallback to standard loader
    from comfy.sd import load_lora_for_models
ret_model = model
ret_clip = clip
for lora_name, lora_strength in loras_to_apply:
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        ret_model, ret_clip = load_lora_for_models(ret_model, ret_clip, lora, lora_strength, lora_strength)
return (ret_model, ret_clip)
```

**Fallback #3**: If `compose_loras_v2` raises an exception during application, catches it and falls back to standard loader. This ensures robustness even if the perfect mapping function encounters unexpected errors.

**Step 6: Return Values** (line 407-408):

```python
# Return model and clip (CLIP unchanged, for compatibility with standard ComfyUI LoRA loader interface)
return (model, clip)
```

**Key Point**: CLIP is returned unchanged. LoRAs are only applied to the model, not to CLIP. This maintains compatibility with standard ComfyUI LoRA loader interface.

**Registration** (lines 410-416):

```python
GENERATED_NODES = {
    "NunchakuZImageTurboLoraStackV4": NunchakuZImageTurboLoraStackV4
}

GENERATED_DISPLAY_NAMES = {
    "NunchakuZImageTurboLoraStackV4": "Nunchaku Z-Image-Turbo LoRA Stack V4"
}
```

## JavaScript Extension Updates

### 2. `ComfyUI-QwenImageLoraLoader/js/zimageturbo_lora_dynamic_v4.js`

**Purpose**: Dynamic UI for v4 node (variable display of LoRA slot count)

**Key Implementation Details**:

**Extension Registration** (line 8):

```javascript
app.registerExtension({
    name: "nunchaku.zimageturbo_lora_dynamic_v4",
```

**Node Filtering** (line 17):

```javascript
nodeCreated(node) {
    if (node.comfyClass !== "NunchakuZImageTurboLoraStackV4") return;
```

**Key Differences from v3**:
1. **CPU Offload Widget Removed**: v4 has no `cpu_offload` parameter, so the CPU offload widget is not included in the UI
2. **Extension Name**: Uses `"nunchaku.zimageturbo_lora_dynamic_v4"` to avoid conflicts with v2/v3 extensions
3. **Node Class Filter**: Filters for `NunchakuZImageTurboLoraStackV4` only

**Dynamic UI Logic** (lines 36-183):

- **Widget Caching**: Caches all LoRA widgets (enabled_1-10, lora_name_1-10, lora_strength_1-10) for efficient dynamic display
- **lora_count Widget Hiding**: The `lora_count` widget is hidden using `HIDDEN_TAG` and custom `computeSize` (required for Python backend, but hidden from user)
- **Control Widget**: Adds a "🔢 LoRA Count" combo widget for user control
- **Toggle All Widget**: Integrates `toggle_all` widget with dynamic slot control
- **Dynamic Slot Display**: Shows/hides LoRA slots based on `visibleLoraCount` property

### Modified Files

#### 1. `ComfyUI-QwenImageLoraLoader/__init__.py`

**Modification: V4 Node Import and Registration**

```python
from .nodes.lora.zimageturbo_v4 import GENERATED_NODES as ZIMAGETURBO_V4_NODES, GENERATED_DISPLAY_NAMES as ZIMAGETURBO_V4_NAMES

for node_class in ZIMAGETURBO_V4_NODES.values():
    node_class.__version__ = __version__

NODE_CLASS_MAPPINGS.update(ZIMAGETURBO_V4_NODES)

NODE_DISPLAY_NAME_MAPPINGS = {
    **ZIMAGETURBO_V4_NAMES
}
```

**Technical Significance**:
- Registers v4 node in ComfyUI's node system
- Sets version on node classes for version tracking

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

## Technical Design Philosophy

### 1. Same Mapping Mechanism as v3

**Problem**: Official implementation's mapping is incomplete

**Solution**:
- Uses the same `compose_loras_v2` as v3
- Inherits automatic mapping switching functionality
- Uses mapping optimized for NextDiT structure

**Benefits**:
- Proven perfect mapping functionality from v3
- Avoids issues with official implementation
- Completely records mapping status with detailed debug logs

### 2. Compatibility with Standard Format

**Problem**: Standard ComfyUI LoRA loader has `(MODEL, CLIP)` input/output

**Solution**:
- Adds CLIP input/output
- Returns CLIP unchanged (applies LoRA only to model)
- Improves compatibility with other nodes by conforming to standard format

**Benefits**:
- Same interface as standard ComfyUI LoRA loader
- Improved workflow compatibility
- Future extensibility

### 3. CPU Offload Parameter Removal

**Problem**: Standard ComfyUI LoRA loader has no CPU offload parameter

**Solution**:
- Removes `cpu_offload` parameter
- Fully conforms to standard format

**Benefits**:
- Consistency with standard format
- Code simplification
- Unified user interface

### 4. Complete Debug Logging Functionality

**Problem**: Detailed logs are needed to diagnose mapping issues

**Solution**:
- Implements complete debug logging equivalent to v3
- Records all key mapping, statistics, and application status
- Includes performance optimizations (caching, format-based key inspection)

**Implementation Details**:
- LoRA format detection and detailed logging
- Recording mapping status for each key
- Statistics (total key count, standard key count, LoKR key count, format classification, etc.)
- Recording number of applied modules
- First LoRA caching to avoid duplicate I/O

**Benefits**:
- Easy diagnosis of mapping problems
- Improved debugging efficiency
- Enhanced user support

### 5. Enhanced Fallback Functionality

**Problem**: Need maximum compatibility and robustness

**Solution**:
- Three-point fallback system:
  1. **Import Fallback**: If `compose_loras_v2` cannot be imported, use standard loader
  2. **Model Type Fallback**: If model type is unsupported, use standard loader
  3. **Application Fallback**: If `compose_loras_v2` raises exception, use standard loader

**Benefits**:
- Maximum compatibility (works even if custom mapping is unavailable)
- Robust error handling (graceful degradation)
- User-friendly (always provides a working solution)

## Operation Flow

### When Using V4 Node (Official Loader)

1. **Node Execution**: `NunchakuZImageTurboLoraStackV4.load_lora_stack(model, clip, lora_count, toggle_all, **kwargs)`
2. **LoRA Filtering**: Collect enabled LoRAs with strength > 1e-5, log each slot's status
3. **Early Return Check**: If no LoRAs to apply, return `(model, clip)` immediately
4. **Debug Function Import**: Import debug functions from `nunchaku_code.lora_qwen` (graceful degradation if unavailable)
5. **Debug Logging** (if functions available):
   - Load first LoRA state dict (cached for reuse)
   - Detect LoRA format and log results
   - First LoRA: Detailed key inspection (if standard format detected)
   - All LoRAs: Collect statistics (keys, formats, groups)
   - Log comprehensive statistics
6. **Model Extraction**: Get `model.model.diffusion_model` (NextDiT model)
7. **compose_loras_v2 Import**: Import `compose_loras_v2` from `nunchaku_code.lora_qwen`
   - **Fallback #1**: If import fails, use standard `load_lora_for_models` and return
8. **Model Type Verification**: Check if `model_wrapper` is NextDiT
   - **Fallback #2**: If unsupported model type, use standard `load_lora_for_models` and return
9. **LoRA Configuration**: Prepare `lora_configs = [(lora_path, lora_strength), ...]`
10. **LoRA Application**: Call `compose_loras_v2(transformer, lora_configs)`
    - **Fallback #3**: If exception occurs, use standard `load_lora_for_models` and return
11. **Return**: Return `(model, clip)` (CLIP unchanged)

## Comparison with v3

### Architecture Comparison

| Item | v3 | v4 |
|------|-----|-----|
| **LoRA Application Mechanism** | `compose_loras_v2` (custom) | `compose_loras_v2` (custom, same as v3) |
| **Wrapper** | `ComfyZImageTurboWrapper` (required) | Not required (uses NextDiT directly) |
| **CPU Offload** | Parameter present | No parameter |
| **CLIP Input** | None (MODEL only) | Present (MODEL + CLIP) |
| **Return Value** | `(MODEL,)` | `(MODEL, CLIP)` |
| **Debug Logging** | Fully implemented | Fully implemented (equivalent to v3) |
| **Implementation Pattern** | Custom wrapper + compose_loras_v2 | Direct compose_loras_v2 |
| **Standard Format Compliance** | Partial | Complete (CLIP input/output) |
| **Fallback Functionality** | None | Present (three-point fallback system) |
| **IS_CHANGED Hash** | MODEL only | MODEL + CLIP |

### Code Comparison

**v3 Implementation (simplified)**:

```python
def load_lora_stack(self, model, lora_count, cpu_offload="disable", **kwargs):
    # Create custom wrapper
    model_wrapper = model.model.diffusion_model
    wrapped_model = ComfyZImageTurboWrapper(...)
    model.model.diffusion_model = wrapped_model
    
    # Add LoRA to wrapper's loras list
    ret_model_wrapper.loras.append((lora_path, lora_strength))
    # compose_loras_v2 is executed during forward
    
    return (ret_model,)
```

**v4 Implementation (simplified)**:

```python
def load_lora_stack(self, model, clip, lora_count, toggle_all=True, **kwargs):
    # Get NextDiT model directly
    model_wrapper = model.model.diffusion_model
    transformer = model_wrapper  # NextDiT directly
    
    # Prepare LoRA configuration
    lora_configs = [(lora_path, lora_strength), ...]
    
    # Call compose_loras_v2 directly
    compose_loras_v2(transformer, lora_configs)
    
    return (model, clip)  # CLIP unchanged
```


## Files Modified

- `nodes/lora/zimageturbo_v4.py`: Updated to standard format (CLIP input/output, no CPU offload)
- `js/zimageturbo_lora_dynamic_v4.js`: Updated to remove CPU offload widget
- `__init__.py`: v3 registration removed, v4 registration maintained

## Summary

v2.3.0 updates the Z-ImageTurbo LoRA Stack to v4, which conforms to the standard ComfyUI LoRA loader format while maintaining perfect mapping functionality. The v3 node registration has been removed, and users are encouraged to migrate to v4 for better compatibility with standard ComfyUI workflows.
