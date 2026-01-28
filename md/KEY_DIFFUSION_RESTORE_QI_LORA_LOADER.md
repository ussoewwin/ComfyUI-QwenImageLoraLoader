# Key Diffusion Restoration Fix for QI LoRA Loader — Complete Explanation

## 1. Overview

This document summarizes the modifications made to add **Key Diffusion (key analysis logging)** equivalent to Z-Image Turbo (ZIT) to each Qwen Image LoRA node in **ComfyUI-QwenImageLoraLoader**.

- **What is Key Diffusion**: Analyzes the state_dict keys of the first LoRA and outputs debug logs to the console in the format:  
  `Key: <original key> -> Mapped to: <mapped target> (Group: <group>)`
- **Reference Implementation**: Lines 219–241 in `nodes/lora/zimageturbo_v4.py` (First LoRA: Detailed key inspection).
- **Approach**: Reproduce the exact logic from the above block, with ZIT using `logger` and QI side using **`print()`** for output (to ensure display in ComfyUI console).
- **Log ON/OFF (Environment Variable)**: Controlled by `nunchaku_log`. **Default is muted**, and Key Diffusion / `[APPLY]` / `[AWQ_MOD]` are only output when `nunchaku_log=1`.

---

## 2. Modified Files List

| File | Node/Method | Insertion Location |
|------|-------------|-------------------|
| `nodes/lora/qwenimage.py` | `NunchakuQwenImageLoraLoader.load_lora` | Insert block immediately before `model_wrapper = model.model.diffusion_model` |
| `nodes/lora/qwenimage.py` | `NunchakuQwenImageLoraStack.load_lora_stack` (Legacy) | Same as above |
| `nodes/lora/qwenimage_v2.py` | `NunchakuQwenImageLoraStackV2.load_lora_stack` | Same as above |
| `nodes/lora/qwenimage_v3.py` | `NunchakuQwenImageLoraStackV3.load_lora_stack` | Same as above |

**Important**: The editing target is  
`ComfyUI\custom_nodes\ComfyUI-QwenImageLoraLoader\`  
and below. Do not edit paths with the same name in other directories.

---

## 3. Reference: Z-Image Turbo v4 Key Analysis Block

**File**: `nodes/lora/zimageturbo_v4.py`  
**Lines**: 219–241

In ZIT, the following block is executed when `idx == 0` in the for loop within `load_lora_stack` (using `logger.info` / `logger.warning`).

```python
# First LoRA: Detailed key inspection (same as v3 compose_loras_v2)
if idx == 0:
    logger.info(f"--- DEBUG: Inspecting keys for LoRA 1 (Strength: {lora_strength}) ---")

    _first_detection = _detect_lora_format(lora_state_dict)
    if _first_detection["has_standard"]:
        for key in lora_state_dict.keys():
            parsed_res = _classify_and_map_key(key)
            if parsed_res:
                group, base_key, comp, ab = parsed_res
                mapped_name = f"{base_key}.{comp}.{ab}" if comp and ab else (f"{base_key}.{ab}" if ab else base_key)
                logger.info(f"Key: {key} -> Mapped to: {mapped_name} (Group: {group})")
            else:
                logger.warning(f"Key: {key} -> UNMATCHED (Ignored)")
    else:
        logger.warning("⚠️  Unsupported LoRA format detected (No standard keys).")
        logger.warning(f"   Skipping detailed key inspection of {len(lora_state_dict)} keys to prevent console freeze.")
        logger.warning("   Note: This LoRA will likely have no effect or will be skipped entirely.")

    logger.info("--- DEBUG: End key inspection ---")
```

On the QI side, this logic is **exactly** followed, with only the output changed to `print()`.

---

## 4. Common Code Block Inserted

The code inserted in each node consists of the following two stages:

1. **Preparation**: `lora_loader_dir` · import · get first LoRA path · get state_dict with `_load_lora_state_dict`  
2. **Key Diffusion**: `_detect_lora_format` → `_log_lora_format_detection` → key analysis equivalent to ZIT 219–240 (`print` version)

Variable name differences are as follows:

- **Loader (single)**: Uses `lora_name`, `lora_strength` as-is.  
  Since only the first LoRA is handled, `loras_to_apply` does not exist.
- **Stack (Legacy / V2 / V3)**: Gets the first file name and strength with `_fn, _fs = loras_to_apply[0]`,  
  and uses `_fn`, `_fs` as the file name and strength.

Below, the "contents of the block" inserted are shown in a general form (using Stack variable names `_fn`, `_fs`).

```python
# ZIT zimageturbo_v4.py lines 201-241: load -> detect -> log -> key inspection (exact same block, logger -> print)
import sys
import importlib.util
lora_loader_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if lora_loader_dir not in sys.path:
    sys.path.insert(0, lora_loader_dir)
from nunchaku_code.lora_qwen import _classify_and_map_key, _load_lora_state_dict, _detect_lora_format, _log_lora_format_detection
# Stack case: _fn, _fs = loras_to_apply[0]
# Loader case: lora_name, lora_strength already exist
lora_path = folder_paths.get_full_path_or_raise("loras", _fn)   # lora_name for Loader
lora_state_dict = _load_lora_state_dict(lora_path)
if lora_state_dict:
    detection = _detect_lora_format(lora_state_dict)
    _log_lora_format_detection(str(_fn), detection)              # str(lora_name) for Loader
    # First LoRA: Detailed key inspection (same as zimageturbo_v4.py lines 219-240)
    print(f"--- DEBUG: Inspecting keys for LoRA 1 (Strength: {_fs}) ---")   # lora_strength for Loader
    _first_detection = _detect_lora_format(lora_state_dict)
    if _first_detection["has_standard"]:
        for key in lora_state_dict.keys():
            parsed_res = _classify_and_map_key(key)
            if parsed_res:
                group, base_key, comp, ab = parsed_res
                mapped_name = f"{base_key}.{comp}.{ab}" if comp and ab else (f"{base_key}.{ab}" if ab else base_key)
                print(f"Key: {key} -> Mapped to: {mapped_name} (Group: {group})")
            else:
                print(f"Key: {key} -> UNMATCHED (Ignored)")
    else:
        print("⚠️  Unsupported LoRA format detected (No standard keys).")
        print(f"   Skipping detailed key inspection of {len(lora_state_dict)} keys to prevent console freeze.")
        print("   Note: This LoRA will likely have no effect or will be skipped entirely.")
    print("--- DEBUG: End key inspection ---")

model_wrapper = model.model.diffusion_model
# Existing spec / wrapper loading etc. continues below
```

---

## 5. Meaning and Role of the Fix (Line by Line)

| Process | Meaning/Role |
|---------|--------------|
| `lora_loader_dir = os.path.dirname(...)` | Finds the custom node root to enable import of `nunchaku_code.lora_qwen`. |
| `sys.path.insert(0, lora_loader_dir)` | Adds the above root to the path. Supports ComfyUI custom node placement. |
| `from nunchaku_code.lora_qwen import _classify_and_map_key, _load_lora_state_dict, _detect_lora_format, _log_lora_format_detection` | Uses the same helpers as ZIT. Performs key classification, LoRA loading, format detection, and format logging. |
| `folder_paths.get_full_path_or_raise("loras", ...)` | Gets the full path from ComfyUI's "loras" folder. Exits with exception on failure. |
| `_load_lora_state_dict(lora_path)` | Gets LoRA state_dict using the same loading logic as ZIT. |
| `_detect_lora_format(lora_state_dict)` | Detects presence of Standard / LoKR / LoHa / IA3, etc. |
| `_log_lora_format_detection(...)` | Outputs detection results in existing format (using logger). |
| `print("--- DEBUG: Inspecting keys for LoRA 1 (Strength: ...) ---")` | Key Diffusion start line. Reproduces ZIT's `logger.info` content with `print`. |
| `_first_detection["has_standard"]` | Determines if standard LoRA keys (lora_up/down, lora_A/B, etc.) are included. |
| `for key in lora_state_dict.keys():` + `_classify_and_map_key` | Interprets each key with Qwen Image mapping and decomposes into `group`, `base_key`, `comp`, `ab`. |
| `print(f"Key: {key} -> Mapped to: {mapped_name} (Group: {group})")` | Key Diffusion body. Outputs key→mapped target→group to console. |
| `print(f"Key: {key} -> UNMATCHED (Ignored)")` | Logs keys that were not mapped. |
| 3 lines of `print("⚠️  Unsupported...")` etc. | For unsupported formats. Key enumeration is not performed as it can freeze with many keys. |
| `print("--- DEBUG: End key inspection ---")` | Key Diffusion end line. |
| `model_wrapper = model.model.diffusion_model` | As before, touches the model **after** key analysis. |

**Reason for Using `print()`**  
In some ComfyUI environments, `logger.info` may not appear in the console due to logging level or handler settings, so the Key Diffusion part is unified with **`print()`** to ensure it appears in the console.

---

## 6. Insertion Locations and Variable Correspondence by File

### 6.1 `nodes/lora/qwenimage.py` — Loader (`load_lora`)

- **Insertion Location**: Immediately after `if abs(lora_strength) < 1e-5: return (model,)`, and  
  **immediately before `model_wrapper = model.model.diffusion_model`**.
- **Variables**:  
  - Path retrieval: `lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)`  
  - Strength and name: Uses `lora_strength`, `lora_name` as-is.  
- **Code Range**: Around lines 95–123 in the same file (varies by environment).

### 6.2 `nodes/lora/qwenimage.py` — Legacy Stack (`load_lora_stack`)

- **Insertion Location**: Immediately after `if not loras_to_apply: return (model,)`, and  
  **immediately before `model_wrapper = model.model.diffusion_model`**.
- **Variables**:  
  - Gets first file name and strength with `_fn, _fs = loras_to_apply[0]`.  
  - `lora_path = folder_paths.get_full_path_or_raise("loras", _fn)`.  
  - Strength is `_fs`, display name is `str(_fn)`.  
- **Code Range**: Around lines 309–339 in the same file.

### 6.3 `nodes/lora/qwenimage_v2.py` — V2 Stack (`load_lora_stack`)

- **Insertion Location**: Immediately after `if not loras_to_apply: return (model,)`, and  
  **immediately before `model_wrapper = model.model.diffusion_model`**.
- **Variables**: Same as Legacy with `_fn`, `_fs`.  
- **Code Range**: Around lines 136–165 in the same file.

### 6.4 `nodes/lora/qwenimage_v3.py` — V3 Stack (`load_lora_stack`)

- **Insertion Location**: Same as Legacy / V2.  
- **Variables**: Same with `_fn`, `_fs`.  
- **Code Range**: Around lines 175–203 in the same file.  
- **Note**: If `import sys` is already done at the top, only `import importlib.util` needs to be added in the key analysis block.

---

## 7. Execution Order Summary

In any node, processing proceeds in the following order:

1. Check strength and `loras_to_apply` (existing early return).
2. **Key Diffusion processing (added this time)**  
   - import · path resolution  
   - `_load_lora_state_dict(lora_path)`  
   - `_detect_lora_format` → `_log_lora_format_detection`  
   - `--- DEBUG: Inspecting keys for LoRA 1 (Strength: ...) ---` ~  
     `Key: ... -> Mapped to: ... (Group: ...)` or UNMATCHED/unsupported format warning ~  
     `--- DEBUG: End key inspection ---`
3. `model_wrapper = model.model.diffusion_model` and subsequent existing processing (wrapping, compose, etc.).

In terms of "executing key analysis for only the first LoRA before touching the model," this aligns with ZIT's `idx == 0` block.

---

## 8. APIs Used from nunchaku_code.lora_qwen

The following 4 are used similarly to ZIT:

- **`_load_lora_state_dict(lora_path)`**  
  Loads state_dict from LoRA file (or path). Supports both safetensors and torch.
- **`_detect_lora_format(state_dict)`**  
  Returns a dict indicating presence of Standard / LoKR / LoHa / IA3, etc.  
  Uses `["has_standard"]` for Key Diffusion branching.
- **`_log_lora_format_detection(name, detection)`**  
  Outputs the above detection results in existing format.
- **`_classify_and_map_key(key)`**  
  Interprets the key with Qwen Image key mapping and  
  returns a tuple `(group, base_key, comp, ab)`, or `None` if unsupported.

These are defined in the existing `nunchaku_code/lora_qwen.py`, and **no new functions were added**. The logic from ZIT lines 219–241 is "copied as-is" to each QI node, with only the output destination changed to `print()`.

---

## 9. Summary

- **Purpose**: Output the same Key Diffusion (key analysis logging) as ZIT to the console in QI LoRA Loader (single, Legacy, V2, V3).
- **Reference**: `zimageturbo_v4.py` lines 219–241.
- **Modified Files**:  
  `qwenimage.py` (Loader / Legacy), `qwenimage_v2.py`, `qwenimage_v3.py`  
  under `ComfyUI\custom_nodes\ComfyUI-QwenImageLoraLoader\nodes\lora\`.
- **Content**: In all cases, insert the same key analysis block as ZIT **immediately before `model_wrapper = model.model.diffusion_model`**,  
  and output with `print()` instead of `logger`.
- **Dependencies**: Uses  
  `_classify_and_map_key`, `_load_lora_state_dict`, `_detect_lora_format`, `_log_lora_format_detection`  
  from `nunchaku_code.lora_qwen`. No new functions were added.

---

## 10. Addition: Log Muting via `nunchaku_log` (Default OFF → ON only when `nunchaku_log=1`)

After Key Diffusion restoration, to prevent the console from being flooded with logs, **log output is controlled collectively via the environment variable `nunchaku_log`**.

### 10.1 Specification

- **Default (unset)**: Logs are muted (not output)
- **`nunchaku_log=1`**: Logs are output

### 10.2 Muted Targets

The following logs only appear when `nunchaku_log=1` (not output when unset):

- **Key Diffusion** (`--- DEBUG: Inspecting keys for LoRA 1 ...` ~ `--- DEBUG: End key inspection ---`)
- **Key application logs** (`[APPLY] LoRA applied to: ...`)
- **AWQ Modulation logs** (`[AWQ_MOD] ... Storing LoRA weights for manual Planar injection`)

### 10.3 Implementation Points (Where It's Checked)

- `NUNCHAKU_LOG_ENABLED` is defined in `nunchaku_code/lora_qwen.py` and evaluates `nunchaku_log`.
- Each QI node (Loader / Legacy / V2 / V3) adds `from nunchaku_code.lora_qwen import NUNCHAKU_LOG_ENABLED` and guards the entire Key Diffusion block with `if lora_state_dict and NUNCHAKU_LOG_ENABLED:`.
- `[APPLY]` and `[AWQ_MOD]` are guarded by `if NUNCHAKU_LOG_ENABLED:` on the `nunchaku_code/lora_qwen.py` side.

This document is saved as a complete explanation of the "target files, inserted code, and meaning" of the modifications made to restore Key Diffusion.
