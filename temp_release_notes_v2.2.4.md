## Overview

Qwen Image models have modulation layers `img_mod.1` and `txt_mod.1` in each transformer block. These layers are linear layers that generate shift/scale/gate parameters for AdaNorm.

In Nunchaku's AWQ quantized models, these layers are implemented as quantized linear layers called `AWQW4A16Linear`. Standard LoRA application methods (directly modifying weights) cannot be applied to quantized layers, so we adopted a method that adds LoRA in the forward path.

However, as a result of implementation and testing, applying LoRA to `img_mod.1` and `txt_mod.1` caused image noise. Therefore, we implemented a safety measure to **skip LoRA application to these layers by default**.

## Detailed Code Implementation Explanation

### 1. Environment Variable Loading

**File**: `nunchaku_code/lora_qwen.py`  
**Line Numbers**: 23-28

```python
# Safety switch:
# QwenImage's modulation linears (`img_mod.1` / `txt_mod.1`) are extremely sensitive because their
# output is reshaped into shift/scale/gate parameters. With AWQ quantization (AWQW4A16Linear),
# applying LoRA here often results in severe noise. Default to skipping these two layers.
# Users can override by setting env var `QWENIMAGE_LORA_APPLY_AWQ_MOD=1`.
_APPLY_AWQ_MOD = str(os.getenv("QWENIMAGE_LORA_APPLY_AWQ_MOD", "0")).strip().lower() in ("1", "true", "yes", "y", "on")
```

**Explanation**:

`_APPLY_AWQ_MOD` is a global variable that controls whether to apply LoRA to AWQ modulation layers.

1. `os.getenv("QWENIMAGE_LORA_APPLY_AWQ_MOD", "0")` reads the environment variable. The default value is "0" (disabled).
2. `str(...).strip().lower()` normalizes the string (removes leading/trailing whitespace and converts to lowercase).
3. `in ("1", "true", "yes", "y", "on")` checks if it is a valid value (values considered true).
4. As a result, if the environment variable is not set, or has values like "0" / "false" / "no", it becomes `False` and is skipped by default.

**Debug Log Display**:

This variable itself is not displayed in logs, but logs are output by the skip logic described later.

### 2. AWQW4A16Linear Module Detection

**File**: `nunchaku_code/lora_qwen.py`  
**Line Numbers**: 1385-1392

```python
is_awq_w4a16 = (
    module.__class__.__name__ == "AWQW4A16Linear"
    and hasattr(module, "qweight")
    and hasattr(module, "wscales")
    and hasattr(module, "wzeros")
    and hasattr(module, "in_features")
    and hasattr(module, "out_features")
)
```

**Explanation**:

Within the `compose_loras_v2` function, this determines whether each module is `AWQW4A16Linear`.

1. `module.__class__.__name__ == "AWQW4A16Linear"` checks the class name. We use class name instead of `isinstance()` because import paths may differ.
2. `hasattr(module, "qweight")` checks if the quantized weight tensor exists. In AWQ, weights are quantized to INT4 and stored in `qweight`.
3. `hasattr(module, "wscales")` and `hasattr(module, "wzeros")` check if quantization scales and zero points exist.
4. `hasattr(module, "in_features")` and `hasattr(module, "out_features")` check if standard linear layer attributes exist.

If all these conditions are met, `is_awq_w4a16 = True` and the module is treated as an AWQ quantized layer.

**Debug Log Display**:

This detection process itself is not displayed in logs, but logs are output by subsequent processing.

### 3. AWQ Modulation Layer Skip Logic

**File**: `nunchaku_code/lora_qwen.py`  
**Line Numbers**: 1394-1400

```python
# Default-skip AWQ modulation layers to avoid catastrophic noise.
if is_awq_w4a16 and (".img_mod.1" in resolved_name or ".txt_mod.1" in resolved_name) and not _APPLY_AWQ_MOD:
    logger.warning(
        f"[SKIP] {resolved_name}: AWQ modulation layer LoRA is disabled by default (prevents noise). "
        f"Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable."
    )
    continue
```

**Explanation**:

This processing skips LoRA application to AWQ quantized modulation layers (`img_mod.1` or `txt_mod.1`) by default.

When:
1. `is_awq_w4a16` is `True`, and
2. `resolved_name` contains `.img_mod.1` or `.txt_mod.1`, and
3. `_APPLY_AWQ_MOD` is `False` (environment variable is disabled)

it skips LoRA application and proceeds to the next module (`continue`).

**Debug Log Display**:

When skipped, a WARNING level log like the following is output:

```
[SKIP] transformer_blocks.0.img_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
[SKIP] transformer_blocks.0.txt_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
[SKIP] transformer_blocks.1.img_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
[SKIP] transformer_blocks.1.txt_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
...
```

This log is output for each `img_mod.1` and `txt_mod.1` in each transformer block (typically 0 to 59, 60 blocks total). That is, up to 120 skip logs may be output with default settings.

### 4. LoRA Application to AWQW4A16Linear

**File**: `nunchaku_code/lora_qwen.py`  
**Line Numbers**: 1018-1043

```python
elif module.__class__.__name__ == "AWQW4A16Linear" and hasattr(module, "qweight") and hasattr(module, "wscales") and hasattr(module, "wzeros"):
    if not hasattr(model, "_lora_slots"):
        model._lora_slots = {}

    # Save original forward once
    if not hasattr(module, "_lora_original_forward"):
        module._lora_original_forward = module.forward

    # Store LoRA weights on the module (already scaled/concatenated by caller)
    module._lora_A = A
    module._lora_B = B

    def _forward_with_lora(x: torch.Tensor) -> torch.Tensor:
        # Base quantized forward
        y = module._lora_original_forward(x)
        # LoRA branch (dtype/device should already match caller cast)
        # A: [rank, in], B: [out, rank]
        lora_out = (x @ module._lora_A.T) @ module._lora_B.T
        return y + lora_out.to(dtype=y.dtype, device=y.device)

    module.forward = _forward_with_lora

    # Track for reset
    model._lora_slots[module_name] = {
        "type": "awq_w4a16",
    }
```

**Explanation**:

This processing implements a method that adds LoRA terms in the forward path without modifying quantized weights when applying LoRA to AWQ quantized layers.

1. **LoRA Slot Initialization**: Initializes `model._lora_slots` if it doesn't exist. This dictionary is needed for resetting later.

2. **Original Forward Method Storage**: Saves the original `forward` method to `module._lora_original_forward`. This allows restoring the original behavior during reset. This processing runs only once (due to `hasattr` check).

3. **LoRA Weight Storage**: Stores LoRA A and B matrices to `module._lora_A` and `module._lora_B`. These weights have already been scaled and concatenated by the caller (`compose_loras_v2`).

4. **LoRA-Enabled Forward Implementation**: Defines an internal function `_forward_with_lora`. This function:
   - Executes normal forward with quantized weights via `module._lora_original_forward(x)`.
   - Calculates LoRA terms via `(x @ module._lora_A.T) @ module._lora_B.T`. Here, `A` is shape `[rank, in_features]`, `B` is shape `[out_features, rank]`. `A.T` is `[in_features, rank]`, `B.T` is `[rank, out_features]`, so `x @ A.T @ B.T` is `[batch, in_features] @ [in_features, rank] @ [rank, out_features] = [batch, out_features]`.
   - Adds LoRA terms to quantized forward results via `y + lora_out.to(dtype=y.dtype, device=y.device)`. `to(dtype=..., device=...)` aligns data type and device.

5. **Forward Method Replacement**: Replaces the module's forward method with the LoRA-enabled version via `module.forward = _forward_with_lora`.

6. **Slot Information Recording**: Records `{"type": "awq_w4a16"}` to `model._lora_slots[module_name]`. This allows identifying that this module is an AWQ quantized layer during reset.

**Debug Log Display**:

When this processing is executed, the following log is output by subsequent processing in the `compose_loras_v2` function (lines 1460-1461):

```
[APPLY] LoRA applied to: transformer_blocks.0.img_mod.1
```

However, with default settings (`_APPLY_AWQ_MOD = False`), `img_mod.1` and `txt_mod.1` are skipped, so this log is not output.

### 5. B Matrix Reordering for Modulation Layers

**File**: `nunchaku_code/lora_qwen.py`  
**Line Numbers**: 1417-1433

```python
# QwenImage modulation layers (img_mod.1 / txt_mod.1) are AWQ-quantized in Nunchaku
# and their output is *interleaved* as [B, dim, 6] flattened, then later reordered by:
#   view(B, dim, 6).transpose(1, 2).reshape(B, 6*dim)
#
# Most LoRAs are trained against the *standard* (non-interleaved) layout (6 blocks of dim).
# Because our AWQ LoRA path adds the LoRA output BEFORE that reorder, we must convert the
# LoRA "up" matrix B from standard->interleaved order (permute output channels).
if (".img_mod.1" in resolved_name) or (".txt_mod.1" in resolved_name):
    if B.ndim == 2 and (B.shape[0] % 6 == 0):
        dim = B.shape[0] // 6
        # standard [6, dim, r] -> interleaved [dim, 6, r] -> [6*dim, r]
        B = B.contiguous().view(6, dim, B.shape[1]).transpose(0, 1).reshape(B.shape[0], B.shape[1])
    else:
        logger.warning(
            f"{resolved_name}: expected mod up-matrix with out_features divisible by 6, got B{tuple(B.shape)}; "
            f"skipping mod-channel reorder"
        )
```

**Explanation**:

In Qwen Image's modulation layers, output is used as shift/scale/gate parameters (6 blocks total). In Nunchaku's implementation, this output is processed in "interleaved" format.

1. **Problem Background**: Most LoRAs are trained against the standard (non-interleaved) layout (6 blocks × dim). However, in the AWQ LoRA path, since LoRA output is added **before** the reordering process, the B matrix must be converted from standard to interleaved format.

2. **Shape Check**: `B.ndim == 2 and (B.shape[0] % 6 == 0)` checks if B is 2-dimensional and if `out_features` is divisible by 6. This assumes a 6-block structure (shift/scale/gate × 2).

3. **Reordering Processing**:
   - `dim = B.shape[0] // 6` calculates the dimension count for each block.
   - `B.contiguous().view(6, dim, B.shape[1])` reshapes B to shape `[6, dim, rank]`. `contiguous()` is needed to ensure contiguous memory layout.
   - `.transpose(0, 1)` converts to `[dim, 6, rank]`. This converts from standard format `[6, dim, rank]` to interleaved format `[dim, 6, rank]`.
   - `.reshape(B.shape[0], B.shape[1])` finally returns to shape `[6*dim, rank]`.

4. **Error Handling**: If B's shape differs from expectations (e.g., not divisible by 6), reordering is skipped and a warning log is output.

**Debug Log Display**:

When reordering is skipped (shape differs from expectations), a WARNING log like the following is output:

```
transformer_blocks.0.img_mod.1: expected mod up-matrix with out_features divisible by 6, got B(1234, 16); skipping mod-channel reorder
```

However, with default settings, `img_mod.1` and `txt_mod.1` are skipped, so this processing itself is not executed (only executed when enabled via environment variable).

### 6. LoRA Reset Processing (AWQ Quantized Layer Restoration)

**File**: `nunchaku_code/lora_qwen.py`  
**Line Numbers**: 1568-1581

```python
elif module_type == "awq_w4a16":
    # Restore original forward and remove attached LoRA tensors
    if hasattr(module, "_lora_original_forward"):
        try:
            module.forward = module._lora_original_forward
        except Exception:
            # Safety: never fail reset
            pass
    for attr in ("_lora_A", "_lora_B", "_lora_original_forward"):
        if hasattr(module, attr):
            try:
                delattr(module, attr)
            except Exception:
                pass
```

**Explanation**:

Reset processing for LoRA applied to AWQ quantized layers (restoring to original state) within the `reset_lora_v2` function.

1. **Forward Method Restoration**: Checks if the original forward method is saved via `hasattr(module, "_lora_original_forward")`. If saved, restores the original forward method via `module.forward = module._lora_original_forward`.

2. **LoRA Attribute Deletion**: Deletes each attribute `_lora_A`, `_lora_B`, `_lora_original_forward`. This restores the module to its original state.

3. **Error Handling**: Each operation is wrapped in `try-except` to ensure the reset process doesn't fail even if errors occur (as the comment "Safety: never fail reset" indicates, reset must always succeed).

**Debug Log Display**:

At the end of the `reset_lora_v2` function (line 1585), the following INFO level log is output:

```
All LoRA weights have been reset from the model.
```

This log is output after reset of all LoRA types (Nunchaku LoRA-ready, standard Linear, AWQ quantized layers) is completed.

## Log Appearance (Default Settings: Skip Enabled)

With default settings (`QWENIMAGE_LORA_APPLY_AWQ_MOD` not set or 0), logs are output as follows:

```
[SKIP] transformer_blocks.0.img_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
[SKIP] transformer_blocks.0.txt_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
[SKIP] transformer_blocks.1.img_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
[SKIP] transformer_blocks.1.txt_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
...
[SKIP] transformer_blocks.59.img_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
[SKIP] transformer_blocks.59.txt_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
```

For each `img_mod.1` and `txt_mod.1` in each transformer block (0 to 59, 60 blocks total), a WARNING level skip log is output. That is, up to 120 skip logs are output.

After that, a log of the number of applied modules is output:

```
Applied LoRA compositions to 480 modules.
```

This number 480 is the total number of modules (600) minus the number of skipped modules (120). Each transformer block typically has 10 LoRA application locations, but 2 of them (`img_mod.1` and `txt_mod.1`) are skipped, so 60 blocks × 8 locations = 480 modules are applied.

## Log Appearance (Environment Variable Enabled: Skip Disabled)

When environment variable `QWENIMAGE_LORA_APPLY_AWQ_MOD=1` is set, skip logs are not output, and instead apply logs are output:

```
[APPLY] LoRA applied to: transformer_blocks.0.img_mod.1
[APPLY] LoRA applied to: transformer_blocks.0.txt_mod.1
[APPLY] LoRA applied to: transformer_blocks.1.img_mod.1
[APPLY] LoRA applied to: transformer_blocks.1.txt_mod.1
...
[APPLY] LoRA applied to: transformer_blocks.59.img_mod.1
[APPLY] LoRA applied to: transformer_blocks.59.txt_mod.1
```

After that, a log of the number of applied modules is output:

```
Applied LoRA compositions to 600 modules.
```

This number 600 is the total number of modules (60 blocks × 10 locations).

However, **enabling the environment variable has a high probability of causing image noise**, so this setting is not recommended.

## Reason for `Applied LoRA compositions to 480 modules.`

Qwen Image (Double Block) typically applies to approximately 10 locations in each `transformer_blocks.X` (when LoRA keys exist):

1. `attn.to_qkv` (fused Q/K/V attention)
2. `attn.to_out.0`
3. `attn.add_qkv_proj` (additional Q/K/V projection)
4. `attn.to_add_out`
5. `img_mlp.net.0.proj`
6. `img_mlp.net.2`
7. `img_mod.1` ← **Skip target**
8. `txt_mlp.net.0.proj`
9. `txt_mlp.net.2`
10. `txt_mod.1` ← **Skip target**

With default settings, 2 locations (`img_mod.1` and `txt_mod.1`) are skipped.

For **60 blocks** from `transformer_blocks.0` to `59`:

- Skip count: 60 × 2 = 120
- Apply count: 600 - 120 = 480

This matches the log.

## If You Still Want to Apply to Modulation Layers (Not Recommended)

In conclusion, **there is no guarantee that application will not cause noise** (varies by LoRA/strength/model instance, low reproducibility).  
If you still want to try, you can force it ON with an environment variable.

### PowerShell (Before Startup)

```powershell
setx QWENIMAGE_LORA_APPLY_AWQ_MOD 1
```

To restore (OFF):

```powershell
setx QWENIMAGE_LORA_APPLY_AWQ_MOD 0
```

After setting, **ComfyUI restart is required** (does not reflect in already running processes).

## Summary

- **Default Behavior**: LoRA application to `img_mod.1` and `txt_mod.1` is skipped (to prevent noise).
- **Skip Logs**: With default settings, a WARNING log is output for each modulation layer (up to 120 entries).
- **Apply Logs**: When enabled via environment variable, an INFO log is output for each modulation layer (up to 120 entries).
- **Apply Count**: 480 modules with defaults, 600 modules when environment variable is enabled.
- **Environment Variable**: Can be force-enabled with `QWENIMAGE_LORA_APPLY_AWQ_MOD=1`, but this is not recommended due to noise risk.

