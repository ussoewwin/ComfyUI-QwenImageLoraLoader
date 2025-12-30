# Applying LoRA to Qwen Image (AWQ) img_mod.1 / txt_mod.1 (modulation) Layers - Complete Explanation

## Overview

In Qwen Image models, each transformer block has `img_mod.1` and `txt_mod.1` modulation layers. These layers are linear layers that generate shift/scale/gate parameters for AdaNorm.

In Nunchaku's AWQ quantized models, these layers are implemented as quantized linear layers called `AWQW4A16Linear`. Since the standard LoRA application method (directly modifying weights) cannot be applied to quantized layers, we adopt a method of adding LoRA during the forward pass.

However, as a result of implementation and testing, applying LoRA to `img_mod.1` and `txt_mod.1` causes image noise. Therefore, we have implemented a **safety mechanism that skips LoRA application to these layers by default**.

## Detailed Code Explanation

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

1. `os.getenv("QWENIMAGE_LORA_APPLY_AWQ_MOD", "0")` loads the environment variable. The default value is "0" (disabled).
2. `str(...).strip().lower()` normalizes the string (removes leading/trailing whitespace and converts to lowercase).
3. `in ("1", "true", "yes", "y", "on")` checks if it's a valid value (values considered true).
4. As a result, if the environment variable is not set, or has values like "0" / "false" / "no", it becomes `False` and is skipped by default.

**Debug Log Display**:

This variable itself is not displayed in logs, but logs are output in the skip logic described later.

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

Within the `compose_loras_v2` function, this checks whether each module is `AWQW4A16Linear`.

1. `module.__class__.__name__ == "AWQW4A16Linear"` checks the class name. We use class name comparison instead of `isinstance()` to account for possible differences in import paths.
2. `hasattr(module, "qweight")` checks if the quantized weight tensor exists. In AWQ, weights are quantized to INT4 and stored in `qweight`.
3. `hasattr(module, "wscales")` and `hasattr(module, "wzeros")` check if quantization scales and zero points exist.
4. `hasattr(module, "in_features")` and `hasattr(module, "out_features")` check if standard linear layer attributes exist.

If all these conditions are met, `is_awq_w4a16 = True` and it is treated as an AWQ quantized layer.

**Debug Log Display**:

This detection process itself is not displayed in logs, but logs are output in subsequent processing.

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

If:
1. `is_awq_w4a16` is `True`, and
2. `resolved_name` contains `.img_mod.1` or `.txt_mod.1`, and
3. `_APPLY_AWQ_MOD` is `False` (environment variable is disabled)

then LoRA application is skipped and processing continues to the next module (`continue`).

**Debug Log Display**:

When skipped, a WARNING level log like the following is output:

```
[SKIP] transformer_blocks.0.img_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
[SKIP] transformer_blocks.0.txt_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
[SKIP] transformer_blocks.1.img_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
[SKIP] transformer_blocks.1.txt_mod.1: AWQ modulation layer LoRA is disabled by default (prevents noise). Set QWENIMAGE_LORA_APPLY_AWQ_MOD=1 to force-enable.
...
```

This log is output for each `img_mod.1` and `txt_mod.1` of each transformer block (typically 0 to 59, 60 blocks total). In other words, up to 120 skip logs may be output with default settings.

### 4. LoRA Application Processing to AWQW4A16Linear

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

This processing implements a method of adding LoRA terms during the forward pass without modifying quantized weights when applying LoRA to AWQ quantized layers.

1. **LoRA Slot Initialization**: If `model._lora_slots` doesn't exist, it is initialized. This dictionary is needed for resetting later.

2. **Save Original Forward Method**: The original `forward` method is saved to `module._lora_original_forward`. This allows restoring the original behavior during reset. This processing is executed only once (via `hasattr` check).

3. **Store LoRA Weights**: LoRA A and B matrices are stored in `module._lora_A` and `module._lora_B`. These weights have already been scaled and concatenated by the caller (`compose_loras_v2`).

4. **Implement LoRA-enabled Forward**: An internal function `_forward_with_lora` is defined. This function:
   - Executes normal forward with quantized weights via `module._lora_original_forward(x)`.
   - Calculates LoRA terms via `(x @ module._lora_A.T) @ module._lora_B.T`. Here, `A` has shape `[rank, in_features]` and `B` has shape `[out_features, rank]`. `A.T` becomes `[in_features, rank]` and `B.T` becomes `[rank, out_features]`, so `x @ A.T @ B.T` becomes `[batch, in_features] @ [in_features, rank] @ [rank, out_features] = [batch, out_features]`.
   - Adds LoRA terms to the quantized forward result via `y + lora_out.to(dtype=y.dtype, device=y.device)`. `to(dtype=..., device=...)` aligns data types and devices.

5. **Replace Forward Method**: `module.forward = _forward_with_lora` replaces the module's forward method with the LoRA-enabled version.

6. **Record Slot Information**: `{"type": "awq_w4a16"}` is recorded in `model._lora_slots[module_name]`. This allows identifying that this module is an AWQ quantized layer during reset.

**Debug Log Display**:

When this processing is executed, the following log is output in subsequent processing of the `compose_loras_v2` function (lines 1460-1461):

```
[APPLY] LoRA applied to: transformer_blocks.0.img_mod.1
```

However, with default settings (`_APPLY_AWQ_MOD = False`), `img_mod.1` and `txt_mod.1` are skipped, so this log is not output.

### 5. B Matrix Reordering Processing for Modulation Layers

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

In Qwen Image modulation layers, the output is used as shift/scale/gate parameters (6 blocks total). In Nunchaku's implementation, this output is processed in "interleaved" format.

1. **Problem Background**: Most LoRAs are trained against the standard (non-interleaved) layout (6 blocks × dim). However, in the AWQ LoRA path, LoRA output is added **before** the reordering process, so the B matrix must be converted from standard format to interleaved format.

2. **Shape Check**: `B.ndim == 2 and (B.shape[0] % 6 == 0)` checks if B is 2-dimensional and if `out_features` is divisible by 6. This assumes a structure of 6 blocks (shift/scale/gate × 2).

3. **Reordering Processing**: 
   - `dim = B.shape[0] // 6` calculates the dimension of each block.
   - `B.contiguous().view(6, dim, B.shape[1])` reshapes B to shape `[6, dim, rank]`. `contiguous()` is needed to ensure contiguous memory layout.
   - `.transpose(0, 1)` converts it to `[dim, 6, rank]`. This converts from standard format `[6, dim, rank]` to interleaved format `[dim, 6, rank]`.
   - `.reshape(B.shape[0], B.shape[1])` finally reshapes it back to shape `[6*dim, rank]`.

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

This is the reset (restore) processing for LoRA applied to AWQ quantized layers within the `reset_lora_v2` function.

1. **Restore Forward Method**: `hasattr(module, "_lora_original_forward")` checks if the original forward method is saved. If saved, `module.forward = module._lora_original_forward` restores the original forward method.

2. **Delete LoRA Attributes**: Each attribute `_lora_A`, `_lora_B`, and `_lora_original_forward` is deleted. This restores the module to its original state.

3. **Error Handling**: Each operation is wrapped in `try-except` to ensure the reset process doesn't fail even if errors occur (as the comment "Safety: never fail reset" indicates, reset must always succeed).

**Debug Log Display**:

At the end of the `reset_lora_v2` function (line 1585), the following INFO level log is output:

```
All LoRA weights have been reset from the model.
```

This log is output after reset of all LoRA types (Nunchaku LoRA-ready, standard Linear, AWQ quantized layers) is complete.

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

A WARNING level skip log is output for each `img_mod.1` and `txt_mod.1` of each transformer block (0 to 59, 60 blocks total). In other words, up to 120 skip logs are output.

After that, a log of the number of applied modules is output:

```
Applied LoRA compositions to 480 modules.
```

This number 480 is the total number of modules (600) minus the number of skipped modules (120). Each transformer block typically has 10 LoRA application points, but 2 of them (`img_mod.1` and `txt_mod.1`) are skipped, so 60 blocks × 8 points = 480 modules are applied.

## Log Appearance (Environment Variable Enabled: Skip Disabled)

When the environment variable `QWENIMAGE_LORA_APPLY_AWQ_MOD=1` is set, skip logs are not output, and instead application logs are output:

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

This number 600 is the total number of modules (60 blocks × 10 points).

However, **enabling the environment variable may cause image noise**, so this setting is not recommended.

## Why `Applied LoRA compositions to 480 modules.` Appears

Qwen Image (Double Block) typically has approximately 10 application points per `transformer_blocks.X` (when LoRA keys exist):

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

With default settings, 2 points (`img_mod.1` and `txt_mod.1`) are skipped.

For `transformer_blocks.0` to `59` (60 blocks):

- Skip count: 60 × 2 = 120
- Application count: 600 - 120 = 480

This matches the log.

## If You Still Want to Apply to Modulation Layers (Not Recommended)

In conclusion, **there is no guarantee that it can be applied without noise** (varies by LoRA/strength/model instance, reproducibility is low).  
If you still want to try, you can force enable it with an environment variable.

### PowerShell (Before Startup)

```powershell
setx QWENIMAGE_LORA_APPLY_AWQ_MOD 1
```

To revert (OFF):

```powershell
setx QWENIMAGE_LORA_APPLY_AWQ_MOD 0
```

**ComfyUI restart is required** after setting (does not apply to already running processes).

## Summary

- **Default Behavior**: LoRA application to `img_mod.1` and `txt_mod.1` is skipped (to prevent noise).
- **Skip Logs**: With default settings, a WARNING log is output for each modulation layer (up to 120 entries).
- **Application Logs**: When enabled via environment variable, an INFO log is output for each modulation layer (up to 120 entries).
- **Application Count**: 480 modules with default settings, 600 modules when environment variable is enabled.
- **Environment Variable**: `QWENIMAGE_LORA_APPLY_AWQ_MOD=1` can force enable, but not recommended due to noise risk.
