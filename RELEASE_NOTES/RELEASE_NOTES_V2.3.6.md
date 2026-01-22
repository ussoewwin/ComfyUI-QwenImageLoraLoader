# Release Notes v2.3.6: AWQ Modulation Layer LoRA Fix

## Overview

This release implements a critical fix for LoRA application to AWQ quantized modulation layers (`img_mod.1` / `txt_mod.1`) in Qwen Image models. The fix uses a **Runtime Monkey Patch** approach with **Manual Planar Injection** to resolve noise issues that occurred when applying LoRA to these sensitive layers.

---

## Background and Problem

### Issue: LoRA Application Failure on AWQ Modulation Layers

The Nunchaku backend's AWQ kernel stores modulation layer weights (`img_mod`, `txt_mod`) in an optimized special layout (Interleaved). When using standard LoRA application methods (addition via hooks), the computed result's shape and memory layout would become inconsistent, causing **`RuntimeError` or severe noise**. Initially, LoRA application to modulation layers was forcibly skipped to avoid this issue.

### Solution: Manual Planar Injection

To solve this problem, we built a custom path that bypasses hooks:
1. **Remove Skip**: Load LoRA weights on layers that were previously skipped.
2. **Manual Injection**: Instead of using hooks, manually add pre-computed LoRA weights (`A @ B`) inside the model.
3. **SiLU & Dtype Adjustment**: Match the model's internal behavior by applying `SiLU` to the input before LoRA application and casting to `Float16` for computation.

---

## Architecture: Runtime Monkey Patch

We avoided directly modifying the Nunchaku library (`ComfyUI-nunchaku`) to prevent conflicts during updates and complexity in environment management. Instead, we adopted a method that **rewrites Nunchaku's methods at runtime from the custom node side**.

### Processing Flow

1. **On Startup**: `ComfyUI-QwenImageLoraLoader` is loaded.
2. **Patch Application**: `__init__.py` calls `patches/nunchaku_patch.py`.
3. **Method Replacement**: The `NunchakuQwenImageTransformerBlock.forward` method in memory is replaced with our modified method (with Manual Injection implemented).
4. **During Inference**: The model uses the modified method for computation, and LoRA is correctly applied.

---

## Complete Implementation Details (Full Code Explanation)

The following explains the implementation details for all 5 component files.

### ① Backend Logic: `nunchaku_code/lora_qwen.py`

**Role**: Handles LoRA weight loading, scaling, and **data injection for "Manual Planar Injection"**.

**Key Code Block**:
When AWQ modulation layers are detected, normal hook application is avoided, and weights are attached to the module as a tuple.

```python
# nunchaku_code/lora_qwen.py

# For AWQ quantized model AND modulation layer (img_mod/txt_mod)
if is_awq_w4a16 and is_modulation_layer:
    logger.info(f"[AWQ_MOD] {resolved_name}: Storing LoRA weights for manual Planar injection")
    
    mod = module
    
    # 【Core】Store pre-computed weights (A, B) as a tuple on the module
    # Here we do not perform computation, but simply hold the data as an attribute.
    mod._nunchaku_lora_bundle = (final_A, final_B)
    
    # Remove and disable existing patches (hooks) if they exist
    # This prevents double application and shape mismatch errors.
    if hasattr(mod, "_lora_original_forward"):
        mod.forward = mod._lora_original_forward
        del mod._lora_original_forward
    
    # Tag and update counter
    mod._is_modulation_layer = True
    mod_layer_applied_count += 1
else:
    # Other layers proceed normally
    _apply_lora_to_module(module, final_A, final_B, resolved_name, model)
```

---

### ② The Switch: `wrappers/qwenimage.py`

**Role**: Holds configuration values received from ComfyUI nodes and relays them to backend loader functions.

```python
# wrappers/qwenimage.py

class ComfyQwenImageWrapper(nn.Module):
    def __init__(self, model, config, loras=None, auto_cpu_offload=True, apply_awq_mod="auto"):
        # ...
        self.apply_awq_mod = apply_awq_mod  # Hold configuration value

    def forward(self, x, timestep, context, ...):
        # ...
        # When LoRA recomposition is needed, pass the held flag
        is_supported_format = compose_loras_v2(self.model, self.loras, apply_awq_mod=self.apply_awq_mod)
        # ...
```

---

### ③ Node Definition: `nodes/lora/qwenimage_v2.py`

**Role**: Defines user interface (node) inputs. Included in hash calculation to detect changes.

```python
# nodes/lora/qwenimage_v2.py

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # ...
                "apply_awq_mod": (
                    "BOOLEAN",
                    {
                        "default": False,  # Default OFF for safety
                        "tooltip": "Force enable LoRA application to AWQ modulation layers...",
                    },
                ),
            },
        }

    # Include in hash value via IS_CHANGED to trigger re-execution on change
    @classmethod
    def IS_CHANGED(cls, model, lora_count, cpu_offload="disable", **kwargs):
        m = hashlib.sha256()
        # ...
        m.update(str(kwargs.get("apply_awq_mod", False)).encode())
        return m.digest().hex()
```

---

### ④ Frontend UI: `js/z_qwen_lora_dynamic.js`

**Role**: Dynamically renders and controls widgets (switches) in the browser.

```javascript
// js/z_qwen_lora_dynamic.js

        node.updateLoraSlots = function () {
            // ...
            // Add apply_awq_mod widget from cache
            if (node.cachedApplyAwqMod) {
                this.widgets.push(node.cachedApplyAwqMod);
            }
            // ...
            
            // Include in height calculation (add height only when displayed)
            const APPLY_AWQ_MOD_H = node.cachedApplyAwqMod ? 30 : 0;
            const targetH = HEADER_H + CPU_OFFLOAD_H + APPLY_AWQ_MOD_H + (count * SLOT_H) + PADDING;

            this.setSize([this.size[0], targetH]);
        };
```

---

### ⑤ The Patcher: `patches/nunchaku_patch.py` & `__init__.py`

**Role**: This is the core file of the fix. Defines a `forward` method (originally intended to be written in the model itself) implementing Manual Planar Injection and injects it into the Nunchaku class.

```python
# patches/nunchaku_patch.py

def forward_with_manual_planar_injection(self, ...):
    # ... (normal processing) ...

    # --- Manual Planar Injection (img_mod) ---
    img_mod_params = self.img_mod(temb)
    
    # If bundle is injected, compute and add
    if hasattr(self.img_mod[1], "_nunchaku_lora_bundle"):
            A, B = self.img_mod[1]._nunchaku_lora_bundle
            
            # 【Important】Pre-processing to match model structure
            # 1. Modulation layer is after SiLU, so apply SiLU to LoRA input too
            input_tensor = self.img_mod[0](temb) 
            
            # 2. Cast to avoid dtype mismatch (BF16 vs FP16)
            lora = (input_tensor.to(dtype=A.dtype) @ A.t()) @ B.t()
            
            img_mod_params = img_mod_params + lora.to(dtype=img_mod_params.dtype, device=img_mod_params.device)
    # -----------------------------------------

    # ... (txt_mod similar) ...
    
    return encoder_hidden_states, hidden_states

# Patch application function
def apply_nunchaku_patch():
    try:
        from nunchaku.models.qwenimage import NunchakuQwenImageTransformerBlock
        # Replace method (Monkey Patch)
        NunchakuQwenImageTransformerBlock.forward = forward_with_manual_planar_injection
        return True
    except Exception as e:
        logger.error(...)
```

```python
# __init__.py

# Automatically apply on node load
try:
    from .patches.nunchaku_patch import apply_nunchaku_patch
    if apply_nunchaku_patch():
        logger.info("Successfully applied Nunchaku Manual Planar Injection monkey patch.")
    # ...
```

---

## How It Works

### Complete Pipeline

This implementation establishes a consistent pipeline from user interface to backend to patch application:

*   **UI/Node**: Receives user intent (toggle ON).
*   **Wrapper**: Relays configuration.
*   **BackEnd**: Computes weights and holds them as "bundle" for Manual Injection.
*   **Monkey Patch**: Changes model behavior at runtime, receives the bundle and computes correctly.

This achieves advanced Manual Planar Injection functionality without breaking existing libraries.

---

## Results and Benefits

This architecture achieves the following:

*   **Full Support for Style LoRAs on AWQ Models**: Powerful LoRAs like Flat Color now work correctly without noise
*   **Zero Impact on Nunchaku Core**: No changes to `ComfyUI-nunchaku` core files, ensuring compatibility with updates
*   **Portability**: Simply installing this custom node automatically applies the fix
*   **User Control**: Users can enable/disable AWQ modulation layer LoRA application via toggle
*   **Safety First**: Default is disabled to prevent noise, users opt-in when needed

**Impact**: This fix enables all standard format LoRA keys that were previously excluded from AWQ modulation layers to be fully applied. Previously, these layers were skipped to prevent noise, but now they work correctly with the Manual Planar Injection approach.

⚠️ **Warning**: **This is an experimental feature currently implemented only in V2 nodes.**

---

## Breaking Changes

None.

---

## Migration Guide

No migration required. The fix is automatically applied when the custom node is loaded. Users can enable AWQ modulation layer LoRA application via the "Apply AWQ Mod" toggle in V2 nodes.

**Note**: If you were previously using environment variable `QWENIMAGE_LORA_APPLY_AWQ_MOD=1`, you can now use the UI toggle instead for more granular control per workflow.

---

## Additional Fix: Z-Image-Turbo Module Resolution

### Problem Discovered

During the implementation of the AWQ modulation layer fix, a critical issue was discovered affecting Z-Image-Turbo (NextDiT) models. The stricter `module is None` checks introduced in the AWQ fix exposed an existing problem where **120 out of 150 LoRA keys were being silently skipped** due to module resolution failures.

### Root Cause

Z-Image-Turbo models use NextDiT architecture, which has different module naming conventions compared to Qwen Image models:

- **Qwen Image**: `layers.N.attention.to_qkv`
- **NextDiT**: `layers.N.attention.qkv` (no `to_` prefix)

- **Qwen Image**: `layers.N.attention.to_out.0`
- **NextDiT**: `layers.N.attention.out` (no `to_` prefix and no `.0` suffix)

- **Qwen Image**: `layers.N.feed_forward.net.0.proj`
- **NextDiT**: `layers.N.feed_forward.w13` (nunchaku-patched) or `layers.N.feed_forward.w1` (unpatched)

- **Qwen Image**: `layers.N.feed_forward.net.2`
- **NextDiT**: `layers.N.feed_forward.w2`

The `_resolve_module_name` function lacked fallback paths for these NextDiT naming conventions, causing module resolution to fail and LoRA keys to be skipped.

### Solution: NextDiT Fallback Mappings

**Modified File**: `nunchaku_code/lora_qwen.py`

**Function**: `_resolve_module_name` (lines 412-465)

**Changes**:

1. **Added NextDiT Attention Module Fallbacks** (lines 427-437):
   ```python
   # Z-Image-Turbo (NextDiT) fallback mappings
   # NextDiT uses: layers.N.attention.qkv (not .to_qkv)
   if ".attention.to_qkv" in name:
       alt = name.replace(".attention.to_qkv", ".attention.qkv")
       m = _get_module_by_name(model, alt)
       if m is not None: return alt, m
   # NextDiT uses: layers.N.attention.out (not .to_out.0)
   if ".attention.to_out.0" in name:
       alt = name.replace(".attention.to_out.0", ".attention.out")
       m = _get_module_by_name(model, alt)
       if m is not None: return alt, m
   ```

2. **Added NextDiT Feed Forward Module Fallbacks** (lines 438-452):
   ```python
   # NextDiT feed_forward: .net.0.proj -> .w13 (for GLU) or .w1/.w3 (unpatched)
   if ".feed_forward.net.0.proj" in name:
       # Try w13 first (nunchaku-patched)
       alt = name.replace(".feed_forward.net.0.proj", ".feed_forward.w13")
       m = _get_module_by_name(model, alt)
       if m is not None: return alt, m
       # Try w1 (unpatched, for gate)
       alt = name.replace(".feed_forward.net.0.proj", ".feed_forward.w1")
       m = _get_module_by_name(model, alt)
       if m is not None: return alt, m
   # NextDiT feed_forward: .net.2 -> .w2
   if ".feed_forward.net.2" in name:
       alt = name.replace(".feed_forward.net.2", ".feed_forward.w2")
       m = _get_module_by_name(model, alt)
       if m is not None: return alt, m
   ```

3. **Improved Error Logging** (lines 1360-1363):
   ```python
   # Skip if module not found
   if module is None:
       logger.warning(f"[MISS] Module not found: {module_name_key} (resolved: {resolved_name})")
       skipped_modules_count += 1
       continue
   ```
   
   Changed from `logger.debug` to `logger.warning` to make module resolution failures visible, and added `skipped_modules_count += 1` to properly track skipped modules.

### Impact

- **Before Fix**: 150 LoRA keys → only 30 applied (120 skipped silently)
- **After Fix**: 150 LoRA keys → all 150 applied successfully

This fix ensures that Z-Image-Turbo models receive full LoRA support, matching the functionality available for Qwen Image models.

### Technical Details

**Module Resolution Process**:

1. **Primary Resolution**: Attempts to resolve using the original key name (e.g., `layers.0.attention.to_qkv`)
2. **Fallback Resolution**: If primary fails, attempts NextDiT naming conventions (e.g., `layers.0.attention.qkv`)
3. **Result**: Returns the resolved module object or `None` if all attempts fail

**Why This Wasn't Caught Earlier**:

The issue existed before but was masked by less strict error handling. The AWQ fix introduced stricter `module is None` checks, which exposed the underlying module resolution failures. This is a positive side effect of the improved error handling, as it revealed a critical bug affecting Z-Image-Turbo users.

---

## Related Issues

This fix resolves noise issues when applying LoRA to AWQ quantized modulation layers, which was previously causing complete image corruption in some cases. The Runtime Monkey Patch approach ensures the fix is applied automatically without requiring any manual modifications to Nunchaku or ComfyUI core files.

**Additional Issue Resolved**: [Issue #44](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/44) - Module resolution failures for Z-Image-Turbo models causing 120 out of 150 LoRA keys to be skipped.
