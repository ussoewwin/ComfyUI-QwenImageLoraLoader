# Release Notes v2.3.7: V3 Nodes Always Enable AWQ Modulation Layer LoRA

## Overview

This release updates V3 nodes (`NunchakuQwenImageLoraStackV3`) to **always enable** AWQ modulation layer LoRA application without requiring the `apply_awq_mod` toggle that was needed in V2 nodes. After v2.3.6 implementation, no noise issues have been reported, so we determined it is safe to always enable AWQ modulation layer LoRA application in V3 nodes.

---

## Reason for Change

### Background

In v2.3.6, we implemented the **Runtime Monkey Patch with Manual Planar Injection** fix for AWQ quantized modulation layers (`img_mod.1` / `txt_mod.1`). This fix resolved the noise issues that occurred when applying LoRA to these sensitive layers.

### Decision to Always Enable in V3

After v2.3.6 was released and used in production:
- **No noise issues have been reported** by users
- **Extensive testing by the developer** confirmed the fix works correctly without noise issues
- The Manual Planar Injection approach has proven stable
- The fix works correctly for all tested LoRA formats

Based on this validation (both user reports and developer testing), we determined it is safe to **always enable** AWQ modulation layer LoRA application in V3 nodes, eliminating the need for users to manually toggle the `apply_awq_mod` switch.

### V2 Switch Retention

The `apply_awq_mod` switch in V2 nodes is **currently retained** for:
- **Backward compatibility**: Existing V2 workflows continue to work as before
- **User preference**: V2 users can still manually control AWQ modulation layer LoRA application if desired

---

## Modified Files

| File | Changes |
|------|---------|
| `nodes/lora/qwenimage_v3.py` | Updated to always set `apply_awq_mod=True` when creating or updating wrapper instances |

---

## Code Changes and Explanation

### Modified File: `nodes/lora/qwenimage_v3.py`

#### Change 1: New Wrapper Creation (Lines 218-227)

**Before:**
```python
wrapped_model = ComfyQwenImageWrapper(
    model_wrapper,
    getattr(model_wrapper, "config", {}),
    None,  # customized_forward
    {},  # forward_kwargs
    cpu_offload,  # cpu_offload_setting
    4.0,  # vram_margin_gb
)
```

**After:**
```python
logger.info(f"📦 Creating ComfyQwenImageWrapper with cpu_offload='{cpu_offload}', apply_awq_mod=True (V3: always enabled)")
wrapped_model = ComfyQwenImageWrapper(
    model_wrapper,
    getattr(model_wrapper, "config", {}),
    None,  # customized_forward
    {},  # forward_kwargs
    cpu_offload,  # cpu_offload_setting
    4.0,  # vram_margin_gb
    apply_awq_mod=True,  # V3: Always enable AWQ modulation layer LoRA (no switch needed)
)
```

**Explanation:**
- When creating a new `ComfyQwenImageWrapper` for V3 nodes, we now **explicitly pass `apply_awq_mod=True`** as the 7th parameter
- This ensures that AWQ modulation layer LoRA application is **always enabled** for V3 nodes
- The log message clarifies that this is a V3-specific behavior (always enabled)

#### Change 2: Existing Wrapper Update (Lines 210-213)

**Before:**
```python
if model_wrapper.cpu_offload_setting != cpu_offload:
    logger.info(f"🔄 Updating CPU offload setting from '{model_wrapper.cpu_offload_setting}' to '{cpu_offload}'")
    model_wrapper.cpu_offload_setting = cpu_offload
transformer = model_wrapper.model
```

**After:**
```python
if model_wrapper.cpu_offload_setting != cpu_offload:
    logger.info(f"🔄 Updating CPU offload setting from '{model_wrapper.cpu_offload_setting}' to '{cpu_offload}'")
    model_wrapper.cpu_offload_setting = cpu_offload
# V3: Always ensure apply_awq_mod is True (no switch, always enabled)
if hasattr(model_wrapper, "apply_awq_mod") and model_wrapper.apply_awq_mod != True:
    logger.info(f"🔄 Updating AWQ mod setting from '{model_wrapper.apply_awq_mod}' to 'True' (V3: always enabled)")
    model_wrapper.apply_awq_mod = True
transformer = model_wrapper.model
```

**Explanation:**
- When a model is already wrapped (e.g., from a previous V2 node or earlier V3 node), we check if `apply_awq_mod` exists and is not `True`
- If it's not `True`, we **force it to `True`** to ensure V3 behavior (always enabled)
- This handles cases where a model was previously wrapped with V2 (which may have `apply_awq_mod=False`) and is now being used with V3
- The log message indicates this is a V3-specific update

---

## Impact

### For V3 Users

- **No manual toggle needed**: V3 users no longer need to manually enable the AWQ modulation layer toggle
- **Automatic application**: The Manual Planar Injection fix is automatically applied for all LoRAs in V3 nodes
- **Simplified workflow**: One less setting to configure when using V3 nodes

### For V2 Users

- **No changes**: V2 nodes continue to work exactly as before
- **Switch retained**: The `apply_awq_mod` toggle remains available in V2 nodes for manual control
- **Backward compatible**: Existing V2 workflows are unaffected

---

## Technical Details

### How It Works

1. **V3 Node Initialization**: When `NunchakuQwenImageLoraStackV3` creates or updates a `ComfyQwenImageWrapper`, it sets `apply_awq_mod=True`
2. **Wrapper Behavior**: The wrapper passes `apply_awq_mod=True` to `compose_loras_v2()` in the backend
3. **Backend Processing**: In `nunchaku_code/lora_qwen.py`, when `apply_awq_mod=True`, AWQ modulation layers are processed using Manual Planar Injection
4. **Runtime Injection**: The `NunchakuQwenImageTransformerBlock.forward` method (patched at runtime) injects LoRA weights directly into the modulation layer output

### Relationship to v2.3.6

This release builds upon the v2.3.6 Manual Planar Injection implementation:
- **v2.3.6**: Implemented the fix with a toggle (`apply_awq_mod`) in V2 nodes
- **v2.3.7**: Validates the fix is stable and enables it by default in V3 nodes

---

## Migration Notes

### From V2 to V3

If you're migrating from V2 to V3 nodes:
- **No action required**: V3 nodes automatically enable AWQ modulation layer LoRA
- **If you had `apply_awq_mod=True` in V2**: Behavior is identical in V3
- **If you had `apply_awq_mod=False` in V2**: V3 will now apply AWQ modulation layer LoRA (this is the intended behavior)

### Staying with V2

If you prefer to continue using V2 nodes:
- **No changes**: V2 nodes work exactly as before
- **Toggle available**: You can still control AWQ modulation layer LoRA application via the `apply_awq_mod` toggle

---

## Related Issues

- AWQ modulation layer noise issues (resolved in v2.3.6 with Manual Planar Injection)

---

## Summary

| Aspect | Details |
|--------|---------|
| **Release Type** | Feature Update |
| **Target Nodes** | `NunchakuQwenImageLoraStackV3` |
| **Change** | AWQ modulation layer LoRA always enabled (no switch needed) |
| **Reason** | No noise issues reported after v2.3.6 implementation |
| **V2 Compatibility** | Switch retained for backward compatibility |
| **Breaking Changes** | None (V2 nodes unchanged) |

This update simplifies V3 node usage while maintaining full backward compatibility with V2 nodes.
