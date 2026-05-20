<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.2"><font color="#ffffff"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.4.2.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## Overview

This release fixes two issues in **ComfyUI-QwenImageLoraLoader** when using **Nunchaku Qwen Image** models with **Qwen Image Union ControlNet (Fun ControlNet)** and when other code accesses **`model_wrapper.model`** (e.g. `NunchakuQwenImageLoraStackV3`):

1. **`AttributeError`**: `'ComfyQwenImageWrapper' object has no attribute 'process_img'` when ControlNet calls `base_model.process_img(x)`.
2. **`RecursionError`**: `maximum recursion depth exceeded` when `__getattr__` follows `model` → `model_wrapper` → `model` in a loop.

---

## Modified files

- **`ComfyUI-QwenImageLoraLoader/wrappers/qwenimage.py`**
  - Class: `ComfyQwenImageWrapper` (inherits `nn.Module`)

---

## Issue 1: Missing `process_img` for ControlNet

### Symptom

- **Error**: `AttributeError: 'ComfyQwenImageWrapper' object has no attribute 'process_img'`
- **Where**: ComfyUI `comfy/ldm/qwen_image/controlnet.py` → `QwenImageFunControlNetModel.forward` → `base_model.process_img(x)`
- **When**: Nunchaku Qwen Image loaded via this extension, used with **Qwen Image Union ControlNet (Fun ControlNet)**

### Cause

`ComfyQwenImageWrapper` did not implement the **`process_img`** hook that ComfyUI’s Qwen Image ControlNet stack expects on the base diffusion model.

### Fix

Register **`process_img`** on `ComfyQwenImageWrapper` and delegate to the inner Nunchaku model’s `process_img` when present; otherwise return the input unchanged.

---

## Issue 2: `__getattr__` recursion on `model`

### Symptom

- **Error**: `RecursionError: maximum recursion depth exceeded`
- **Where**: e.g. `NunchakuQwenImageLoraStackV3` accessing `model_wrapper.model`
- **When**: `__getattr__` on `ComfyQwenImageWrapper` forwarded unknown attributes to `self.model`, and `self.model` pointed back at the wrapper → infinite loop.

### Fix

- If `name == "model"`, return the inner Nunchaku module directly (no re-entry through the wrapper).
- Other attributes still use `getattr(self.model, name)` with a safe fallback.

---

## Code changes (summary)

| Area | Change |
|------|--------|
| `process_img` | Added; delegates to inner model when available |
| `__getattr__` | `"model"` returns inner module; breaks wrapper recursion |
| `__init__` | `self.model = self.model_wrapper` (unchanged pattern; fix is in `__getattr__`) |

---

## Summary

| Item | Before (v2.4.1) | After (v2.4.2) |
|------|----------------|----------------|
| Qwen Image + Fun/Union ControlNet | `process_img` missing | `process_img` available |
| `model_wrapper.model` access | Recursion possible | Recursion guarded |
| Chinese release notes | — | [`zhmd/v2.4.2.md`](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.4.2.md) |

**Upgrade:** ComfyUI-QwenImageLoraLoader **v2.4.1 → v2.4.2** (Manager update recommended).
