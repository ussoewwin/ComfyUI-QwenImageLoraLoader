# Qwen Image ControlNet Support and __getattr__ Recursion Error Fix — Detailed Explanation

This document fully explains the cause and solution (with code) for the following two fixes applied to **ComfyUI-QwenImageLoraLoader**’s `ComfyQwenImageWrapper`:

1. **AttributeError**: `'ComfyQwenImageWrapper' object has no attribute 'process_img'`  
   → Error when using Qwen Image Union ControlNet (e.g. Fun ControlNet) together with Nunchaku Qwen Image model
2. **RecursionError**: `maximum recursion depth exceeded`  
   → Infinite recursion when accessing `model_wrapper.model` in `NunchakuQwenImageLoraStackV3` and similar code

---

## Files Modified

- **ComfyUI-QwenImageLoraLoader/wrappers/qwenimage.py**
  - Class: `ComfyQwenImageWrapper` (inherits from `nn.Module`)

---

## Issue 1: Missing base_model Interface Required by ControlNet

### Symptom

- **Error**: `AttributeError: 'ComfyQwenImageWrapper' object has no attribute 'process_img'`
- **Location**: Inside `QwenImageFunControlNetModel.forward` in ComfyUI’s `comfy/ldm/qwen_image/controlnet.py`
- **Condition**: When running KSampler in a workflow that loads a Nunchaku Qwen Image model and applies **Qwen Image Union ControlNet (Fun ControlNet)**

### Call Path (Summary)

```
KSampler
  → common_ksampler
    → comfy.sample.sample
      → … (TiledDiffusion / EasyCache etc. wrappers)
        → samplers: calc_cond_batch
          → controlnet.get_control
            → control_model(x=..., base_model=model, ...)
              → QwenImageFunControlNetModel.forward(..., base_model=...)
                → base_model.process_img(x)   ← AttributeError when model is ComfyQwenImageWrapper
```

The `model` passed to the sampler is already wrapped as **`ComfyQwenImageWrapper`** by ComfyQwenImageLoader etc. when using Nunchaku. ControlNet receives this object as `base_model` and calls e.g. `base_model.process_img(x)`.

### What ComfyUI’s ControlNet Expects from base_model

Inside `forward` in `comfy/ldm/qwen_image/controlnet.py`, the following are required on `base_model`:

| Requirement | Usage (example) |
|-------------|------------------|
| `base_model.process_img(x)` | Obtain hidden_states / img_ids from latent x |
| `base_model.patch_size` | Text position computation |
| `base_model.pe_embedder(ids)` | Position encoding |
| `base_model.img_in(hidden_states)` | Linear projection of image tokens |
| `base_model.txt_norm(context)` / `base_model.txt_in(...)` | Text-side normalization and projection |
| `base_model.time_text_embed(timesteps, ...)` | Timestep embedding |

ComfyUI’s `QwenImageTransformer2DModel` (and Nunchaku’s `NunchakuQwenImageTransformer2DModel`) have all of these. **The wrapper `ComfyQwenImageWrapper` did not expose these attributes other than `forward`**, so when `base_model` was the wrapper, an AttributeError occurred.

### Root Cause Summary

- **Cause**: `ComfyQwenImageWrapper` was focused only on delegating the inference `forward`, and did **not** provide the **base_model interface** (`process_img` and the attributes above) that ControlNet assumes.
- **Responsibility**: ComfyUI’s ControlNet assumes that the passed `base_model` is Qwen Image–like; the wrapper must satisfy the same interface (this fix is on the wrapper side).

---

## Fix for Issue 1

### Fix 1-A: Add `process_img` Method

**Goal**: Allow ControlNet to call `base_model.process_img(x)`.

**Added code** (`wrappers/qwenimage.py`):

```python
def process_img(self, x, index=0, h_offset=0, w_offset=0):
    """
    Delegate to the inner model so Qwen Image ControlNet (e.g. Fun ControlNet)
    can call base_model.process_img(x). Required when using Union CN with
    Nunchaku Qwen Image model wrapped by this wrapper.
    """
    if self.model is None:
        raise RuntimeError("Model has been unloaded. Cannot call process_img.")
    return self.model.process_img(x, index=index, h_offset=h_offset, w_offset=w_offset)
```

- Signature and return value match ComfyUI’s `QwenImageTransformer2DModel.process_img` (e.g. 3-tuple `(hidden_states, img_ids, orig_shape)`); the call is delegated to the inner `self.model` (Nunchaku model).
- Here `self.model` is accessed via **normal attribute access** (not through `__getattr__`) to avoid the recursion described in Issue 2. Because `process_img` is defined as a method on `ComfyQwenImageWrapper`, `__getattr__` is not used for this call, and `self.model` is resolved via `nn.Module`’s `_modules["model"]`.

### Fix 1-B: Forward Other base_model Attributes via `__getattr__`

**Goal**: When the wrapper does not have attributes such as `patch_size`, `pe_embedder`, `img_in`, `txt_norm`, `txt_in`, `time_text_embed` that ControlNet uses, forward them to the inner model.

**Added code** (final form including the recursion fix for Issue 2; see below):

- A `__getattr__` was added that, when an attribute name not present on the wrapper is accessed, returns the attribute of the same name from the inner `self.model`.
- This allows ControlNet to use the above attributes transparently even when `base_model` is `ComfyQwenImageWrapper`.

---

## Issue 2: Infinite Recursion from self.model Access Inside __getattr__

### Symptom

- **Error**: `RecursionError: maximum recursion depth exceeded`
- **Location**: In `load_lora_stack` in `ComfyUI-QwenImageLoraLoader/nodes/lora/qwenimage_v3.py`, at `transformer = model_wrapper.model`
- **Condition**: When `NunchakuQwenImageLoraStackV3` (or similar) receives a `model_wrapper` already wrapped with `ComfyQwenImageWrapper` and accesses **`model_wrapper.model`** to obtain the inner transformer

### Stack Trace (Excerpt)

```
File "...\qwenimage_v3.py", line 255, in load_lora_stack
    transformer = model_wrapper.model
File "...\wrappers\qwenimage.py", line 85, in __getattr__
    if self.model is None:
[Previous line repeated 1493 more times]
RecursionError: maximum recursion depth exceeded
```

So: “get `model` → `__getattr__` is called → it references `self.model` → `__getattr__` is called again” — a loop.

### Root Cause: How nn.Module Stores Attributes and __getattr__

- `ComfyQwenImageWrapper` subclasses `torch.nn.Module`.
- In `nn.Module`, **child modules** assigned with e.g. `self.model = model` are stored in **`_modules["model"]`**, not in `__dict__["model"]` (per `Module.__setattr__`).
- With normal attribute access, `Module.__getattribute__` consults `_modules`, so `obj.model` is resolved without recursion.
- **`__getattr__(self, name)` is only invoked when the normal `__getattribute__` did not find the attribute.** Depending on the environment and Python’s resolution order, access to `model` can be routed to `__getattr__`.
- If **inside `__getattr__` we then reference `self.model`**, attribute resolution runs again for “`model` on `self`”, and the same `__getattr__` is invoked again. When the name is `"model"`, touching `self.model` to return the inner model becomes the **source of recursion**.

Summary:

- **Cause**: Referencing `self.model` inside `__getattr__` causes infinite recursion when `__getattr__` is called for the name `"model"`, because that reference triggers `__getattr__` again.
- **Responsibility**: Only this wrapper implementation. Not a bug in ComfyUI or Nunchaku.

---

## Fix for Issue 2: Do Not Reference self.model Inside __getattr__

**Approach**: **Never use the attribute access `self.model` inside `__getattr__`.** Obtain the inner model from `object.__getattribute__(self, "_modules")`.

- Because `nn.Module` keeps child modules in `self._modules`, we can get `_modules` without going through `__getattr__` via `object.__getattribute__(self, "_modules")`.
- The `"model"` entry in that `_modules` is the inner Nunchaku model held by the wrapper.
- Callers using `model_wrapper.model` expect the “inner transformer”; when `name == "model"`, we return that inner model.

**Full `__getattr__` after the fix** (`wrappers/qwenimage.py`):

```python
def __getattr__(self, name):
    """
    Forward attribute access to the inner model so Qwen Image ControlNet
    (e.g. Fun ControlNet) can access patch_size, pe_embedder, img_in,
    txt_norm, txt_in, time_text_embed, etc. when base_model is this wrapper.
    Use _modules to get the inner model to avoid recursion: inside this
    __getattr__, accessing self.model would trigger __getattr__ again.
    """
    try:
        inner = object.__getattribute__(self, "_modules").get("model")
    except (AttributeError, KeyError):
        inner = None
    if inner is None:
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    if name == "model":
        return inner
    return getattr(inner, name)
```

### Points

1. **`object.__getattribute__(self, "_modules")`**  
   - Retrieves `_modules` without going through this class’s or `nn.Module`’s `__getattr__`.
2. **`.get("model")`**  
   - Gets the inner model registered as a child module; `None` if missing.
3. **When `name == "model"`, return `inner`**  
   - Matches the node’s expectation that `model_wrapper.model` returns the inner transformer.
4. **For any other name**  
   - ControlNet attributes like `patch_size`, `pe_embedder`, `img_in` are all forwarded to the inner model via `getattr(inner, name)`.

This satisfies both requirements:

- `model_wrapper.model` → returns the inner model without recursion  
- `base_model.patch_size` etc. → forwarded to the inner model so ControlNet’s expected interface is provided  

---

## Order of Fixes and Dependencies

1. **For Issue 1**:  
   - Add `process_img`  
   - Add `__getattr__` for ControlNet attribute forwarding (initially implemented with a `self.model` reference)
2. **Issue 2 appears**:  
   - That `self.model` reference inside `__getattr__` caused recursion when accessing `model_wrapper.model`.
3. **For Issue 2**:  
   - Stop using `self.model` inside `__getattr__` and obtain the inner model via `object.__getattribute__(self, "_modules").get("model")`.

Result: **Both Issue 1 (ControlNet) and Issue 2 (LoRA stack model access) are resolved.**

---

## Summary of Modified Code (Relevant Parts)

Excerpt of what was added/changed on `ComfyQwenImageWrapper`:

```python
# --- Fix 1-A: Add process_img ---
def process_img(self, x, index=0, h_offset=0, w_offset=0):
    """
    Delegate to the inner model so Qwen Image ControlNet (e.g. Fun ControlNet)
    can call base_model.process_img(x). Required when using Union CN with
    Nunchaku Qwen Image model wrapped by this wrapper.
    """
    if self.model is None:
        raise RuntimeError("Model has been unloaded. Cannot call process_img.")
    return self.model.process_img(x, index=index, h_offset=h_offset, w_offset=w_offset)

# --- Fix 1-B & Issue 2: __getattr__ (recursion-safe) ---
def __getattr__(self, name):
    """
    Forward attribute access to the inner model so Qwen Image ControlNet
    (e.g. Fun ControlNet) can access patch_size, pe_embedder, img_in,
    txt_norm, txt_in, time_text_embed, etc. when base_model is this wrapper.
    Use _modules to get the inner model to avoid recursion: inside this
    __getattr__, accessing self.model would trigger __getattr__ again.
    """
    try:
        inner = object.__getattribute__(self, "_modules").get("model")
    except (AttributeError, KeyError):
        inner = None
    if inner is None:
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    if name == "model":
        return inner
    return getattr(inner, name)
```

- The `self.model` in `process_img` does not go through `__getattr__` for that call (because `process_img` is called as a method on the wrapper), so it does not recurse.
- All attribute access (`model_wrapper.model`, `base_model.patch_size`, etc.) goes through the above `__getattr__`, which only uses the `inner` taken from `_modules["model"]`, so recursion does not occur.

---

## Technical Notes

### ControlNet Usage in ComfyUI (Reference)

The aim of this change is to make `QwenImageFunControlNetModel.forward` in `comfy/ldm/qwen_image/controlnet.py` work transparently when `base_model` is the wrapper.

```python
# Inside forward in controlnet.py (excerpt)
hidden_states, img_ids, _ = base_model.process_img(x)
# ...
txt_start = round(
    max(
        ((x.shape[-1] + (base_model.patch_size // 2)) // base_model.patch_size) // 2,
        ((x.shape[-2] + (base_model.patch_size // 2)) // base_model.patch_size) // 2,
    )
)
# ...
image_rotary_emb = base_model.pe_embedder(ids).to(x.dtype).contiguous()
hidden_states = base_model.img_in(hidden_states)
encoder_hidden_states = base_model.txt_norm(context)
encoder_hidden_states = base_model.txt_in(encoder_hidden_states)
temb = (
    base_model.time_text_embed(timesteps, hidden_states)
    if guidance is None
    else base_model.time_text_embed(timesteps, guidance, hidden_states)
)
```

Because the wrapper forwards all of the above to the inner model via `process_img` and `__getattr__`, this code works unchanged when `base_model` is `ComfyQwenImageWrapper`.

### Relation to Other Issues

- **Issue #25 (ComfyUI 0.4.0 Model Management Errors)** is a **different NoneType-related problem** involving ComfyUI’s `model_management.py` and GC/weak references. It has a different cause and symptoms from the **AttributeError (process_img)** and **RecursionError (__getattr__)** addressed here.  
  See [COMFYUI_0.4.0_MODEL_MANAGEMENT_ERRORS.md](./COMFYUI_0.4.0_MODEL_MANAGEMENT_ERRORS.md) for details.

---

## Summary

| Item | Issue 1 (ControlNet) | Issue 2 (RecursionError) |
|------|----------------------|---------------------------|
| Symptom | `AttributeError: ... has no attribute 'process_img'` | `RecursionError: maximum recursion depth exceeded` |
| Cause | Wrapper did not provide ControlNet’s base_model interface | Referencing `self.model` inside `__getattr__` caused recursion when getting `model` |
| Fix | Add `process_img` + attribute-forwarding `__getattr__` | Inside `__getattr__`, get inner model from `_modules["model"]` |
| File | `wrappers/qwenimage.py` | Same |

With these two fixes:

- **Qwen Image Union ControlNet (e.g. Fun ControlNet)** can be used together with **Nunchaku Qwen Image model**, and
- Code that obtains the inner transformer via `model_wrapper.model` (e.g. **NunchakuQwenImageLoraStackV3**) runs without recursion.
