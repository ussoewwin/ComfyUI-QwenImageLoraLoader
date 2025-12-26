# Technical Documentation

This document provides technical details about the implementation of ComfyUI-QwenImageLoraLoader, focusing on architectural decisions and design patterns.

## Guidance Resolution Strategy

### Problem Statement

The `QwenImageTransformer2DModel.forward` method has the following signature:

```python
def forward(
    self,
    x,
    timestep,
    context,
    attention_mask=None,
    guidance=None,
    ref_latents=None,
    transformer_options={},
    **kwargs
)
```

Note that `guidance` is defined as a **positional argument** (not `**kwargs`).

### The Issue

When calling the model from the wrapper layer (`ComfyQwenImageWrapper`), if `guidance` is passed as a keyword argument while ComfyUI's internal mechanisms also pass it positionally, Python raises:

```
TypeError: NunchakuQwenImageTransformer2DModel._forward() got multiple values for argument 'guidance'
```

This occurs because:
1. ComfyUI's `WrapperExecutor` may pass `guidance` positionally through the forward chain
2. The wrapper was passing `guidance` as a keyword argument
3. Python detects the conflict and raises an error

### Solution

We intentionally pass `guidance` as a **positional argument** to match the exact signature of `QwenImageTransformer2DModel.forward`:

```python
return self.model(
    x,
    timestep,
    context,
    None,                  # attention_mask
    guidance_value,        # guidance (positional)
    None,                  # ref_latents
    transformer_options_cleaned,
    **final_kwargs,
)
```

This ensures:
- ✅ `guidance` is always passed in exactly one way (positional)
- ✅ No conflicts with ComfyUI's dynamic argument merging
- ✅ Compatibility with `WrapperExecutor` and `patcher_extension`
- ✅ Python's calling convention is strictly followed

### Positional vs Keyword Argument Policy

**Rule**: When wrapping ComfyUI model methods, always match the original method's signature exactly, using positional arguments for parameters that are defined positionally (not in `**kwargs`).

**Rationale**:
- ComfyUI's patching system (`WrapperExecutor`, `patcher_extension`) may inject arguments in specific positions
- Mixing positional and keyword arguments for the same parameter causes Python-level errors
- Following the original signature ensures compatibility with all ComfyUI internal mechanisms

### Current Implementation Details

**`attention_mask` and `ref_latents`**:
- Currently passed as `None` because they are not used by the current Qwen Image pipelines in ComfyUI
- `attention_mask` is not utilized in the current implementation
- `ref_latents` is only used in image-to-image / ControlNet scenarios, which are handled through other mechanisms

**Future Compatibility**:
- If upstream ComfyUI changes to pass `attention_mask` via `transformer_options`, our implementation will need to be updated
- If `ref_latents` becomes a standard parameter for LoRA / Control implementations, we should extract it from `transformer_options` or `**kwargs` and pass it positionally

### Execution Flow

```
ComfyUI Sampler
    ↓
ComfyQwenImageWrapper.forward()
    ↓
ComfyQwenImageWrapper._execute_model()
    ↓
NunchakuQwenImageTransformer2DModel.forward()  [positional: guidance]
    ↓
WrapperExecutor.execute()
    ↓
NunchakuQwenImageTransformer2DModel._forward()  [positional: guidance]
```

**Key Points**:
- `guidance` normalization happens at the wrapper layer (`_execute_model`)
- All downstream calls use positional arguments
- No keyword argument conflicts can occur

---

## Related Issues

- **Issue #32**: `TypeError: NunchakuQwenImageTransformer2DModel._forward() got multiple values for argument 'guidance'`
  - **Status**: Fixed in v2.0.5
  - **Root Cause**: Keyword argument conflict with positional argument
  - **Solution**: Changed to positional argument passing

