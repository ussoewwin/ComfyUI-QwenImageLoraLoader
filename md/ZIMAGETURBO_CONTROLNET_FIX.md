# Detailed Explanation of Z-Image-Turbo ControlNet Support Fix in ComfyUI-QwenImageLoraLoader

## Reason for Additional Fix

The unofficial Z-Image-Turbo loader (`ComfyUI-nunchaku-unofficial-z-image-turbo-loader`) monkey-patched `NunchakuZImageTransformer2DModel.forward` to accept `transformer_options` and `control` parameters.

However, the `ComfyZImageTurboWrapper` class in ComfyUI-QwenImageLoraLoader was not correctly passing these parameters to the underlying `NunchakuZImageTransformer2DModel.forward`.

`ComfyZImageTurboWrapper` is a wrapper that separates LoRA composition and forward pass, passing arguments received in the `forward` method to `NunchakuZImageTransformer2DModel.forward` via the `_execute_model` method. We needed to add `transformer_options` and `control` to this transmission path.

## Modified File

`ComfyUI-QwenImageLoraLoader/wrappers/zimageturbo.py`

## Fix Details

### Fix 1: forward Method Signature (Lines 76-85)

#### Before

```python
def forward(
        self,
        x,
        timestep,
        context=None,
        y=None,
        guidance=None,
        **kwargs,
):
```

#### After

```python
def forward(
        self,
        x,
        timestep,
        context=None,
        y=None,
        guidance=None,
        control=None,
        transformer_options={},
        **kwargs,
):
```

#### Explanation

Added `control` and `transformer_options` to the `forward` method signature to receive them from ComfyUI's ModelPatcher.

### Fix 2: transformer_options Merge Processing in forward Method (Lines 93-100)

#### Fix Content

```python
# Remove guidance, transformer_options, and attention_mask from kwargs
if "guidance" in kwargs:
    kwargs.pop("guidance")
if "transformer_options" in kwargs:
    if isinstance(transformer_options, dict) and isinstance(kwargs["transformer_options"], dict):
        transformer_options = {**transformer_options, **kwargs.pop("transformer_options")}
    else:
        kwargs.pop("transformer_options")
if "attention_mask" in kwargs:
    kwargs.pop("attention_mask")
```

#### Explanation

If `transformer_options` is included in `kwargs`, merge it with the explicit parameter. The explicit parameter takes precedence.

### Fix 3: Argument Passing to _execute_model Method (Lines 219, 221)

#### Fix Content

Modified calls to `_execute_model` within the `forward` method to pass `control` and `transformer_options`:

```python
with cache_context(self._cache_context):
    out = self._execute_model(x, timestep, context, y, guidance, control, transformer_options, **kwargs)
else:
    out = self._execute_model(x, timestep, context, y, guidance, control, transformer_options, **kwargs)
```

#### Explanation

By passing `control` and `transformer_options` to `_execute_model`, they can be transmitted to the underlying `NunchakuZImageTransformer2DModel.forward`.

### Fix 4: _execute_model Method Signature (Line 239)

#### Before

```python
def _execute_model(self, x, timestep, context, y, guidance, **kwargs):
```

#### After

```python
def _execute_model(self, x, timestep, context, y, guidance, control, transformer_options, **kwargs):
```

#### Explanation

Added `control` and `transformer_options` to the `_execute_model` signature.

### Fix 5: Transmission When Using customized_forward (Lines 282-300)

#### Fix Content

Processing when `customized_forward` is set (when using the custom forward from the unofficial ZIT loader):

```python
if self.customized_forward:
    with torch.inference_mode():
        forward_kwargs_cleaned = {k: v for k, v in self.forward_kwargs.items() if k not in ("guidance", "ref_latents", "transformer_options", "attention_mask")}
        transformer_options_cleaned = dict(transformer_options) if transformer_options else {}
        transformer_options_cleaned.pop("guidance", None)
        transformer_options_cleaned.pop("ref_latents", None)
        
        return self.customized_forward(
            self.model,
            hidden_states=x,
            encoder_hidden_states=context,
            timestep=timestep,
            guidance=guidance if self.config.get("guidance_embed", False) else None,
            ref_latents=ref_latents_value,
            control=control,
            transformer_options=transformer_options_cleaned,
            **forward_kwargs_cleaned,
            **kwargs,
        )
```

#### Explanation

Pass `control` and `transformer_options` when calling `customized_forward`. Remove `guidance` and `ref_latents` from `transformer_options` (to avoid duplication).

### Fix 6: Transmission When Directly Calling forward (Lines 389-409)

#### Fix Content

Processing when `customized_forward` is not set (when directly calling `NunchakuZImageTransformer2DModel.forward`):

```python
# Call Z-Image-Turbo forward with correct signature
# Pass control and transformer_options to allow Model Patcher (double_block patches) to work
# Check if the model's forward method accepts control parameter
import inspect
forward_sig = inspect.signature(self.model.forward)
forward_params = set(forward_sig.parameters.keys())

zimage_kwargs_clean = {k: v for k, v in zimage_kwargs.items() if k not in ('control', 'transformer_options')}

# Build kwargs based on what the forward method accepts
forward_kwargs = zimage_kwargs_clean.copy()
if 'control' in forward_params:
    forward_kwargs['control'] = control
if 'transformer_options' in forward_params:
    forward_kwargs['transformer_options'] = transformer_options

model_output = self.model(
    x_list,
    t_zimage,
    cap_feats=cap_feats,
    **forward_kwargs
)
```

#### Explanation

- Check the signature of `self.model.forward` using `inspect.signature`
- Add `control` and `transformer_options` to `forward_kwargs` only if they are accepted
- This allows the `_patched_forward` (monkey-patched forward method) from the unofficial ZIT loader to correctly receive `transformer_options` and `control`

## Key Design Decisions

### Dynamic Parameter Checking with inspect.signature

Using `inspect.signature` to dynamically check whether the `forward` method accepts `control` and `transformer_options`. This ensures:

1. Backward compatibility is maintained (no errors in older versions that don't accept these parameters)
2. Works appropriately regardless of whether the monkey patch from the unofficial ZIT loader is applied
3. Easy to adapt to future extensions

### Integration with Unofficial ZIT Loader

This fix enables the following integration flow:

1. ComfyUI's ModelPatcher passes `transformer_options` and `control` to `ComfyZImageTurboWrapper.forward`
2. `ComfyZImageTurboWrapper.forward` passes these to `_execute_model`
3. `_execute_model` checks with `inspect.signature` before passing to `NunchakuZImageTransformer2DModel.forward`
4. The `_patched_forward` (monkey-patched forward) from the unofficial ZIT loader receives `transformer_options` and `control` and applies the `double_block` patch

## Summary

Modified the `ComfyZImageTurboWrapper` class in ComfyUI-QwenImageLoraLoader to correctly pass `control` and `transformer_options` parameters.

Main changes:

1. Added `control` and `transformer_options` to the `forward` method signature
2. Added `control` and `transformer_options` to the `_execute_model` method signature
3. Implemented using `inspect.signature` to pass these parameters only if the underlying `forward` method accepts them

This enables the ControlNet support monkey patch added in the unofficial ZIT loader to work correctly, achieving compatibility between ControlNet and LoRA.

