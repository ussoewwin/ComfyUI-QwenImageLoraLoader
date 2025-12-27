# Diffsynth (ControlNet) Official NextDiT Support - Complete Technical Documentation

## Reason for Development

ComfyUI-QwenImageLoraLoader had implemented Diffsynth (ControlNet) support for the unofficial Z-Image loader (`ComfyUI-nunchaku-unofficial-z-image-turbo-loader`), but it did not work with ComfyUI-nunchaku's official Z-Image loader (ComfyUI Lumina2 / NextDiT).

**Root Causes**:

1. **Parameter Passing Issues Due to Forward Signature Differences**
   - Unofficial: `NunchakuZImageTransformer2DModel.forward(x, t, cap_feats=..., **kwargs)` where `transformer_options` and `control` are received as explicit parameters via monkey-patch
   - Official: `NextDiT.forward(x, timesteps, context, num_tokens, attention_mask=None, **kwargs)` where `transformer_options` is read via `**kwargs` (`kwargs.get("transformer_options", {})`)

2. **transformer_options Transmission Path**
   - ComfyUI's ModelPatcher registers Diffsynth patches in `transformer_options["patches"]["double_block"]`. NextDiT reads this `patches["double_block"]` in its internal `_forward()` method and applies patches after each transformer block. However, if `transformer_options` is not passed via `**kwargs`, the patches do not reach NextDiT and Diffsynth does not work.

3. **Limitations of Existing Implementation**
   - `ComfyZImageTurboWrapper._execute_model()` dynamically detected the forward signature using `inspect.signature()`, but when `transformer_options` does **not exist as an explicit parameter** (i.e., when passed via `**kwargs`), it could be missed in the conditional branch.

4. **Maintaining Compatibility with Unofficial Loader**
   - Processing for the unofficial loader (via `customized_forward` or direct forward calls) was already working, so the design needed to add official support without breaking the unofficial loader's behavior.

## Modified Files

### `ComfyUI-QwenImageLoraLoader/wrappers/zimageturbo.py`

**Modification Location: NextDiT Branch in `_execute_model()` Method (lines 442-472)**

**Problem Before Fix**:

```python
# Branch 2: ComfyUI Lumina2 / NextDiT signature
if "context" in forward_params and "num_tokens" in forward_params:
    # ... (processing of x, context, num_tokens) ...
    
    forward_kwargs = {}
    if "attention_mask" in forward_params:
        forward_kwargs["attention_mask"] = attention_mask_value
    if "transformer_options" in forward_params:
        forward_kwargs["transformer_options"] = transformer_options
    if "control" in forward_params:
        forward_kwargs["control"] = control
    
    return self.model(
        x_tensor,
        timestep,
        context=context,
        num_tokens=num_tokens_value,
        **forward_kwargs,
        **kwargs,
    )
```

**Problem**: When `transformer_options` does not exist as an explicit parameter (not in `forward_params`), it was not added to `forward_kwargs`, and might not be included in `**kwargs` either. Since NextDiT reads it via `**kwargs.get("transformer_options", {})`, if it is not passed, Diffsynth patches do not reach NextDiT.

**After Fix**:

```python
# Branch 2: ComfyUI Lumina2 / NextDiT signature
# forward(x, timesteps, context, num_tokens, attention_mask=None, **kwargs)
if "context" in forward_params and "num_tokens" in forward_params:
    # ... (processing of x, context, num_tokens) ...
    
    # NextDiT forward is: forward(x, timesteps, context, num_tokens, attention_mask=None, **kwargs)
    # It consumes `transformer_options` via **kwargs (not a named parameter),
    # so we must pass it when the function supports VAR_KEYWORD.
    supports_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_sig.parameters.values()
    )
    
    forward_kwargs = {}
    if "attention_mask" in forward_params:
        forward_kwargs["attention_mask"] = attention_mask_value
    
    if supports_kwargs:
        # Ensure patches (e.g. double_block from DiffSynth/ControlNet) reach NextDiT
        forward_kwargs["transformer_options"] = transformer_options
        # Safe to forward; NextDiT ignores it if unused.
        if control is not None:
            forward_kwargs["control"] = control
    else:
        if "transformer_options" in forward_params:
            forward_kwargs["transformer_options"] = transformer_options
        if "control" in forward_params:
            forward_kwargs["control"] = control
    
    # NOTE: NextDiT expects `context` and `num_tokens` as required args
    return self.model(
        x_tensor,
        timestep,
        context=context,
        num_tokens=num_tokens_value,
        **forward_kwargs,
        **kwargs,
    )
```

**Technical Meaning**:

1. **VAR_KEYWORD Detection**
   - `inspect.Parameter.VAR_KEYWORD` represents `**kwargs` in Python function signatures. `any(p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_sig.parameters.values())` checks whether the forward method supports `**kwargs`.

2. **When `supports_kwargs` is True**
   - When `**kwargs` is supported (like NextDiT), `transformer_options` and `control` are always added to `forward_kwargs`. This ensures they are passed via `**forward_kwargs`. Since NextDiT reads via `kwargs.get("transformer_options", {})`, this allows Diffsynth patches to reach NextDiT.

3. **When `supports_kwargs` is False (Backward Compatibility)**
   - When `**kwargs` is not supported (rare, but for safety), it falls back to the traditional approach of adding only if they exist as explicit parameters.

4. **Handling of `control`**
   - Similarly, when `supports_kwargs` is True, `control` is added only when `control is not None` (safe because NextDiT does not error even if unused). When False, explicit parameter checks are performed.

## Official NextDiT Implementation Verification

The `NextDiT` class in `ComfyUI/comfy/ldm/lumina/model.py`:

```python
def forward(self, x, timesteps, context, num_tokens, attention_mask=None, **kwargs):
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
        self._forward,
        self,
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, kwargs.get("transformer_options", {}))
    ).execute(x, timesteps, context, num_tokens, attention_mask, **kwargs)

def _forward(self, x, timesteps, context, num_tokens, attention_mask=None, transformer_options={}, **kwargs):
    # ...
    patches = transformer_options.get("patches", {})
    # ...
    transformer_options["total_blocks"] = len(self.layers)
    transformer_options["block_type"] = "double"
    for i, layer in enumerate(self.layers):
        transformer_options["block_index"] = i
        img = layer(img, mask, freqs_cis, adaln_input, transformer_options=transformer_options)
        if "double_block" in patches:
            for p in patches["double_block"]:
                out = p({...})  # Apply Diffsynth patch
```

**Key Point**: NextDiT's `forward()` receives `**kwargs` and extracts the dictionary via `kwargs.get("transformer_options", {})`. In `_forward()`, it reads `double_block` patches via `patches = transformer_options.get("patches", {})` and applies them after each block. Therefore, **ensuring `transformer_options` is passed via `**kwargs` is a necessary condition for Diffsynth to work**.

## Differences and Compatibility with Unofficial Loader

**Unofficial Loader (`ComfyUI-nunchaku-unofficial-z-image-turbo-loader`)**:
- Monkey-patches `NunchakuZImageTransformer2DModel.forward` to add `transformer_options` and `control` as **explicit parameters**
- Signature: `_patched_forward(self, x, t, cap_feats=None, *args, control=None, transformer_options=None, **kwargs)`
- In this case, it is processed in Branch 1 (when `cap_feats` exists) of `ComfyZImageTurboWrapper._execute_model()`, and works correctly with the existing implementation

**Official Loader (NextDiT)**:
- Signature: `forward(x, timesteps, context, num_tokens, attention_mask=None, **kwargs)`
- `transformer_options` is received via `**kwargs` (not an explicit parameter)
- With this fix, Branch 2 checks `supports_kwargs` and ensures it is passed when `**kwargs` is supported

**Compatibility Mechanism**:
- Branch 1 (Unofficial): When `cap_feats` exists, processed with existing implementation (checks explicit parameters via `inspect.signature`)
- Branch 2 (Official): When `context` and `num_tokens` exist, checks `VAR_KEYWORD` and ensures passing via `**kwargs`

This allows **Diffsynth to work with both unofficial and official loaders**, and the existing unofficial users are not affected by the addition of official support.

## Technical Design Philosophy

### 1. Dynamic Detection via VAR_KEYWORD

**Problem**: Need to determine at runtime whether `transformer_options` is an explicit parameter or passed via `**kwargs`

**Solution**:
- Detects `**kwargs` support using `inspect.Parameter.VAR_KEYWORD`, and when supported, always adds to `forward_kwargs`

**Benefits**:
- Supports both explicit parameters and `**kwargs`
- Flexible to future signature changes
- No need for type checks or isinstance checks (signature-based detection)

### 2. Safe Parameter Transmission

**Policy**:
- When `supports_kwargs` is True: Always add `transformer_options` and `control` (necessary because NextDiT reads from `**kwargs`)
- When `supports_kwargs` is False: Fall back to explicit parameter checks (backward compatibility)

**Result**:
- Diffsynth works reliably with official NextDiT
- Unofficial loader behavior maintained with existing implementation (no impact)
- Can adapt to future new loaders

### 3. Maintaining Compatibility with Unofficial Loader

**Policy**:
- Do not change Branch 1 (unofficial) at all
- Add fixes only to Branch 2 (official)
- Both branches operate independently, so they do not affect each other

**Result**:
- No impact on existing unofficial users
- Official users can use Diffsynth without additional configuration
- Implementation with minimal code complexity

## Operation Flow

### When Using Official NextDiT + Diffsynth

1. Diffsynth node execution: `NunchakuQwenImageDiffsynthControlnet.diffsynth_controlnet_nunchaku()` registers patches via `set_model_double_block_patch()`
2. ComfyUI ModelPatcher: Registers patches in `transformer_options["patches"]["double_block"]`
3. `ComfyZImageTurboWrapper.forward()`: Receives `transformer_options` and passes it to `_execute_model()`
4. `_execute_model()`: Detects NextDiT's forward signature using `inspect.signature()`
5. Branch 2: Confirms existence of `context` and `num_tokens`, processes as NextDiT format
6. VAR_KEYWORD detection: Checks `**kwargs` support via `supports_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD ...)`
7. Parameter addition: When `supports_kwargs` is True, adds via `forward_kwargs["transformer_options"] = transformer_options`
8. NextDiT.forward() call: `self.model(x_tensor, timestep, context=context, num_tokens=num_tokens_value, **forward_kwargs, **kwargs)`
9. NextDiT._forward(): Extracts `transformer_options` via `kwargs.get("transformer_options", {})`
10. Patch application: Extracts `double_block` patches via `patches = transformer_options.get("patches", {})` and applies after each block

### When Using Unofficial Loader + Diffsynth

1. Diffsynth node execution: Similarly registers patches
2. `ComfyZImageTurboWrapper.forward()`: Receives `transformer_options` and passes it to `_execute_model()`
3. `_execute_model()`: Detects unofficial forward signature using `inspect.signature()`
4. Branch 1: Confirms existence of `cap_feats`, processes as unofficial format
5. Explicit parameter check: Checks explicit parameters via `if "transformer_options" in forward_params`
6. Parameter addition: When it exists as an explicit parameter, adds via `forward_kwargs["transformer_options"] = transformer_options`
7. `_patched_forward()` call: Monkey-patched forward receives `transformer_options` as an explicit parameter
8. Patch application: Applies patches after each block via `_apply_double_block_patches()`

## Conclusion

The official NextDiT support for Diffsynth (ControlNet) has achieved the following:

1. **Official Loader Support**: Diffsynth (ControlNet) works with NextDiT format models
2. **Reliable Parameter Transmission**: `VAR_KEYWORD` detection ensures `transformer_options` is reliably passed via `**kwargs`
3. **Compatibility with Unofficial Loader**: Unofficial loader behavior is maintained with existing implementation, no impact
4. **Minimal Code Changes**: Existing unofficial branch unchanged, fixes added only to official branch

With this design, users can use either the official or unofficial loader, and Diffsynth (ControlNet) will work appropriately with both.

