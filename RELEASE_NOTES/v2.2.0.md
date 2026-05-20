<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.2.0.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# Nunchaku Z-Image-Turbo LoRA Loader V3 - Complete Technical Documentation

## Reason for Development

We had been prototyping a LoRA loader for our custom unofficial Nunchaku Z-Image Turbo loader ([Unofficial Loader Test Records](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/md/technical_explanation.md)). With the release of ComfyUI-nunchaku's official Z-Image-Turbo model loader (which uses NextDiT), we reorganized this LoRA loader to work with the official model loader. This was not due to any issues with the official model loader, but rather to adapt our existing prototype to work with ComfyUI-nunchaku's official implementation.

**Please use V3 for ComfyUI-nunchaku's official Z-Image-Turbo model loader. V2 is for testing purposes only with our custom unofficial loader.**

The following differences required adaptation:

**Model Structure Differences**

- Custom unofficial loader: `NunchakuZImageTransformer2DModel` (diffusers format, `to_qkv` / `feed_forward.net.0.proj`)
- ComfyUI-nunchaku's official model loader: `NextDiT` (ComfyUI Lumina2 format, `attention.qkv` / `feed_forward.w1/w2/w3` or `w13`)

**Forward Signature Differences**

- Custom unofficial: `forward(x_list: List[torch.Tensor], t_zimage, cap_feats=...)`
- ComfyUI-nunchaku's official: `forward(x: torch.Tensor, timesteps, context, num_tokens, attention_mask=...)`

**LoRA Mapping Mismatch**

- The existing v2 loader was designed for the custom unofficial structure (`to_qkv` / `net.0.proj`), and failed to resolve keys for ComfyUI-nunchaku's official structure (`qkv` / `w13`)

**Maintenance of v2 for Testing**

- v2 code remains for testing purposes with our custom unofficial loader
- v3 is provided as a new file to work with ComfyUI-nunchaku's official model loader
- Both use the same `compose_loras_v2`, but work correctly via automatic mapping switching

## Newly Created Files

### 1. `ComfyUI-QwenImageLoraLoader/nodes/lora/zimageturbo_v3.py`

**Purpose**: LoRA stack node for ComfyUI-nunchaku's official Z-Image-Turbo model loader (`NextDiT`)

**Main Implementation**:

The class name is `NunchakuZImageTurboLoraStackV2`, but it is actually dedicated to V3 (registration name changed via GENERATED_NODES)

**Key Logic (lines 169-191)**:

```python
elif model_wrapper_type_name == "NextDiT" and model_wrapper_module == "comfy.ldm.lumina.model":
    logger.info("⚡ Official loader detected (NextDiT), wrapping with ComfyZImageTurboWrapper")
    transformer = model_wrapper
    logger.info(f"🚀 Creating ComfyZImageTurboWrapper for NextDiT with cpu_offload='{cpu_offload}'")
    wrapped_model = ComfyZImageTurboWrapper(
        transformer,
        getattr(transformer, 'config', {}),
        None,
        {},
        cpu_offload,
        4.0,
    )
    model.model.diffusion_model = wrapped_model
    model_wrapper = wrapped_model
    transformer = model_wrapper.model
else:
    logger.error(f"❌ Model type mismatch! Type: {model_wrapper_type_name}, Module: {model_wrapper_module}")
    logger.error("V3 is for ComfyUI-nunchaku's official Z-Image-Turbo model loader only. For custom unofficial loader, please use V2.")
    raise TypeError(f"This LoRA loader (V3) only works with ComfyUI-nunchaku's official Z-Image-Turbo model loader, but got {model_wrapper_type_name}.")
```

**Technical Meaning**:

- When `NextDiT` is detected, wrap it with `ComfyZImageTurboWrapper`
- By directly replacing `diffusion_model`, maintain compatibility with existing ComfyUI workflows
- Error messages clearly indicate that V3 is for ComfyUI-nunchaku's official model loader

**Registration (lines 226-232)**:

```python
GENERATED_NODES = {
    "NunchakuZImageTurboLoraStackV3": NunchakuZImageTurboLoraStackV2
}

GENERATED_DISPLAY_NAMES = {
    "NunchakuZImageTurboLoraStackV3": "Nunchaku Z-Image-Turbo LoRA Stack V3"
}
```

### 2. `ComfyUI-QwenImageLoraLoader/js/zimageturbo_lora_dynamic_v3.js`

**Purpose**: Dynamic UI for v3 node (variable display of LoRA slot count)

**Main Implementation**:

- Uses the same UI logic as v2 (line 8: `name: "nunchaku.zimageturbo_lora_dynamic_v3"` to avoid conflicts with v2)
- Line 17: Filters for v3-only with `if (node.comfyClass !== "NunchakuZImageTurboLoraStackV3")`

**Technical Meaning**:

- Separates JavaScript extensions for v2 and v3, avoiding name collisions
- UI logic is reused from v2 (no changes needed)

## Modified Files

### 1. `ComfyUI-QwenImageLoraLoader/wrappers/zimageturbo.py`

**Modification 1: NextDiT Support in forward Method (lines 425-458)**

```python
if "context" in forward_params and "num_tokens" in forward_params:
    if isinstance(x, list):
        x_stack = torch.stack([t for t in x if t is not None], dim=0)
        if x_stack.ndim == 5 and x_stack.shape[2] == 1:
            x_stack = x_stack.squeeze(2)
        x_tensor = x_stack
    else:
        x_tensor = x

    if num_tokens_value is None and isinstance(context, torch.Tensor) and context.ndim >= 2:
        num_tokens_value = context.shape[1]

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

**Technical Meaning**:

- Dynamically detects forward signature using `inspect.signature`
- When `context` and `num_tokens` exist, processes as NextDiT format (used by ComfyUI-nunchaku's official model loader)
- If `num_tokens` is not specified, automatically derives from `context.shape[1]`
- Passes `attention_mask`, `transformer_options`, and `control` as keyword arguments to maintain compatibility with extension features like ControlNet

**Modification 2: Extraction of `num_tokens` and `attention_mask` from kwargs (lines 241-242)**

```python
ref_latents_value = kwargs.pop("ref_latents", None)
num_tokens_value = kwargs.pop("num_tokens", None)
attention_mask_value = kwargs.pop("attention_mask", None)
```

**Technical Meaning**:

- ComfyUI passes `num_tokens` and `attention_mask` via kwargs, so they are extracted beforehand
- `kwargs.pop` prevents duplicate parameter passing

### 2. `ComfyUI-QwenImageLoraLoader/nunchaku_code/lora_qwen.py`

**Modification 1: NextDiT Mapping Definitions (lines 30-54)**

```python
ZIMAGE_NEXTDIT_UNPATCHED_KEY_MAPPING = [
    (re.compile(r"^(layers)[._](\d+)[._]attention[._]to[._]([qkv])$"), r"\1.\2.attention.qkv", "qkv", lambda m: m.group(3).upper()),
    (re.compile(r"^(layers)[._](\d+)[._]attention[._]to[._]out(?:[._]0)?$"), r"\1.\2.attention.out", "regular", None),
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._]w1$"), r"\1.\2.feed_forward.w1", "regular", None),
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._]w2$"), r"\1.\2.feed_forward.w2", "regular", None),
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._]w3$"), r"\1.\2.feed_forward.w3", "regular", None),
]

ZIMAGE_NEXTDIT_NUNCHAKU_PATCHED_KEY_MAPPING = [
    (re.compile(r"^(layers)[._](\d+)[._]attention[._]to[._]([qkv])$"), r"\1.\2.attention.qkv", "qkv", lambda m: m.group(3).upper()),
    (re.compile(r"^(layers)[._](\d+)[._]attention[._]to[._]out(?:[._]0)?$"), r"\1.\2.attention.out", "regular", None),
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._](w1|w3)$"), r"\1.\2.feed_forward.w13", "glu", lambda m: m.group(3)),
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._]w2$"), r"\1.\2.feed_forward.w2", "regular", None),
]
```

**Technical Meaning**:

- UNPATCHED: Mapping for vanilla NextDiT (with `w1/w2/w3` separated)
- NUNCHAKU_PATCHED: Mapping for when ComfyUI-nunchaku replaces `feed_forward` with `w13` (w1+w3 fused)
- The `glu` group specification causes `_fuse_glu_lora` to fuse and apply w1 and w3

**Modification 2: Model Structure Detection and Automatic Mapping Switching (lines 984-1014)**

```python
global _ACTIVE_KEY_MAPPING
prev_mapping = _ACTIVE_KEY_MAPPING
try:
    nextdit_markers = (
        "layers.0.attention.qkv",
        "layers.0.attention.out",
        "layers.0.feed_forward.w1",
        "layers.0.feed_forward.w2",
        "layers.0.feed_forward.w3",
        "layers.0.feed_forward.w13",
    )
    is_nextdit_style = any(_get_module_by_name(model, p) is not None for p in nextdit_markers)
    if is_nextdit_style:
        has_w13 = _get_module_by_name(model, "layers.0.feed_forward.w13") is not None
        if has_w13:
            _ACTIVE_KEY_MAPPING = ZIMAGE_NEXTDIT_NUNCHAKU_PATCHED_KEY_MAPPING + KEY_MAPPING
        else:
            _ACTIVE_KEY_MAPPING = ZIMAGE_NEXTDIT_UNPATCHED_KEY_MAPPING + KEY_MAPPING
    else:
        _ACTIVE_KEY_MAPPING = None
except Exception:
    _ACTIVE_KEY_MAPPING = None
```

**Technical Meaning**:

- Dynamically detects model structure using `_get_module_by_name`
- Determines nunchaku-patched/unpatched by the presence of `layers.0.feed_forward.w13`
- Sets appropriate mapping to `_ACTIVE_KEY_MAPPING`, which is prioritized in `_classify_and_map_key` (line 249)

**Modification 3: Active Mapping Priority in _classify_and_map_key (line 249)**

```python
mapping_to_use = _ACTIVE_KEY_MAPPING if _ACTIVE_KEY_MAPPING is not None else KEY_MAPPING
```

**Technical Meaning**:

- When `_ACTIVE_KEY_MAPPING` is set (NextDiT detected), it is prioritized
- When not set, uses the traditional `KEY_MAPPING` (for custom unofficial/v2)
- This allows v2 and v3 to share the same `compose_loras_v2` function while switching only the mapping

### 3. `ComfyUI-QwenImageLoraLoader/__init__.py`

**Modification: V3 Node Import and Registration (lines 24, 33-34, 40, 60)**

```python
from .nodes.lora.zimageturbo_v3 import GENERATED_NODES as ZIMAGETURBO_V3_NODES, GENERATED_DISPLAY_NAMES as ZIMAGETURBO_V3_NAMES

for node_class in ZIMAGETURBO_V3_NODES.values():
    node_class.__version__ = __version__

NODE_CLASS_MAPPINGS.update(ZIMAGETURBO_V3_NODES)

NODE_DISPLAY_NAME_MAPPINGS = {
    **ZIMAGETURBO_V3_NAMES
}
```

**Technical Meaning**:

- Registers v3 node into ComfyUI's node system
- v3 is provided for use with ComfyUI-nunchaku's official Z-Image-Turbo model loader

## Technical Design Philosophy

### 1. Automatic Mapping Switching

**Problem**: Module names differ between custom unofficial (`to_qkv`/`net.0.proj`) and ComfyUI-nunchaku's official (`qkv`/`w13`)

**Solution**:

- Uses a global variable `_ACTIVE_KEY_MAPPING` to switch mappings at runtime
- Detects model structure within `compose_loras_v2` and sets appropriate mapping
- Prioritizes `_ACTIVE_KEY_MAPPING` in `_classify_and_map_key`

**Benefits**:

- v2/v3 share the same `compose_loras_v2` function
- Minimizes code duplication
- To support new model structures, only mapping definitions need to be added

### 2. Dynamic Forward Signature Detection

**Problem**: Forward signatures differ between custom Z-Image (`cap_feats`) and ComfyUI-nunchaku's official NextDiT (`context`/`num_tokens`)

**Solution**:

- Dynamically retrieves forward parameters using `inspect.signature`
- Branches to Z-Image if `cap_feats` exists, or NextDiT if `context` and `num_tokens` exist

**Benefits**:

- Single `ComfyZImageTurboWrapper` handles both formats
- No need for type checks or isinstance checks (high flexibility)

### 3. GLU Fusion Handling

**Problem**: ComfyUI-nunchaku fuses `feed_forward.w1/w3` into `w13`

**Solution**:

- `ZIMAGE_NEXTDIT_NUNCHAKU_PATCHED_KEY_MAPPING` maps LoRA's `w1`/`w3` to `w13`
- Specifies `glu` as the group, and applies via fusion using `_fuse_glu_lora`

**Technical Details**:

- `_fuse_glu_lora` is a function that mathematically fuses w1 and w3 LoRAs and applies them to w13
- This allows LoRA files in `w1`/`w3` format to be correctly applied to patched models (`w13`)

### 4. v2 Maintenance for Testing

**Policy**:

- v2 remains for testing purposes with our custom unofficial loader
- v3 is provided as a new file to work with ComfyUI-nunchaku's official Z-Image-Turbo model loader
- Both use the same `compose_loras_v2`, but work correctly via automatic mapping switching

**Result**:

- v3 is for use with ComfyUI-nunchaku's official Z-Image-Turbo model loader
- Users should use V3 when working with ComfyUI-nunchaku's official model loader

## Operation Flow

### When Using V3 Node (ComfyUI-nunchaku's Official Model Loader)

- Node execution: `NunchakuZImageTurboLoraStackV3.load_lora_stack()`
- Model detection: Checks `model_wrapper_type_name == "NextDiT"`
- Wrapper creation: Wraps `NextDiT` with `ComfyZImageTurboWrapper`
- LoRA composition: Calls `compose_loras_v2()`
- Mapping switch: Detects `layers.0.feed_forward.w13` using `_get_module_by_name()`
- Mapping application: Sets `ZIMAGE_NEXTDIT_NUNCHAKU_PATCHED_KEY_MAPPING` to `_ACTIVE_KEY_MAPPING`
- Key resolution: Uses `_ACTIVE_KEY_MAPPING` in `_classify_and_map_key()`
- LoRA application: Applies `w1`/`w3` → `w13` (GLU fusion), `to_q/k/v` → `qkv` (QKV fusion)

### During Forward Execution (NextDiT)

- Signature detection: Checks `context` and `num_tokens` using `inspect.signature()`
- Branching: Processes as NextDiT format
- Argument preparation: Unifies `x` to tensor format, derives `num_tokens`
- Call: `model.forward(x, timestep, context=context, num_tokens=num_tokens, ...)`

## Conclusion

The development of v3 has achieved the following:

- **ComfyUI-nunchaku Official Model Loader Support**: Enables LoRA application to models loaded with ComfyUI-nunchaku's official Z-Image-Turbo model loader (`NextDiT` format)
- **Automatic Mapping Switching**: Detects model structure and automatically selects appropriate mapping
- **Code Reuse**: Extended existing `compose_loras_v2` and `ComfyZImageTurboWrapper`, minimizing duplication

**Please use V3 when working with ComfyUI-nunchaku's official Z-Image-Turbo model loader.**
