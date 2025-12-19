# Diffsynth ControlNet Support for Nunchaku Models

## Overview

This document describes the implementation details for enabling diffsynth ControlNet to work with Nunchaku quantized Qwen Image models. The standard `QwenImageDiffsynthControlnet` node does not work with Nunchaku models because the Nunchaku model's `_forward` method does not call `double_block` patches, which are essential for diffsynth ControlNet functionality.

## Problem Statement

The standard `QwenImageDiffsynthControlnet` node registers `double_block` patches via `set_model_double_block_patch()`, which stores patches in `transformer_options["patches"]["double_block"]`. However, the Nunchaku model's `_forward` method in `ComfyUI-nunchaku/models/qwenimage.py` does not iterate through and call these patches during block processing, causing diffsynth ControlNet to be completely ignored.

## Solution

The solution consists of two main components:

1. **Created Node**: `NunchakuQwenImageDiffsynthControlnet` - A dedicated node for registering diffsynth ControlNet patches with Nunchaku models
2. **Modified File**: `ComfyUI-nunchaku/models/qwenimage.py` - Added `double_block` patch calling logic to the `_forward` method, matching the standard Qwen Image model implementation

---

## Created Files

### 1. `ComfyUI-QwenImageLoraLoader/nodes/controlnet.py`

**Purpose**: Create a dedicated diffsynth ControlNet loader node for Nunchaku models.

**Key Components**:

#### `DiffSynthCnetPatch` Class
- Handles diffsynth ControlNet patch logic
- Encodes input images to latent space
- Applies control block modifications at each transformer block
- Implements `__call__()` method that receives:
  - `x`: Original input tensor
  - `img`: Hidden states (image stream)
  - `block_index`: Current transformer block index
  - Returns modified `img` in kwargs

#### `ZImageControlPatch` Class
- Handles ZImage ControlNet patch logic (for `ZImage_Control` model type)
- More complex implementation with noise refiner support
- Handles both `double_block` and `noise_refiner` patches

#### `NunchakuQwenImageDiffsynthControlnet` Node Class

```python
class NunchakuQwenImageDiffsynthControlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "model_patch": ("MODEL_PATCH",),
                              "vae": ("VAE",),
                              "image": ("IMAGE",),
                              "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              },
                "optional": {"mask": ("MASK",)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "diffsynth_controlnet_nunchaku"
    EXPERIMENTAL = True
    CATEGORY = "advanced/loaders/qwen"

    def diffsynth_controlnet_nunchaku(self, model, model_patch, vae, image, strength, mask=None):
        model_patched = model.clone()
        image = image[:, :, :, :3]
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.ndim == 4:
                mask = mask.unsqueeze(2)
            mask = 1.0 - mask

        if isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control):
            patch = ZImageControlPatch(model_patch, vae, image, strength, mask=mask)
            model_patched.set_model_noise_refiner_patch(patch)
            model_patched.set_model_double_block_patch(patch)
        else:
            model_patched.set_model_double_block_patch(DiffSynthCnetPatch(model_patch, vae, image, strength, mask))
        return (model_patched,)
```

**How it works**:
- Clones the input model using `model.clone()`
- Creates either `DiffSynthCnetPatch` or `ZImageControlPatch` based on model type
- Registers the patch using `set_model_double_block_patch()`
- The patch is stored in `transformer_options["patches"]["double_block"]` for later execution

---

## Modified Files

### 1. `ComfyUI-nunchaku/models/qwenimage.py`

**Purpose**: Add `double_block` patch calling logic to the Nunchaku model's `_forward` method.

**Location**: `_forward` method, block processing loop (approximately lines 729-777)

#### Changes Made

**Before**:
```python
patches_replace = transformer_options.get("patches_replace", {})
blocks_replace = patches_replace.get("dit", {})

# Setup compute stream for offloading
compute_stream = torch.cuda.current_stream()
if self.offload:
    self.offload_manager.initialize(compute_stream)

for i, block in enumerate(self.transformer_blocks):
    with torch.cuda.stream(compute_stream):
        # ... block processing ...
        # No double_block patch calling
```

**After**:
```python
patches_replace = transformer_options.get("patches_replace", {})
patches = transformer_options.get("patches", {})  # ← Added
blocks_replace = patches_replace.get("dit", {})

transformer_options["total_blocks"] = len(self.transformer_blocks)  # ← Added
transformer_options["block_type"] = "double"  # ← Added

# Log if double_block patches are present
has_double_block_patches = "double_block" in patches
if has_double_block_patches:
    num_patches = len(patches["double_block"])
    logger.info(f"🔧 Nunchaku QwenImage: Applying {num_patches} double_block patch(es) for diffsynth ControlNet")

# Setup compute stream for offloading
compute_stream = torch.cuda.current_stream()
if self.offload:
    self.offload_manager.initialize(compute_stream)

for i, block in enumerate(self.transformer_blocks):
    transformer_options["block_index"] = i  # ← Added
    with torch.cuda.stream(compute_stream):
        # ... block processing ...
        
        # Apply double_block patches (for diffsynth ControlNet)  # ← Added
        if "double_block" in patches:
            for p_idx, p in enumerate(patches["double_block"]):
                logger.debug(f"🔧 Nunchaku QwenImage: Calling double_block patch {p_idx+1}/{num_patches} at block {i}/{len(self.transformer_blocks)-1}")
                out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]
```

#### Detailed Explanation of Added Code

1. **Retrieve `patches` from `transformer_options`**:
   ```python
   patches = transformer_options.get("patches", {})
   ```
   - Extracts the `patches` dictionary from `transformer_options`
   - Contains `double_block` patches registered by `NunchakuQwenImageDiffsynthControlnet`

2. **Set `transformer_options` metadata**:
   ```python
   transformer_options["total_blocks"] = len(self.transformer_blocks)
   transformer_options["block_type"] = "double"
   ```
   - Sets the total number of transformer blocks
   - Sets the block type to `"double"` (matching standard Qwen Image model behavior)
   - Allows patches to access block information

3. **Set `block_index` for each block**:
   ```python
   transformer_options["block_index"] = i
   ```
   - Sets the current block index in `transformer_options`
   - Allows patches to know which block is being processed

4. **Call `double_block` patches after each block**:
   ```python
   if "double_block" in patches:
       for p_idx, p in enumerate(patches["double_block"]):
           out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})
           hidden_states = out["img"]
           encoder_hidden_states = out["txt"]
   ```
   - Iterates through all `double_block` patches after each block processing
   - Calls each patch with required information:
     - `img`: Hidden states (image stream)
     - `txt`: Encoder hidden states (text stream)
     - `x`: Original input tensor
     - `block_index`: Current block index
     - `transformer_options`: Full transformer options dictionary
   - Updates `hidden_states` and `encoder_hidden_states` with patch return values

5. **Logging**:
   ```python
   logger.info(f"🔧 Nunchaku QwenImage: Applying {num_patches} double_block patch(es) for diffsynth ControlNet")
   logger.debug(f"🔧 Nunchaku QwenImage: Calling double_block patch {p_idx+1}/{num_patches} at block {i}/{len(self.transformer_blocks)-1}")
   ```
   - Logs when patches are detected and applied
   - Logs each patch call at debug level for detailed tracking

#### Why This Modification Was Necessary

The standard Qwen Image model (`comfy/ldm/qwen_image/model.py`) has the following code in its `_forward` method (lines 492-496):

```python
if "double_block" in patches:
    for p in patches["double_block"]:
        out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})
        hidden_states = out["img"]
        encoder_hidden_states = out["txt"]
```

The Nunchaku model's `_forward` method was missing this critical code, causing registered patches to never be called, resulting in diffsynth ControlNet being completely ignored.

---

### 2. `ComfyUI-QwenImageLoraLoader/__init__.py`

**Purpose**: Register the `NunchakuQwenImageDiffsynthControlnet` node with ComfyUI.

**Changes Made**:

```python
from .nodes.controlnet import NunchakuQwenImageDiffsynthControlnet

# Add version to classes before creating NODE_CLASS_MAPPINGS
NunchakuQwenImageDiffsynthControlnet.__version__ = __version__

NODE_CLASS_MAPPINGS["NunchakuQwenImageDiffsynthControlnet"] = NunchakuQwenImageDiffsynthControlnet

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuQwenImageDiffsynthControlnet": "Nunchaku Qwen Image Diffsynth Controlnet",
    # ...
}
```

**How it works**:
- Imports the `NunchakuQwenImageDiffsynthControlnet` class
- Registers it in `NODE_CLASS_MAPPINGS` so ComfyUI recognizes it as a node
- Adds a display name in `NODE_DISPLAY_NAME_MAPPINGS` for the UI

---

## Execution Flow

1. **Node Execution**: `NunchakuQwenImageDiffsynthControlnet` node is executed
   - `DiffSynthCnetPatch` or `ZImageControlPatch` is created
   - Patch is registered via `model.set_model_double_block_patch(patch)`
   - Patch is stored in `transformer_options["patches"]["double_block"]`

2. **Sampling Starts**: KSampler (standard or Nunchaku) is executed
   - `comfy.sample.sample()` is called
   - Model's `forward` method is invoked

3. **Model Forward**: `ComfyQwenImageWrapper.forward()` is called
   - `_execute_model()` is called
   - Nunchaku model's `forward()` is invoked

4. **Nunchaku Model Forward**: `NunchakuQwenImageTransformer2DModel.forward()` is called
   - `_forward()` method is invoked

5. **Block Processing Loop**: Inside `_forward()` method, each transformer block is processed
   - After each block processing, `double_block` patches are called (newly added code)
   - `DiffSynthCnetPatch.__call__()` is executed
   - ControlNet influence is applied to `hidden_states`

6. **Result**: Diffsynth ControlNet works correctly with Nunchaku models

---

## Summary

- **Created File**: `ComfyUI-QwenImageLoraLoader/nodes/controlnet.py`
- **Modified Files**:
  1. `ComfyUI-nunchaku/models/qwenimage.py` (Added `double_block` patch calling in `_forward` method)
  2. `ComfyUI-QwenImageLoraLoader/__init__.py` (Node registration)

**Key Insight**: By adding the `double_block` patch calling logic to the Nunchaku model's `_forward` method (matching the standard Qwen Image model implementation), diffsynth ControlNet now works correctly with Nunchaku quantized models.

**Why Standard Node Doesn't Work**: The standard `QwenImageDiffsynthControlnet` node registers patches correctly, but the Nunchaku model's `_forward` method doesn't call them. The `NunchakuQwenImageDiffsynthControlnet` node is required to work with the modified Nunchaku model implementation.

