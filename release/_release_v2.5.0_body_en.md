<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.5.0.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## 1. Complete Explanation of the Issue

The issue where "ControlNet (DiffSynth version) does not work with Nunchaku Qwen Image" was a **complex problem caused by a combination of architectural differences and missing specifications**. Here is a step-by-step explanation.

### 1. Differences in Nunchaku's Model Optimization Approach
Normally, ComfyUI's Transformer (DiT) models have a standard loop structure that processes internal blocks (layers) sequentially.
When Nunchaku optimizes Z Image, it takes the approach of replacing only the components within the blocks (such as Attention) with Nunchaku versions, while maintaining this standard loop.
However, **for the Nunchaku patch for Qwen Image, due to processing speed and structural reasons, it completely discards ComfyUI's standard loop structure and redefines and executes its own `_forward` loop**.

### 2. Ignoring the Standard Patch (`double_block`)
In standard ComfyUI, ControlNet intervenes in the processing of each block using a mechanism called `patches["double_block"]`.
However, as mentioned above, the custom loop of Nunchaku for Qwen Image does not have the code to process this `patches["double_block"]`, and **completely ignores it**. This was the first cause of "ControlNet not working in Nunchaku Qwen Image".

### 3. Fatal Information Loss in the Alternative Patch Route (`x` disappearing)
Therefore, we modified the code to inject ControlNet using another patching feature called `patches_replace["dit"]`, which is also read by Nunchaku's custom loop. As a result, the ControlNet processing itself started executing, but **another fatal problem** occurred.

ControlNet needs to dynamically resize the reference image (condition image like Canny) according to the resolution of the image being generated (Dynamic Resizing). For this resizing, the tensor `x` (latent variable), which holds the base generation resolution information, is absolutely necessary.
However, Nunchaku's custom loop was designed **not to pass this `x` as an argument** when calling `patches_replace`.

### 4. Silent Failure Due to Resolution Mismatch
Because `x` was not passed, ControlNet could not determine the "current generation resolution" and skipped the resizing process.
As a result, the tensor addition process was performed while the actual generated image size (e.g., 1216x1216) and the default size of the reference image (e.g., 768x768) were **mismatched**.
In terms of PyTorch tensor operations, it does not result in an error and silently passes through the processing. However, because data with completely mismatched positions and sizes were added, the effect of ControlNet was not reflected in the final generated image (it visually appeared as if it was not working).

---

## 2. Details of the Solution

To solve this problem, we modified the node side of ComfyUI-QwenImageLoraLoader.

### 1. Added/Modified Filenames
`nodes/controlnet.py`

### 2. Full Text of Added/Modified Code

We mainly created a new `DiffSynthCnetBlockReplace` class and implemented the process to register it when applying Nunchaku Qwen Image.

```python
# ==============================================================================
# Added wrapper class (DiffSynthCnetBlockReplace)
# ==============================================================================
class DiffSynthCnetBlockReplace:
    """
    Wrapper for DiffSynthCnetPatch to work with Nunchaku's `patches_replace` mechanism.
    Nunchaku drops `patches["double_block"]` completely and handles `patches_replace["dit"]`.
    """
    def __init__(self, cnet_patch, block_index):
        self.cnet_patch = cnet_patch
        self.block_index = block_index

    def __call__(self, args, extra_options):
        # Run the original transformer block first
        out = extra_options["original_block"](args)

        # Apply DiffSynth ControlNet residual
        img = out["img"]
        
        # We need `x` to determine target resolution for dynamic resizing, but Nunchaku doesn't pass it to patches_replace.
        # Find `x` in the call stack (BaseModel.apply_model or ComfyQwenImageWrapper.forward):
        import sys
        x = None
        frame = sys._getframe()
        while frame:
            if 'x' in frame.f_locals:
                candidate_x = frame.f_locals['x']
                if isinstance(candidate_x, torch.Tensor) and candidate_x.ndim in (4, 5):
                    x = candidate_x
                    break
            frame = frame.f_back
            
        if x is not None:
            spacial_compression = self.cnet_patch.vae.spacial_compression_encode()
            target_h = x.shape[-2] * spacial_compression
            target_w = x.shape[-1] * spacial_compression
            
            if self.cnet_patch.encoded_image is None or self.cnet_patch.encoded_image_size != (target_h, target_w):
                logger.info(f"[ControlNet Block {self.block_index}] Resizing condition image to {target_w}x{target_h}")
                image_scaled = comfy.utils.common_upscale(self.cnet_patch.image.movedim(-1, 1), target_w, target_h, "area", "center")
                loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
                self.cnet_patch.encoded_image = self.cnet_patch.model_patch.model.process_input_latent_image(self.cnet_patch.encode_latent_cond(image_scaled.movedim(1, -1)))
                self.cnet_patch.encoded_image_size = (target_h, target_w)
                comfy.model_management.load_models_gpu(loaded_models)
        
        encoded_image = self.cnet_patch.encoded_image
        if encoded_image is not None:
            control_residual = self.cnet_patch.model_patch.model.control_block(
                img[:, :encoded_image.shape[1]],
                encoded_image.to(img.dtype),
                self.block_index
            )
            img[:, :encoded_image.shape[1]] += control_residual * self.cnet_patch.strength
            out["img"] = img

        return out

# ==============================================================================
# Modified registration logic (inside diffsynth_controlnet_nunchaku)
# ==============================================================================
        if is_zimage_control:
            # Z Image: Use ZImageControlPatch which uses set_model_noise_refiner_patch & double_block_patch
            logger.info("[ControlNet] Using ZImageControlPatch")
            patch = ZImageControlPatch(model_patch, vae, image, strength, mask)
            model_patched.set_model_noise_refiner_patch(patch)
            model_patched.set_model_double_block_patch(patch)
        elif is_nunchaku_qwenimage:
            # Nunchaku Qwen Image: Must use patches_replace instead of double_block
            logger.info("[ControlNet] Using DiffSynthCnetBlockReplace for Nunchaku Qwen Image")
            
            # Get number of transformer blocks from the diffusion model
            dm = model.model.diffusion_model
            try:
                num_blocks = len(dm.model.transformer_blocks)
            except AttributeError:
                num_blocks = 60  # Default for Qwen Image
                
            logger.info(f"[ControlNet] Registered {num_blocks} patches_replace entries for Nunchaku Qwen Image")
            
            cnet_patch = DiffSynthCnetPatch(model_patch, vae, image, strength, mask)
            # Apply to standard double_block just in case other nodes depend on it
            model_patched.set_model_double_block_patch(cnet_patch)
            
            # Register patches_replace entries for actual ControlNet execution in Nunchaku
            for i in range(num_blocks):
                model_patched.set_model_patch_replace(
                    DiffSynthCnetBlockReplace(cnet_patch, i),
                    "dit", "double_block", i
                )
        else:
            # Standard Qwen Image
            logger.info("[ControlNet] Using DiffSynthCnetPatch (standard)")
            model_patched.set_model_double_block_patch(DiffSynthCnetPatch(model_patch, vae, image, strength, mask))
```

### 3. Its Meaning (Technical Points)

1. **Using `set_model_patch_replace`:**
   Since Nunchaku Qwen Image ignores the standard `double_block`, we registered `DiffSynthCnetBlockReplace` for all 60 blocks using `set_model_patch_replace` so that we can intervene in `blocks_replace[("double_block", i)]` of Nunchaku's custom loop.

2. **Forced Acquisition of `x` via Call Stack Analysis:**
   To solve the problem of `x` not being passed from Nunchaku, we utilized Python's powerful built-in feature `sys._getframe()`.
   This is a feature that allows you to **"look into the internal variables of the calling function (parent function) from the currently executing code"**.
   ```python
   frame = sys._getframe()
   while frame:
       if 'x' in frame.f_locals:
           # Once x is found, retrieve it and break the loop
   ```
   With this logic, we successfully forced the acquisition of `x` existing inside functions like `apply_model` of the ComfyUI core. It is an approach to pull out information from the outside without modifying any code of Nunchaku or the ComfyUI core.

3. **Restoration of Dynamic Resizing:**
   We implemented a process to calculate the original generation resolution (`target_w`, `target_h`) from the shape of the acquired `x` (`x.shape`), and scale the reference image (ControlNet input) to match it.
   As a result, the resolutions of the generated image (`img`) and the reference image (`encoded_image`) perfectly match, the ControlNet residual (`control_residual`) is added to the correct position, and the ControlNet effect is exerted exactly as intended.
