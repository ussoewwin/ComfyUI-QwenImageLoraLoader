# Nunchaku Qwen Image 中 ControlNet 问题的完整解释

## 1. 问题的完整解释

“ControlNet (DiffSynth 版本) 在 Nunchaku Qwen Image 中不起作用”的问题是一个**由于架构差异和规范缺失导致的复杂问题**。下面是逐步解释。

### 1. Nunchaku 模型优化方法的差异
通常，ComfyUI 的 Transformer (DiT) 模型具有标准的循环结构，它按顺序处理内部块（层）。
当 Nunchaku 优化 Z Image 时，它采取的方法是仅将块内的组件（例如 Attention）替换为 Nunchaku 版本，同时保持这个标准的循环。
然而，**对于 Qwen Image 的 Nunchaku 补丁，由于处理速度和结构的原因，它完全放弃了 ComfyUI 的标准循环结构，并重新定义和执行了其自己的 `_forward` 循环**。

### 2. 忽略标准补丁 (`double_block`)
在标准 ComfyUI 中，ControlNet 使用名为 `patches["double_block"]` 的机制干预每个块的处理。
但是，如前所述，Nunchaku 针对 Qwen Image 的自定义循环没有处理此 `patches["double_block"]` 的代码，并**完全忽略了它**。这是“ControlNet 在 Nunchaku Qwen Image 中不起作用”的第一个原因。

### 3. 备用补丁路径中的致命信息丢失（`x` 消失）
因此，我们修改了代码，使用另一种名为 `patches_replace["dit"]` 的补丁功能注入 ControlNet，Nunchaku 的自定义循环也会读取该功能。结果，ControlNet 处理本身开始执行，但发生了**另一个致命问题**。

ControlNet 需要根据正在生成的图像的分辨率动态调整参考图像（如 Canny 等条件图像）的大小（动态调整大小）。对于此调整大小，包含基本生成分辨率信息的张量 `x`（潜在变量）是绝对必要的。
然而，Nunchaku 的自定义循环被设计为在调用 `patches_replace` 时**不将此 `x` 作为参数传递**。

### 4. 分辨率不匹配导致静默失败
由于没有传递 `x`，ControlNet 无法确定“当前的生成分辨率”，从而跳过了调整大小的过程。
结果，在实际生成的图像大小（例如 1216x1216）和参考图像的默认大小（例如 768x768）**不匹配**的情况下进行了张量加法处理。
就 PyTorch 张量操作而言，它不会导致错误，而是静默通过处理。但是，因为添加了位置和大小完全不匹配的数据，ControlNet 的效果没有反映在最终生成的图像中（从视觉上看就像没有起作用一样）。

---

## 2. 解决方案的详细信息

为了解决这个问题，我们在 ComfyUI-QwenImageLoraLoader 的节点侧进行了修改。

### 1. 添加/修改的文件名
`nodes/controlnet.py`

### 2. 添加/修改的代码全文

下面是新添加的包装器类和修改后的注册类的**完整代码**。

```python
class DiffSynthCnetBlockReplace:
    """patches_replace wrapper for DiffSynthCnetPatch on Nunchaku Qwen Image.

    Nunchaku's _forward() processes patches_replace["dit"][("double_block", i)]
    but ignores patches["double_block"]. This class bridges DiffSynthCnetPatch
    into the patches_replace interface so ControlNet works with Nunchaku Qwen Image.
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

        # Detect ControlNet model type
        is_zimage_control = isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control)

        # Detect Nunchaku Qwen Image model
        # Nunchaku's _forward() processes patches_replace but ignores patches["double_block"],
        # so we need to use set_model_patch_replace() instead of set_model_double_block_patch().
        is_nunchaku_qwenimage = False
        if not is_zimage_control and hasattr(model, 'model'):
            # Check model base class name (NunchakuQwenImage from ComfyUI-nunchaku model_base)
            model_base_name = model.model.__class__.__name__
            if model_base_name == 'NunchakuQwenImage':
                is_nunchaku_qwenimage = True
            # Check diffusion model class name
            elif hasattr(model.model, 'diffusion_model'):
                dm_name = model.model.diffusion_model.__class__.__name__
                if dm_name in ('ComfyQwenImageWrapper', 'NunchakuQwenImageTransformer2DModel'):
                    is_nunchaku_qwenimage = True

        logger.info(f"[ControlNet] is_zimage={is_zimage_control}, is_nunchaku_qwenimage={is_nunchaku_qwenimage}")

        if is_zimage_control:
            # ZImage ControlNet (works for both standard and Nunchaku Z-Image)
            logger.info("[ControlNet] Using ZImageControlPatch")
            patch = ZImageControlPatch(model_patch, vae, image, strength, mask=mask)
            model_patched.set_model_noise_refiner_patch(patch)
            model_patched.set_model_double_block_patch(patch)
        elif is_nunchaku_qwenimage:
            # Nunchaku Qwen Image: use patches_replace since _forward ignores patches["double_block"]
            logger.info("[ControlNet] Using DiffSynthCnetBlockReplace for Nunchaku Qwen Image")
            cnet_patch = DiffSynthCnetPatch(model_patch, vae, image, strength, mask)
            # Register via set_model_double_block_patch for model loading (models() discovery)
            model_patched.set_model_double_block_patch(cnet_patch)
            # Get number of transformer blocks from the diffusion model
            dm = model.model.diffusion_model
            try:
                num_blocks = len(dm.transformer_blocks)
            except AttributeError:
                num_blocks = 60  # Default for Qwen Image
                logger.warning(f"[ControlNet] Could not determine block count, using default {num_blocks}")
            # Register patches_replace entries for actual ControlNet execution in Nunchaku
            for i in range(num_blocks):
                model_patched.set_model_patch_replace(
                    DiffSynthCnetBlockReplace(cnet_patch, i),
                    "dit", "double_block", i
                )
            logger.info(f"[ControlNet] Registered {num_blocks} patches_replace entries for Nunchaku Qwen Image")
        else:
            # Standard Qwen Image: use patches["double_block"] (processed by ComfyUI's _forward)
            logger.info("[ControlNet] Using DiffSynthCnetPatch (standard)")
            model_patched.set_model_double_block_patch(DiffSynthCnetPatch(model_patch, vae, image, strength, mask))

        return (model_patched,)
```

### 3. 它的含义（技术要点）

1. **使用 `set_model_patch_replace`：**
   因为 Nunchaku Qwen Image 忽略了标准的 `double_block`，所以我们使用 `set_model_patch_replace` 为全部 60 个块注册了 `DiffSynthCnetBlockReplace`，以便我们能够干预 Nunchaku 自定义循环的 `blocks_replace[("double_block", i)]`。
   在 `NunchakuQwenImageDiffsynthControlnet` 中，我们添加了稳健的逻辑来正确检测标准 Qwen 模型和 Nunchaku Qwen 包装器。

2. **通过调用堆栈分析强制获取 `x`：**
   为了解决 Nunchaku 没有传递 `x` 的问题，我们利用了 Python 强大的内置功能 `sys._getframe()`。
   这个功能允许您**“从当前执行的代码中窥视调用函数（父函数）的内部变量”**。
   ```python
   frame = sys._getframe()
   while frame:
       if 'x' in frame.f_locals:
           # 找到 x 后，提取它并跳出循环
   ```
   通过这种逻辑，我们成功地强制获取了存在于 ComfyUI 核心的 `apply_model` 等函数内部的 `x`。这是一种从外部提取信息的方法，无需修改任何 Nunchaku 或 ComfyUI 核心代码。

3. **恢复动态调整大小：**
   我们实现了一个过程，通过获取到的 `x` 的形状（`x.shape`）计算出原始生成分辨率（`target_w`，`target_h`），并将参考图像（ControlNet 输入）缩放以匹配它。
   结果，生成的图像（`img`）和参考图像（`encoded_image`）的分辨率完美匹配，ControlNet 残差（`control_residual`）被添加到了正确的位置，ControlNet 的效果也如预期般完美呈现。
