import torch
import comfy.utils
import comfy.model_management
import comfy.latent_formats
import comfy.ldm.lumina.controlnet
import logging

logger = logging.getLogger(__name__)


class DiffSynthCnetPatch:
    def __init__(self, model_patch, vae, image, strength, mask=None):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength = strength
        self.mask = mask
        self.encoded_image = model_patch.model.process_input_latent_image(self.encode_latent_cond(image))
        self.encoded_image_size = (image.shape[1], image.shape[2])

    def encode_latent_cond(self, image):
        latent_image = self.vae.encode(image)
        if self.model_patch.model.additional_in_dim > 0:
            if self.mask is None:
                mask_ = torch.ones_like(latent_image)[:, :self.model_patch.model.additional_in_dim // 4]
            else:
                mask_ = comfy.utils.common_upscale(self.mask.mean(dim=1, keepdim=True), latent_image.shape[-1], latent_image.shape[-2], "bilinear", "none")

            return torch.cat([latent_image, mask_], dim=1)
        else:
            return latent_image

    def __call__(self, kwargs):
        x = kwargs.get("x")
        img = kwargs.get("img")
        block_index = kwargs.get("block_index")
        spacial_compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (x.shape[-2] * spacial_compression, x.shape[-1] * spacial_compression):
            image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center")
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.model_patch.model.process_input_latent_image(self.encode_latent_cond(image_scaled.movedim(1, -1)))
            self.encoded_image_size = (image_scaled.shape[-2], image_scaled.shape[-1])
            comfy.model_management.load_models_gpu(loaded_models)

        img[:, :self.encoded_image.shape[1]] += (self.model_patch.model.control_block(img[:, :self.encoded_image.shape[1]], self.encoded_image.to(img.dtype), block_index) * self.strength)
        kwargs['img'] = img
        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
        return self

    def models(self):
        return [self.model_patch]

class ZImageControlPatch:
    def __init__(self, model_patch, vae, image, strength, inpaint_image=None, mask=None):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.inpaint_image = inpaint_image
        self.mask = mask
        self.strength = strength
        self.encoded_image = self.encode_latent_cond(image)
        self.encoded_image_size = (image.shape[1], image.shape[2])
        self.temp_data = None

    def encode_latent_cond(self, control_image, inpaint_image=None):
        latent_image = comfy.latent_formats.Flux().process_in(self.vae.encode(control_image))
        if self.model_patch.model.additional_in_dim > 0:
            if self.mask is None:
                mask_ = torch.zeros_like(latent_image)[:, :1]
            else:
                mask_ = comfy.utils.common_upscale(self.mask.mean(dim=1, keepdim=True), latent_image.shape[-1], latent_image.shape[-2], "bilinear", "none")
            if inpaint_image is None:
                inpaint_image = torch.ones_like(control_image) * 0.5

            inpaint_image_latent = comfy.latent_formats.Flux().process_in(self.vae.encode(inpaint_image))

            return torch.cat([latent_image, mask_, inpaint_image_latent], dim=1)
        else:
            return latent_image

    def __call__(self, kwargs):
        x = kwargs.get("x")
        img = kwargs.get("img")
        img_input = kwargs.get("img_input")
        txt = kwargs.get("txt")
        pe = kwargs.get("pe")
        vec = kwargs.get("vec")
        block_index = kwargs.get("block_index")
        block_type = kwargs.get("block_type", "")
        spacial_compression = self.vae.spacial_compression_encode()
        if self.encoded_image is None or self.encoded_image_size != (x.shape[-2] * spacial_compression, x.shape[-1] * spacial_compression):
            image_scaled = comfy.utils.common_upscale(self.image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center")
            inpaint_scaled = None
            if self.inpaint_image is not None:
                inpaint_scaled = comfy.utils.common_upscale(self.inpaint_image.movedim(-1, 1), x.shape[-1] * spacial_compression, x.shape[-2] * spacial_compression, "area", "center").movedim(1, -1)
            loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
            self.encoded_image = self.encode_latent_cond(image_scaled.movedim(1, -1), inpaint_scaled)
            self.encoded_image_size = (image_scaled.shape[-2], image_scaled.shape[-1])
            comfy.model_management.load_models_gpu(loaded_models)

        cnet_blocks = self.model_patch.model.n_control_layers
        div = round(30 / cnet_blocks)

        cnet_index = (block_index // div)
        cnet_index_float = (block_index / div)

        kwargs.pop("img")  # we do ops in place
        kwargs.pop("txt")

        if cnet_index_float > (cnet_blocks - 1):
            self.temp_data = None
            return kwargs

        if self.temp_data is None or self.temp_data[0] > cnet_index:
            if block_type == "noise_refiner":
                self.temp_data = (-3, (None, self.model_patch.model(txt, self.encoded_image.to(img.dtype), pe, vec)))
            else:
                self.temp_data = (-1, (None, self.model_patch.model(txt, self.encoded_image.to(img.dtype), pe, vec)))

        if block_type == "noise_refiner":
            next_layer = self.temp_data[0] + 1
            self.temp_data = (next_layer, self.model_patch.model.forward_noise_refiner_block(block_index, self.temp_data[1][1], img_input[:, :self.temp_data[1][1].shape[1]], None, pe, vec))
            if self.temp_data[1][0] is not None:
                img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0] * self.strength)
        else:
            while self.temp_data[0] < cnet_index and (self.temp_data[0] + 1) < cnet_blocks:
                next_layer = self.temp_data[0] + 1
                self.temp_data = (next_layer, self.model_patch.model.forward_control_block(next_layer, self.temp_data[1][1], img_input[:, :self.temp_data[1][1].shape[1]], None, pe, vec))

            if cnet_index_float == self.temp_data[0]:
                img[:, :self.temp_data[1][0].shape[1]] += (self.temp_data[1][0] * self.strength)
                if cnet_blocks == self.temp_data[0] + 1:
                    self.temp_data = None

        return kwargs

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.encoded_image = self.encoded_image.to(device_or_dtype)
            self.temp_data = None
        return self

    def models(self):
        return [self.model_patch]


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
        is_krea2_model = False
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
                # Krea2 backbone in ComfyUI (comfy.ldm.krea2.model.SingleStreamDiT)
                elif dm_name == 'SingleStreamDiT':
                    is_krea2_model = True
                else:
                    dm_module = getattr(model.model.diffusion_model.__class__, "__module__", "")
                    if "comfy.ldm.krea2" in dm_module:
                        is_krea2_model = True

        logger.info(
            f"[ControlNet] is_zimage={is_zimage_control}, "
            f"is_nunchaku_qwenimage={is_nunchaku_qwenimage}, "
            f"is_krea2={is_krea2_model}"
        )

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
        elif is_krea2_model:
            # Krea2 model path: keep control processing in this patcher (not in LoRA loader).
            logger.info("[ControlNet] Using DiffSynthCnetPatch (krea2)")
            model_patched.set_model_double_block_patch(DiffSynthCnetPatch(model_patch, vae, image, strength, mask))
        else:
            # Standard Qwen Image: use patches["double_block"] (processed by ComfyUI's _forward)
            logger.info("[ControlNet] Using DiffSynthCnetPatch (standard)")
            model_patched.set_model_double_block_patch(DiffSynthCnetPatch(model_patch, vae, image, strength, mask))

        return (model_patched,)
