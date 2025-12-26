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

        # Debug logging - log all available information
        logger.info("=" * 80)
        logger.info("[NunchakuQwenImageDiffsynthControlnet] Starting ControlNet patch detection")
        logger.info(f"[NunchakuQwenImageDiffsynthControlnet] model type: {type(model).__name__}")
        logger.info(f"[NunchakuQwenImageDiffsynthControlnet] model_patch type: {type(model_patch).__name__}")
        logger.info(f"[NunchakuQwenImageDiffsynthControlnet] model_patch.model type: {type(model_patch.model).__name__}")
        logger.info(f"[NunchakuQwenImageDiffsynthControlnet] model_patch.model.__class__: {model_patch.model.__class__}")
        
        # Check model.model structure
        if hasattr(model, 'model'):
            logger.info(f"[NunchakuQwenImageDiffsynthControlnet] model.model exists, type: {type(model.model).__name__}")
            logger.info(f"[NunchakuQwenImageDiffsynthControlnet] model.model.__class__.__name__: {model.model.__class__.__name__}")
            logger.info(f"[NunchakuQwenImageDiffsynthControlnet] model.model.__class__.__module__: {model.model.__class__.__module__}")
            
            # Check diffusion_model if available
            if hasattr(model.model, 'diffusion_model'):
                logger.info(f"[NunchakuQwenImageDiffsynthControlnet] model.model.diffusion_model type: {type(model.model.diffusion_model).__name__}")
                logger.info(f"[NunchakuQwenImageDiffsynthControlnet] model.model.diffusion_model.__class__.__name__: {model.model.diffusion_model.__class__.__name__}")
        else:
            logger.warning("[NunchakuQwenImageDiffsynthControlnet] model has no 'model' attribute")
        
        # Check if this is a Z-Image ControlNet model
        # model_patch.model is ZImage_Control for both standard Z-Image and Nunchaku Z-Image-Turbo
        is_zimage_control = isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control)
        logger.info(f"[NunchakuQwenImageDiffsynthControlnet] is_zimage_control (ZImage_Control check): {is_zimage_control}")
        
        # Check if model.model is NunchakuZImage (for Nunchaku Z-Image-Turbo)
        is_nunchaku_zimage = False
        if hasattr(model, 'model'):
            model_model_class_name = model.model.__class__.__name__
            logger.info(f"[NunchakuQwenImageDiffsynthControlnet] Checking if model.model is NunchakuZImage: {model_model_class_name}")
            if model_model_class_name == "NunchakuZImage":
                is_nunchaku_zimage = True
                logger.info("[NunchakuQwenImageDiffsynthControlnet] ✓ Detected NunchakuZImage model base")
        
        # Use ZImageControlPatch for Z-Image ControlNet (works for both standard and Nunchaku Z-Image-Turbo)
        if is_zimage_control:
            logger.info("[NunchakuQwenImageDiffsynthControlnet] ✓ Using ZImageControlPatch for Z-Image ControlNet")
            patch = ZImageControlPatch(model_patch, vae, image, strength, mask=mask)
            model_patched.set_model_noise_refiner_patch(patch)
            model_patched.set_model_double_block_patch(patch)
            logger.info("[NunchakuQwenImageDiffsynthControlnet] ✓ Patches registered successfully")
        else:
            logger.info("[NunchakuQwenImageDiffsynthControlnet] Using DiffSynthCnetPatch for standard diffsynth ControlNet")
            model_patched.set_model_double_block_patch(DiffSynthCnetPatch(model_patch, vae, image, strength, mask))
        
        logger.info("=" * 80)
        return (model_patched,)

