import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.utils
import comfy.model_management
import comfy.latent_formats
import comfy.ldm.common_dit
import comfy.ldm.lumina.controlnet
import comfy.patcher_extension
from comfy.weight_adapter.lora import LoRAAdapter
import logging

logger = logging.getLogger(__name__)

KREA2_CONTROL_LATENT_KEY = "krea2_control_latent"
KREA2_CONTROL_WRAPPER_KEY = "krea2_control_inline"


def _classify_controlnet_target(model, model_patch):
    """
    Classify ControlNet target route strictly to avoid mixing model families.

    Returns one of:
      - "zimage"
      - "nunchaku_qwenimage"
      - "qwenimage_standard"
      - "krea2"
      - "unknown"
    """
    if isinstance(model_patch.model, comfy.ldm.lumina.controlnet.ZImage_Control):
        return "zimage"

    if not hasattr(model, "model"):
        return "unknown"

    model_base_name = model.model.__class__.__name__
    if model_base_name == "NunchakuQwenImage":
        return "nunchaku_qwenimage"

    if not hasattr(model.model, "diffusion_model"):
        return "unknown"

    dm = model.model.diffusion_model
    dm_name = dm.__class__.__name__
    dm_module = getattr(dm.__class__, "__module__", "")

    if dm_name in ("ComfyQwenImageWrapper", "NunchakuQwenImageTransformer2DModel"):
        return "qwenimage_standard"

    if dm_name == "SingleStreamDiT" or "comfy.ldm.krea2" in dm_module:
        return "krea2"

    return "unknown"


class _Krea2FirstProjection(nn.Module):
    """
    Runtime projection shim for Krea2 control tokens.
    """

    def __init__(self, expanded_weight, base_in_features, base_first, bias=None):
        super().__init__()
        self.base_in_features = int(base_in_features)
        self.control_in_features = int(expanded_weight.shape[1] - base_in_features)
        if self.control_in_features <= 0:
            raise RuntimeError("Invalid Krea2 control projection width.")
        self.base_first = base_first
        self.weight = nn.Parameter(expanded_weight.detach().cpu().clone(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.detach().cpu().clone(), requires_grad=False)
        self.control_tokens = None

    def forward(self, image_tokens):
        if image_tokens.shape[-1] != self.base_in_features:
            raise RuntimeError(
                f"Krea2 first projection expects {self.base_in_features} image features, got {image_tokens.shape[-1]}."
            )
        if self.control_tokens is None:
            return self.base_first(image_tokens)
        control_tokens = comfy.utils.repeat_to_batch_size(self.control_tokens, image_tokens.shape[0])
        control_tokens = control_tokens.to(device=image_tokens.device, dtype=image_tokens.dtype)
        if control_tokens.shape[1] != image_tokens.shape[1]:
            raise RuntimeError(
                f"Krea2 control token count mismatch: image={image_tokens.shape[1]}, control={control_tokens.shape[1]}."
            )
        x = torch.cat((image_tokens, control_tokens), dim=-1)
        weight = comfy.model_management.cast_to_device(self.weight, x.device, x.dtype)
        bias = None
        if self.bias is not None:
            bias = comfy.model_management.cast_to_device(self.bias, x.device, x.dtype)
        return F.linear(x, weight, bias)


def _krea2_get_lora_state_dict(model_patch):
    state_dict = getattr(getattr(model_patch, "model", None), "state_dict", None)
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise RuntimeError("Krea2 route expects MODEL_PATCH from Krea2 controlnet lora loader.")
    return state_dict


def _krea2_find_expanded_first_weight(state_dict, out_features, in_features):
    candidates = (
        "first.weight",
        "diffusion_model.first.weight",
        "model.diffusion_model.first.weight",
        "transformer.first.weight",
    )
    for key in candidates:
        w = state_dict.get(key)
        if torch.is_tensor(w) and w.ndim == 2 and tuple(w.shape) == (out_features, in_features):
            return key
    return None


def _krea2_find_bias_for_first(state_dict, weight_key, out_features):
    check = []
    if weight_key.endswith(".weight"):
        check.append(weight_key[:-7] + ".bias")
    check.extend(
        (
            "first.bias",
            "diffusion_model.first.bias",
            "model.diffusion_model.first.bias",
            "transformer.first.bias",
        )
    )
    for key in check:
        b = state_dict.get(key)
        if torch.is_tensor(b) and b.ndim == 1 and tuple(b.shape) == (out_features,):
            return b
    return None


def _krea2_lora_pairs(state_dict):
    patterns = (
        (".lora_down.weight", ".lora_up.weight"),
        (".lora_down", ".lora_up"),
        ("_lora.down.weight", "_lora.up.weight"),
        (".A", ".B"),
        (".lora_A.weight", ".lora_B.weight"),
        (".lora_A", ".lora_B"),
    )
    seen = set()
    for down_suffix, up_suffix in patterns:
        for down_key in state_dict.keys():
            if not down_key.endswith(down_suffix):
                continue
            base = down_key[: -len(down_suffix)]
            up_key = base + up_suffix
            if up_key not in state_dict:
                continue
            pair = (down_key, up_key)
            if pair in seen:
                continue
            seen.add(pair)
            yield base, down_key, up_key


def _krea2_target_key(base):
    prefixes = ("model.diffusion_model.", "diffusion_model.", "transformer.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if base.startswith(prefix):
                base = base[len(prefix) :]
                changed = True
    if base.startswith("blocks."):
        return f"diffusion_model.{base}.weight"
    return None


def _krea2_model_key_shape(model_patcher, key):
    try:
        cur = model_patcher.model
        for part in key.split("."):
            cur = getattr(cur, part)
    except Exception:
        return None
    shape = getattr(cur, "shape", None)
    if shape is not None:
        return tuple(shape)
    data = getattr(cur, "data", None)
    if data is not None:
        tensor_shape = getattr(data, "tensor_shape", None)
        if tensor_shape is not None:
            return tuple(tensor_shape)
    tensor_shape = getattr(cur, "tensor_shape", None)
    if tensor_shape is not None:
        return tuple(tensor_shape)
    return None


def _krea2_build_block_patches(state_dict, model_patcher):
    patches = {}
    model_sd = model_patcher.model.state_dict()
    for base, down_key, up_key in _krea2_lora_pairs(state_dict):
        target_key = _krea2_target_key(base)
        if target_key is None:
            continue
        down = state_dict[down_key]
        up = state_dict[up_key]
        if not (torch.is_tensor(down) and torch.is_tensor(up) and down.ndim == 2 and up.ndim == 2):
            continue

        target_shape = _krea2_model_key_shape(model_patcher, target_key)
        if target_shape is None:
            t = model_sd.get(target_key)
            if torch.is_tensor(t):
                target_shape = tuple(t.shape)
        if target_shape is None or len(target_shape) < 2:
            continue

        out_features, in_features = target_shape[0], target_shape[1]
        if not (up.shape[0] == out_features and down.shape[1] == in_features and up.shape[1] == down.shape[0]):
            if down.shape[0] == in_features and up.shape[1] == out_features and down.shape[1] == up.shape[0]:
                down = down.t().contiguous()
                up = up.t().contiguous()
            else:
                continue

        rank = down.shape[0]
        alpha = rank
        alpha_key = None
        for suffix in (".alpha", ".network_alpha", ".scale"):
            candidate = base + suffix
            if candidate in state_dict:
                alpha_key = candidate
                val = state_dict[candidate]
                alpha = float(val.detach().cpu().reshape(-1)[0]) if torch.is_tensor(val) else float(val)
                break

        used = {down_key, up_key}
        if alpha_key is not None:
            used.add(alpha_key)
        patches[target_key] = LoRAAdapter(used, (up, down, alpha, None, None, None))
    return patches


def _krea2_prepare_control_latent(model_patcher, vae, image):
    control_image = image[:, :, :, :3].clamp(0.0, 1.0)
    control_latent = vae.encode(control_image)
    if hasattr(model_patcher.model, "process_latent_in"):
        control_latent = model_patcher.model.process_latent_in(control_latent)
    return control_latent


def _krea2_latent_to_tokens(control_latent, x, patch_size, expected_control_features):
    if x.ndim == 5:
        batch = x.shape[0] * x.shape[2]
    elif x.ndim == 4:
        batch = x.shape[0]
    else:
        raise RuntimeError(f"Krea2 input latent must be 4D or 5D, got {tuple(x.shape)}")

    control = comfy.utils.repeat_to_batch_size(control_latent, batch)
    control = comfy.model_management.cast_to_device(control, x.device, x.dtype)

    target_h, target_w = x.shape[-2], x.shape[-1]
    if control.shape[-2:] != (target_h, target_w):
        control = comfy.utils.common_upscale(control, target_w, target_h, "bilinear", "disabled")

    control = comfy.ldm.common_dit.pad_to_patch_size(control, (patch_size, patch_size))
    b, c, h, w = control.shape
    token_features = c * patch_size * patch_size
    if token_features != expected_control_features:
        raise RuntimeError(
            f"Krea2 control token feature mismatch: got {token_features}, expected {expected_control_features}."
        )
    control = control.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
    return control.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // patch_size) * (w // patch_size), token_features)


def _krea2_extract_transformer_options(args, kwargs):
    transformer_options = kwargs.get("transformer_options")
    if transformer_options is None and len(args) >= 5 and isinstance(args[4], dict):
        transformer_options = args[4]
    if transformer_options is None and len(args) > 0 and isinstance(args[-1], dict):
        transformer_options = args[-1]
    return transformer_options


def _krea2_restore_projection(diffusion_model, projection):
    projection.control_tokens = None
    if getattr(diffusion_model, "first", None) is projection:
        diffusion_model.first = projection.base_first


def _krea2_make_injection(projection):
    def inject(model_patcher):
        dm = getattr(model_patcher.model, "diffusion_model", None)
        if dm is None:
            return
        current_first = getattr(dm, "first", None)
        if isinstance(current_first, _Krea2FirstProjection):
            current_first = current_first.base_first
        if current_first is not None:
            projection.base_first = current_first
            dm.first = current_first
        projection.control_tokens = None

    def eject(model_patcher):
        dm = getattr(model_patcher.model, "diffusion_model", None)
        if dm is not None:
            _krea2_restore_projection(dm, projection)

    return [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)]


def _krea2_restore_callback(model_patcher, *args):
    attachment = model_patcher.get_attachment(KREA2_CONTROL_WRAPPER_KEY)
    if not isinstance(attachment, dict):
        return
    projection = attachment.get("projection")
    if not isinstance(projection, _Krea2FirstProjection):
        return
    dm = getattr(model_patcher.model, "diffusion_model", None)
    if dm is None:
        return
    _krea2_restore_projection(dm, projection)


def _krea2_make_wrapper(projection):
    def wrapper(executor, *args, **kwargs):
        transformer_options = _krea2_extract_transformer_options(args, kwargs)
        if not isinstance(transformer_options, dict):
            raise RuntimeError("Krea2 control wrapper could not read transformer_options.")

        control_latent = transformer_options.get(KREA2_CONTROL_LATENT_KEY)
        if control_latent is None:
            raise RuntimeError("Krea2 control latent missing in transformer_options.")

        diffusion_model = executor.class_obj
        x = args[0]
        previous_first = getattr(diffusion_model, "first", None)
        previous_tokens = projection.control_tokens
        try:
            control_tokens = _krea2_latent_to_tokens(
                control_latent,
                x,
                diffusion_model.patch,
                projection.control_in_features,
            )
            projection.control_tokens = control_tokens
            if getattr(diffusion_model, "first", None) is not projection:
                diffusion_model.first = projection
            return executor(*args, **kwargs)
        finally:
            projection.control_tokens = previous_tokens
            if getattr(diffusion_model, "first", None) is projection:
                diffusion_model.first = projection.base_first if projection.base_first is not None else previous_first

    return wrapper


def _apply_krea2_control(model_patched, model_patch, vae, image, strength):
    state_dict = _krea2_get_lora_state_dict(model_patch)
    first = model_patched.get_model_object("diffusion_model.first")
    first_weight = getattr(first, "weight", None)
    if first_weight is None or len(first_weight.shape) != 2:
        raise RuntimeError("Current MODEL is not Krea2-compatible (missing 2D diffusion_model.first.weight).")
    out_features, base_in_features = int(first_weight.shape[0]), int(first_weight.shape[1])

    expanded_key = _krea2_find_expanded_first_weight(state_dict, out_features, base_in_features * 2)
    if expanded_key is None:
        raise RuntimeError(
            f"Expanded first projection weight ({out_features}, {base_in_features * 2}) not found in Krea2 control LoRA."
        )
    expanded_weight = state_dict[expanded_key]
    expanded_bias = _krea2_find_bias_for_first(state_dict, expanded_key, out_features)
    if expanded_bias is None and hasattr(first, "bias") and torch.is_tensor(first.bias):
        expanded_bias = first.bias.detach()

    projection = _Krea2FirstProjection(expanded_weight, base_in_features, first, expanded_bias)

    lora_patches = _krea2_build_block_patches(state_dict, model_patched)
    if not lora_patches:
        raise RuntimeError("No block LoRA patches matched the current Krea2 model.")
    patched_keys = model_patched.add_patches(lora_patches, strength_patch=strength, strength_model=1.0)
    if not patched_keys:
        raise RuntimeError("Krea2 model did not accept any control LoRA block patches.")

    control_latent = _krea2_prepare_control_latent(model_patched, vae, image)
    model_patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        KREA2_CONTROL_WRAPPER_KEY,
        _krea2_make_wrapper(projection),
    )
    model_patched.set_injections(KREA2_CONTROL_WRAPPER_KEY, _krea2_make_injection(projection))
    model_patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_DETACH,
        KREA2_CONTROL_WRAPPER_KEY,
        _krea2_restore_callback,
    )
    model_patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
        KREA2_CONTROL_WRAPPER_KEY,
        _krea2_restore_callback,
    )
    model_patched.set_attachments(
        KREA2_CONTROL_WRAPPER_KEY,
        {"projection": projection, "patched_model_keys": len(patched_keys)},
    )
    transformer_options = model_patched.model_options.setdefault("transformer_options", {})
    transformer_options[KREA2_CONTROL_LATENT_KEY] = control_latent


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

        route = _classify_controlnet_target(model, model_patch)
        logger.info(f"[ControlNet] route={route}")

        if route == "zimage":
            # ZImage ControlNet (works for both standard and Nunchaku Z-Image)
            logger.info("[ControlNet] Using ZImageControlPatch")
            patch = ZImageControlPatch(model_patch, vae, image, strength, mask=mask)
            model_patched.set_model_noise_refiner_patch(patch)
            model_patched.set_model_double_block_patch(patch)
        elif route == "nunchaku_qwenimage":
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
        elif route == "krea2":
            # Krea2 path: dedicated Krea2 runtime patching flow.
            logger.info("[ControlNet] Applying dedicated Krea2 control route")
            _apply_krea2_control(model_patched, model_patch, vae, image, strength)
        elif route == "qwenimage_standard":
            # Standard Qwen Image: use patches["double_block"] (processed by ComfyUI's _forward)
            logger.info("[ControlNet] Using DiffSynthCnetPatch (qwenimage_standard)")
            model_patched.set_model_double_block_patch(DiffSynthCnetPatch(model_patch, vae, image, strength, mask))
        else:
            raise RuntimeError(
                "Unsupported model/controlnet route. "
                "Control routing is strict to avoid mixing QI/ZI/Nunchaku/Krea2 branches."
            )

        return (model_patched,)
