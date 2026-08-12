import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.utils
import comfy.model_management
import comfy.latent_formats
import comfy.ldm.common_dit
import comfy.ldm.lumina.controlnet
import comfy.patcher_extension
import comfy.conds
from comfy.weight_adapter.lora import LoRAAdapter
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked
from einops import rearrange
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
        self.control_strength = 1.0
        self.ab_logged = False

    def forward(self, image_tokens):
        logger.info(
            "[Krea2Control] first_forward attached=%s image_tokens_shape=%s",
            self.control_tokens is not None,
            tuple(image_tokens.shape),
        )
        if image_tokens.shape[-1] != self.base_in_features:
            raise RuntimeError(
                f"Krea2 first projection expects {self.base_in_features} image features, got {image_tokens.shape[-1]}."
            )
        if self.control_tokens is None:
            return self.base_first(image_tokens)
        control_tokens = comfy.utils.repeat_to_batch_size(self.control_tokens, image_tokens.shape[0])
        control_tokens = control_tokens.to(device=image_tokens.device, dtype=image_tokens.dtype)
        control_tokens = control_tokens * float(self.control_strength)
        if control_tokens.shape[1] != image_tokens.shape[1]:
            raise RuntimeError(
                f"Krea2 control token count mismatch: image={image_tokens.shape[1]}, control={control_tokens.shape[1]}."
            )
        x = torch.cat((image_tokens, control_tokens), dim=-1)
        weight = comfy.model_management.cast_to_device(self.weight, x.device, x.dtype)
        bias = None
        if self.bias is not None:
            bias = comfy.model_management.cast_to_device(self.bias, x.device, x.dtype)
        out = F.linear(x, weight, bias)

        # One-shot A/B diff log per run: control off vs on.
        if not self.ab_logged:
            self.ab_logged = True
            with torch.no_grad():
                base_out = self.base_first(image_tokens)
                delta = (out - base_out).detach().abs().mean().float().cpu().item()
                logger.info(
                    "[Krea2Control] first_output_delta_mean_abs=%.6f control_strength=%.4f",
                    float(delta),
                    float(self.control_strength),
                )
        return out


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
    if base.startswith("txtfusion."):
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


def _krea2_spatial_patch_size(patch_size):
    if isinstance(patch_size, int):
        return int(patch_size), int(patch_size)
    if isinstance(patch_size, (list, tuple)):
        if len(patch_size) == 0:
            raise RuntimeError("Krea2 patch_size is empty.")
        if len(patch_size) == 1:
            v = int(patch_size[0])
            return v, v
        return int(patch_size[-2]), int(patch_size[-1])
    raise RuntimeError(f"Unsupported Krea2 patch_size type: {type(patch_size)}")


def _krea2_control_latent_to_4d(control_latent):
    if control_latent.ndim == 4:
        return control_latent
    if control_latent.ndim == 5:
        b, c, t, h, w = control_latent.shape
        return control_latent.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    raise RuntimeError(f"Krea2 control latent must be 4D or 5D, got {tuple(control_latent.shape)}")


def _krea2_latent_to_tokens(control_latent, x, patch_size, expected_control_features):
    if x.ndim == 5:
        batch = x.shape[0] * x.shape[2]
    elif x.ndim == 4:
        batch = x.shape[0]
    else:
        raise RuntimeError(f"Krea2 input latent must be 4D or 5D, got {tuple(x.shape)}")

    patch_h, patch_w = _krea2_spatial_patch_size(patch_size)

    control_source = _krea2_control_latent_to_4d(control_latent)
    control = comfy.utils.repeat_to_batch_size(control_source, batch)
    control = comfy.model_management.cast_to_device(control, x.device, x.dtype)

    target_h, target_w = x.shape[-2], x.shape[-1]
    if control.shape[-2:] != (target_h, target_w):
        control = comfy.utils.common_upscale(control, target_w, target_h, "bilinear", "disabled")

    control = comfy.ldm.common_dit.pad_to_patch_size(control, (patch_h, patch_w))
    b, c, h, w = control.shape
    token_features = c * patch_h * patch_w
    if token_features != expected_control_features:
        raise RuntimeError(
            f"Krea2 control token feature mismatch: got {token_features}, expected {expected_control_features}."
        )
    control = control.reshape(b, c, h // patch_h, patch_h, w // patch_w, patch_w)
    return control.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // patch_h) * (w // patch_w), token_features)


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
            control_tokens_mean_abs = float(control_tokens.detach().abs().mean().cpu().item())
            logger.info(
                "[Krea2Control] control_tokens_shape=%s mean_abs=%.6f",
                tuple(control_tokens.shape),
                control_tokens_mean_abs,
            )
            projection.control_tokens = control_tokens
            projection.ab_logged = False
            if getattr(diffusion_model, "first", None) is not projection:
                diffusion_model.first = projection
            logger.info(
                "[Krea2Control] first_injected=%s",
                getattr(diffusion_model, "first", None) is projection,
            )
            return executor(*args, **kwargs)
        finally:
            projection.control_tokens = previous_tokens
            if getattr(diffusion_model, "first", None) is projection:
                diffusion_model.first = projection.base_first if projection.base_first is not None else previous_first

    return wrapper


def _classify_krea2_control_type(state_dict):
    """Classify Krea2 control MODEL_PATCH sub-type.

    Returns:
        "depth"    -- has expanded first projection (requires control latent injection)
        "openpose" -- pure block LoRA on DiT + txtfusion (no first projection)
    """
    first_candidates = (
        "first.weight",
        "diffusion_model.first.weight",
        "model.diffusion_model.first.weight",
        "transformer.first.weight",
    )
    for key in first_candidates:
        if key in state_dict:
            return "depth"
    return "openpose"


_KREA2_REF_MAX_PIXELS = 1024 * 1024
_KREA2_REF_SNAP = 16


def _krea2_fit_ref_image(image):
    """Downscale (never upscale) an (B, H, W, C) control image to fit ~1MP,
    snapped to /16 — the ai-toolkit krea2 reference-latent convention."""
    samples = image.movedim(-1, 1)  # (B, C, H, W)
    h, w = samples.shape[2], samples.shape[3]
    scale = min(1.0, math.sqrt(_KREA2_REF_MAX_PIXELS / (w * h)))
    nw = max(int(round(w * scale / _KREA2_REF_SNAP)) * _KREA2_REF_SNAP, _KREA2_REF_SNAP)
    nh = max(int(round(h * scale / _KREA2_REF_SNAP)) * _KREA2_REF_SNAP, _KREA2_REF_SNAP)
    if (nh, nw) == (h, w):
        return image
    return comfy.utils.common_upscale(samples, nw, nh, "area", "disabled").movedim(1, -1)


def _krea2_make_ref_cond_patch(base_model, ref_latent):
    """Build extra_conds / extra_conds_shapes patches that inject a single
    control reference latent with the index_timestep_zero method.

    The Krea2 core model consumes reference_latents natively (ref tokens are
    appended to the image sequence and conditioned at t=0). Injection is
    skipped when the conditioning already carries its own reference_latents
    (e.g. TextEncodeKrea2OstrisEdit), so the two paths never double-inject.
    """
    orig_extra_conds = base_model.extra_conds
    orig_extra_conds_shapes = base_model.extra_conds_shapes

    def extra_conds(**kwargs):
        out = orig_extra_conds(**kwargs)
        if kwargs.get("reference_latents", None) is None and "ref_latents" not in out:
            out["ref_latents"] = comfy.conds.CONDList([base_model.process_latent_in(ref_latent)])
            out["ref_latents_method"] = comfy.conds.CONDConstant("index_timestep_zero")
        return out

    def extra_conds_shapes(**kwargs):
        out = orig_extra_conds_shapes(**kwargs)
        if kwargs.get("reference_latents", None) is None and "ref_latents" not in out:
            out["ref_latents"] = [1, 16, math.prod(ref_latent.size()) // 16]
        return out

    return extra_conds, extra_conds_shapes


def _krea2_patchify(x, patch):
    """(B, C, H, W) -> (B, H/patch * W/patch, C * patch * patch) image tokens."""
    b, c, h, w = x.shape
    return (
        x.reshape(b, c, h // patch, patch, w // patch, patch)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(b, (h // patch) * (w // patch), c * patch * patch)
    )


def _krea2_ref_pack(dit, ref_latents, bs, device, dtype):
    """Pack reference latents into tokens + RoPE positions (axis-0 index 1, 2, ...).

    Mirrors the ai-toolkit / ostris edit convention: each reference gets its
    own y/x grid with axis-0 index i+1, snapped to the DiT patch size.
    Returns (reftok (B, Lr, C*p*p), refpos (B, Lr, 3)).
    """
    patch = dit.patch
    ref_tokens = []
    ref_pos = []
    for i, ref in enumerate(ref_latents, 1):
        if ref.ndim == 5:  # (B, C, T, H, W) Wan21 layout, T == 1 for images
            rb, rc, rt, rh5, rw5 = ref.shape
            ref = ref.reshape(rb * rt, rc, rh5, rw5)
        ref = comfy.ldm.common_dit.pad_to_patch_size(ref.to(device, dtype), (patch, patch))
        ref = comfy.utils.repeat_to_batch_size(ref, bs)
        rh, rw = ref.shape[-2] // patch, ref.shape[-1] // patch
        ref_tokens.append(
            ref.reshape(bs, ref.shape[1], rh, patch, rw, patch)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(bs, rh * rw, ref.shape[1] * patch * patch)
        )
        rid = torch.zeros(rh, rw, 3, device=device, dtype=torch.float32)
        rid[..., 0] = float(i)
        rid[..., 1] = torch.arange(rh, device=device, dtype=torch.float32)[:, None]
        rid[..., 2] = torch.arange(rw, device=device, dtype=torch.float32)[None, :]
        ref_pos.append(rid.reshape(1, rh * rw, 3).repeat(bs, 1, 1))
    return torch.cat(ref_tokens, dim=1), torch.cat(ref_pos, dim=1)


def _krea2_ref_attn_kv(attn, x, freqs, kv_capture=None, kv_cache=None, transformer_options={}):
    """Krea2 Attention forward with optional K/V capture or cached-K/V injection
    (post-RoPE, pre-GQA-expansion)."""
    q, k, v, gate = attn.wq(x), attn.wk(x), attn.wv(x), attn.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=attn.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=attn.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=attn.kvheads)
    q, k = attn.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if kv_capture is not None:
        kv_capture.append((k, v))
    if kv_cache is not None:
        k = torch.cat((k, kv_cache[0].to(k.dtype)), dim=2)
        v = torch.cat((v, kv_cache[1].to(v.dtype)), dim=2)
    if attn.kvheads != attn.heads:
        rep = attn.heads // attn.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = optimized_attention_masked(
        q, k, v, attn.heads, mask=None, skip_reshape=True,
        transformer_options=transformer_options,
    )
    return attn.wo(out * F.sigmoid(gate))


def _krea2_ref_block_forward(block, x, vec, freqs, kv_capture=None, kv_cache=None,
                             transformer_options={}):
    """Krea2 SingleStreamBlock forward with K/V capture / injection."""
    prescale, preshift, pregate, postscale, postshift, postgate = block.mod(vec)
    x = x + pregate * _krea2_ref_attn_kv(
        block.attn,
        (1 + prescale) * block.prenorm(x) + preshift,
        freqs,
        kv_capture=kv_capture,
        kv_cache=kv_cache,
        transformer_options=transformer_options,
    )
    x = x + postgate * block.mlp((1 + postscale) * block.postnorm(x) + postshift)
    return x


def _krea2_ref_precompute_kv(dit, x, timesteps, ref_latents, transformer_options):
    """Run only the clean reference tokens through the blocks at t=0 and record
    each block's post-RoPE K/V (isolated ref attention, kv_cache training mode).

    The ref tokens never see text / noisy tokens, so one pass serves the whole
    denoise — matching ai-toolkit kv_cache-trained Krea2 control LoRAs.
    """
    bs = x.shape[0] * (x.shape[2] if x.ndim == 5 else 1)
    reftok, refpos = _krea2_ref_pack(dit, ref_latents, bs, x.device, x.dtype)
    h = dit.first(reftok)
    t0 = dit.tmlp(
        timestep_embedding(torch.zeros_like(timesteps), dit.tdim)
        .unsqueeze(1)
        .to(h.dtype)
    )
    tvec0 = dit.tproj(t0)
    freqs = dit.pe_embedder(refpos)

    ref_kv = []
    for block in dit.blocks:
        cap = []
        h = _krea2_ref_block_forward(
            block, h, tvec0, freqs, kv_capture=cap,
            transformer_options=transformer_options,
        )
        ref_kv.append(cap[0])
    return ref_kv


def _krea2_forward_with_cached_refs(dit, x, timesteps, context, ref_kv, transformer_options):
    """Krea2 denoising forward with reference tokens replaced by cached K/V:
    the live sequence is just text + noisy image tokens, and every block's
    attention appends the cached ref K/V as extra keys."""
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
    bs, c, H_orig, W_orig = x.shape
    patch = dit.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch
    device = x.device

    context = dit._unpack_context(context)

    img = dit.first(_krea2_patchify(x, patch))

    t = dit.tmlp(timestep_embedding(timesteps, dit.tdim).unsqueeze(1).to(img.dtype))
    tvec = dit.tproj(t)

    context = dit.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = dit.txtmlp(context)

    txtlen, imglen = context.shape[1], img.shape[1]
    combined = torch.cat((context, img), dim=1)

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
    pos = torch.cat((txtpos, imgpos), dim=1)

    freqs = dit.pe_embedder(pos)

    for block, kv in zip(dit.blocks, ref_kv):
        combined = _krea2_ref_block_forward(
            block, combined, tvec, freqs, kv_cache=kv,
            transformer_options=transformer_options,
        )

    final = dit.last(combined, t)
    out = final[:, txtlen:txtlen + imglen, :]
    out = out.reshape(bs, h_, w_, c, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(bs, c, H, W)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, c, H_orig, W_orig).movedim(1, 2)
    return out


def _krea2_ref_fingerprint(ref_latents, bs):
    """Cheap content key for the ref K/V cache (tensors are rebuilt per step,
    so object identity cannot be used)."""
    key = [bs]
    for r in ref_latents:
        rf = r.float()
        key.append((tuple(r.shape), float(rf.sum()), float(rf.square().sum())))
    return tuple(key)


def _krea2_make_ref_kv_forward(dit):
    """Build a diffusion_model.forward replacement that runs the reference
    latents in isolated kv_cache mode: one t=0 ref-only pass precomputes every
    block's K/V, reused on each denoising step as extra attention keys."""
    orig_forward = dit.forward
    state = {"last_sigma": None, "caches": {}}

    def forward(x, timesteps, context, attention_mask=None, transformer_options={},
                ref_latents=None, **kwargs):
        if ref_latents is None or len(ref_latents) == 0:
            return orig_forward(
                x, timesteps, context, attention_mask=attention_mask,
                transformer_options=transformer_options, **kwargs,
            )

        # New-run detection: sigmas only decrease within a run.
        sig = float(timesteps.max())
        sample_sigmas = transformer_options.get("sample_sigmas", None)
        new_run = state["last_sigma"] is None or sig > state["last_sigma"]
        if (
            sample_sigmas is not None
            and sig == float(sample_sigmas[0])
            and sig != state["last_sigma"]
        ):
            new_run = True
        if new_run:
            state["caches"].clear()
        state["last_sigma"] = sig

        bs = x.shape[0] * (x.shape[2] if x.ndim == 5 else 1)
        key = _krea2_ref_fingerprint(ref_latents, bs)
        ref_kv = state["caches"].get(key)
        if ref_kv is None:
            ref_kv = _krea2_ref_precompute_kv(
                dit, x, timesteps, ref_latents, transformer_options
            )
            state["caches"][key] = ref_kv
        return _krea2_forward_with_cached_refs(
            dit, x, timesteps, context, ref_kv, transformer_options
        )

    return forward


def _apply_krea2_openpose_control(model_patched, model_patch, vae, image, strength, use_kv_cache=True):
    """Apply Krea2 openpose-style control LoRA.

    Block LoRA patches on diffusion_model.blocks and diffusion_model.txtfusion
    (layerwise + refiner blocks), plus the control image injected as a native
    Krea2 reference latent (index_timestep_zero) through extra_conds. This is
    the Kontext-style conditioning pathway the openpose control LoRA was
    trained against. No first projection, no wrapper, no control-latent
    injection in the depth sense.
    """
    state_dict = _krea2_get_lora_state_dict(model_patch)

    lora_patches = _krea2_build_block_patches(state_dict, model_patched)
    if not lora_patches:
        raise RuntimeError(
            "No block LoRA patches matched the current Krea2 model for openpose control."
        )

    patched_keys = model_patched.add_patches(
        lora_patches, strength_patch=strength, strength_model=1.0
    )
    if not patched_keys:
        raise RuntimeError(
            "Krea2 model did not accept any openpose control LoRA block patches."
        )

    logger.info(
        "[Krea2OpenposeControl] Applied %d block LoRA patches (strength=%.4f)",
        len(patched_keys),
        float(strength),
    )

    # Inject the control image as a reference latent via the native Krea2 ref
    # pathway (ref tokens appended at t=0, index_timestep_zero method).
    if vae is None or image is None:
        logger.info(
            "[Krea2OpenposeControl] vae/image missing: LoRA applied without control reference injection"
        )
        return

    control_image = _krea2_fit_ref_image(image[:, :, :, :3].clamp(0.0, 1.0))
    ref_latent = vae.encode(control_image)
    logger.info(
        "[Krea2OpenposeControl] control ref latent shape=%s",
        tuple(ref_latent.shape),
    )

    extra_conds, extra_conds_shapes = _krea2_make_ref_cond_patch(model_patched.model, ref_latent)
    model_patched.add_object_patch("extra_conds", extra_conds)
    model_patched.add_object_patch("extra_conds_shapes", extra_conds_shapes)
    logger.info("[Krea2OpenposeControl] reference latent injection armed (index_timestep_zero)")

    # Isolated kv_cache mode: ref tokens never ride in the per-step sequence.
    # One t=0 ref-only pass precomputes every block's K/V, injected as extra
    # attention keys each step — the ai-toolkit kv_cache training convention
    # this control LoRA was trained with (see krea2_controlnet_pose.json).
    if use_kv_cache:
        dit = model_patched.get_model_object("diffusion_model")
        model_patched.add_object_patch(
            "diffusion_model.forward",
            _krea2_make_ref_kv_forward(dit),
        )
        logger.info("[Krea2OpenposeControl] kv_cache isolated-ref forward armed")


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
    projection.control_strength = float(strength)

    lora_patches = _krea2_build_block_patches(state_dict, model_patched)
    if not lora_patches:
        raise RuntimeError("No block LoRA patches matched the current Krea2 model.")
    patched_keys = model_patched.add_patches(lora_patches, strength_patch=strength, strength_model=1.0)
    if not patched_keys:
        raise RuntimeError("Krea2 model did not accept any control LoRA block patches.")

    control_latent = _krea2_prepare_control_latent(model_patched, vae, image)
    logger.info("[Krea2Control] control_latent_shape=%s", tuple(control_latent.shape))
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
                "optional": {"mask": ("MASK",),
                              "use_kv_cache": ("BOOLEAN", {"default": True, "tooltip": "Isolated reference K/V cache mode (ai-toolkit kv_cache training convention). Leave on for Krea2 control LoRAs trained with kv_cache."})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "diffsynth_controlnet_nunchaku"
    EXPERIMENTAL = True

    CATEGORY = "advanced/loaders/qwen"

    def diffsynth_controlnet_nunchaku(self, model, model_patch, vae, image, strength, mask=None, use_kv_cache=True):
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
            # Krea2 path: strictly sub-classify depth vs openpose.
            krea2_state_dict = _krea2_get_lora_state_dict(model_patch)
            krea2_type = _classify_krea2_control_type(krea2_state_dict)
            logger.info("[ControlNet] Krea2 sub-route: %s", krea2_type)

            if krea2_type == "depth":
                logger.info("[ControlNet] Applying dedicated Krea2 depth control route")
                _apply_krea2_control(model_patched, model_patch, vae, image, strength)
            elif krea2_type == "openpose":
                logger.info("[ControlNet] Applying dedicated Krea2 openpose control route")
                _apply_krea2_openpose_control(
                    model_patched, model_patch, vae, image, strength,
                    use_kv_cache=use_kv_cache,
                )
            else:
                raise RuntimeError(
                    f"Unrecognized Krea2 control type: {krea2_type}. "
                    "Expected 'depth' or 'openpose'."
                )
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
