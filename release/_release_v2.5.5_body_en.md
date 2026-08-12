<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.5.5.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## 竭 Purpose

Add support for the **openpose** variant of the Krea2 (`SingleStreamDiT`) ControlNet LoRA
(`krea2_turbo_openpose_controlnet.safetensors`).

### Background

- v2.5.1 implemented Krea2 **depth** ControlNet LoRA (`krea2-depth-control-lora.safetensors`)
  using the "expanded first projection" approach (`first.weight` expanded to 2x input channels
  + control-latent injection).
- The openpose variant is structurally different:
  - **No** `first.weight` expansion (228MB / bfloat16 / rank 32)
  - Pure LoRA over `diffusion_model.blocks.{0-27}` (28 DiT blocks) +
    `diffusion_model.txtfusion.layerwise_blocks.{0,1}` / `refiner_blocks.{0,1}` (256 pairs total)
  - The conditioning pathway used at training time is the **Kontext-style reference latent**
    (`index_timestep_zero`), i.e. the reference-image injection convention used by ostris'
    `comfyui-krea2-ostris-edit` and pysssss' `krea2_controlnet_pose.json` workflow (kv_cache=true).

### Requirements

1. **Exclusive routing**: openpose must not affect depth / zimage / nunchaku_qwenimage /
   qwenimage_standard in any way. The existing 4 routes must remain unchanged (indentation only).
2. **Self-contained**: reproduce the same pose-control effect within this repository alone,
   without depending on ostris / pysssss nodes or workflows.
3. **Loader untouched**: `Krea2ControlNetLoraLoader` and `__init__.py` are not modified
   (the loader just carries the state_dict via MODEL_PATCH; all routing decisions live in
   `controlnet.py`).

---

## 竭｡ New / Modified Files

| File | Type | Change |
|---|---|---|
| `nodes/controlnet.py` | modified (+382 lines) | Core of the openpose route. 13 new functions + routing / INPUT_TYPES changes |
| `README.md` | modified (+60) | New "Krea2 Control (Depth & OpenPose)" section |
| `zhmd/README.md` | modified | Chinese README, same content |
| `RELEASE_NOTES/changelog.md` | modified (+10) | v2.5.5 entry |
| `zhmd/changelogzh.md` | modified (+9) | Chinese changelog, same content |
| `RELEASE_NOTES/v2.5.5.md` | **new** | This technical documentation |

`__init__.py` is **unchanged** (requirement 3).

---

## 竭｢ Full New / Modified Code

### 3.1 `nodes/controlnet.py` changes

#### (a) Added imports

```python
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
```

#### (b) `_krea2_target_key`: txtfusion routing added

```python
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
    if base.startswith("txtfusion."):          # 竊・added
        return f"diffusion_model.{base}.weight"
    return None
```

#### (c) New function group (`_classify_krea2_control_type` .. `_apply_krea2_openpose_control`)

```python
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
    snapped to /16 窶・the ai-toolkit krea2 reference-latent convention."""
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
    denoise 窶・matching ai-toolkit kv_cache-trained Krea2 control LoRAs.
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
    # attention keys each step 窶・the ai-toolkit kv_cache training convention
    # this control LoRA was trained with (see krea2_controlnet_pose.json).
    if use_kv_cache:
        dit = model_patched.get_model_object("diffusion_model")
        model_patched.add_object_patch(
            "diffusion_model.forward",
            _krea2_make_ref_kv_forward(dit),
        )
        logger.info("[Krea2OpenposeControl] kv_cache isolated-ref forward armed")
```

#### (d) INPUT_TYPES change (`use_kv_cache` added)

```python
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
```

#### (e) Routing change (depth / openpose exclusive)

```python
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
```

### 3.2 Documentation changes (summary)

- `README.md` / `zhmd/README.md`: node descriptions updated + new
  `### Krea2 Control (Depth & OpenPose)` section (`#### Krea2 Depth Control` /
  `#### Krea2 OpenPose Control`).
- `RELEASE_NOTES/changelog.md` / `zhmd/changelogzh.md`: v2.5.5 entry added.

---

## 竭｣ Detailed Code Explanation

### 4.0 Overall Architecture

```
Krea2ControlNetLoraLoader (unchanged)
        笏・ MODEL_PATCH (carries the LoRA safetensors state_dict)
        笆ｼ
NunchakuQI&ZITDiffsynthControlnet.diffsynth_controlnet_nunchaku
        笏・ _classify_controlnet_target(model, model_patch)
        笏・   竊・"zimage" | "nunchaku_qwenimage" | "qwenimage_standard" | "krea2" | "unknown"
        笆ｼ  route == "krea2"
   _classify_krea2_control_type(state_dict)      竊・presence of first.weight
        笏懌楳 "depth"    竊・_apply_krea2_control()        (existing, unchanged)
        笏披楳 "openpose" 竊・_apply_krea2_openpose_control() (new)
                          笏懌楳 add_patches(256 block LoRA keys)
                          笏懌楳 extra_conds patch (reference_latents injection)
                          笏披楳 diffusion_model.forward patch (kv_cache mode)
```

### 4.1 `_classify_krea2_control_type` 窶・depth / openpose exclusive classification

Decides by the presence of a `first.weight`-style key in the LoRA state_dict.

- **depth**: has the expanded `first.weight` ((6144, 128)) 竊・existing depth route
- **openpose**: no `first` key (pure block LoRA) 竊・openpose route

The two files are structurally distinct (depth: 861MB float32 rank64 + first expansion /
openpose: 228MB bf16 rank32 pure LoRA), so there is no room for misclassification.
The execute side uses `if/elif`, so **both routes can never be applied simultaneously**.

### 4.2 `_krea2_fit_ref_image` 窶・reference image preprocessing

Follows the ai-toolkit Krea2 reference-latent convention:

- Downscale **only** (never upscale) to at most **1MP** (1024ﾃ・024)
- Snap width/height to **/16** (safe for the DiT patch=2 tokenization)
- Resize via `common_upscale(..., "area", "disabled")` (area = suitable for downscaling)

### 4.3 `_krea2_make_ref_cond_patch` 窶・native ref injection (extra_conds patch)

The Krea2 core model (`Krea2` class in `comfy/model_base.py`) consumes
`reference_latents` / `reference_latents_method` produced by `extra_conds`.

- `extra_conds()`: adds `ref_latents` (`CONDList`) and
  `ref_latents_method = "index_timestep_zero"` to the output dict
- `extra_conds_shapes()`: adds the flattened shape `[1, 16, prod//16]` for memory planning
- **Double-injection guard**: skipped when `kwargs["reference_latents"]` already exists
  (= when `TextEncodeKrea2OstrisEdit` is used in parallel) or `ref_latents` is already in `out`.
  竊・Works standalone, and is safe to combine with a VLM text encoder.

### 4.4 `_krea2_ref_pack` / `_krea2_patchify` 窶・ref tokenization and RoPE positions

`_krea2_patchify` is the regular image tokenization `(B,C,H,W) 竊・(B, L, Cﾂｷpﾂｲ)`.

`_krea2_ref_pack` tokenizes reference latents and sets the **axis-0 index to i+1**
(the ai-toolkit convention: text=0, ref1=1, ref2=2, ...):

```
rid[..., 0] = i        竊・reference index (axis-0)
rid[..., 1] = y grid
rid[..., 2] = x grid
```

- 5D latents `(B, C, T, H, W)` (Wan21 VAE, T=1) are reshaped to 4D
- `pad_to_patch_size` + `repeat_to_batch_size` handle padding and batch alignment

### 4.5 `_krea2_ref_attn_kv` 窶・Attention with K/V capture / injection

Re-implements Krea2 `Attention` with two extension points:

- **capture mode** (`kv_capture`): records `(k, v)` after RoPE, before GQA expansion
- **cache mode** (`kv_cache`): appends the precomputed ref K/V to the key side via `torch.cat`

Order: `wq/wk/wv/gate` 竊・reshape 竊・`qknorm` 竊・`apply_rope` 竊・capture/cache 竊・
GQA expansion (`repeat_interleave`) 竊・`optimized_attention_masked` 竊・`wo(outﾂｷsigmoid(gate))`.

### 4.6 `_krea2_ref_precompute_kv` 窶・t=0 reference-only pass

The **core of kv_cache mode**. Runs only the reference tokens through all blocks and
records each block's post-RoPE K/V:

1. `_krea2_ref_pack` produces `(reftok, refpos)`
2. `dit.first(reftok)` embeds the tokens
3. Conditioned on the **t=0** time vector (`timestep_embedding(zeros)`)
4. Runs the 28 blocks with `kv_capture`, collecting `ref_kv` (28 (k, v) pairs)

Reference tokens never see text or noisy images, so **one pass serves the whole denoise**.

### 4.7 `_krea2_forward_with_cached_refs` 窶・the denoising forward

A re-implementation of the regular Krea2 forward. The live sequence is only
"text + noisy image tokens"; each block's attention appends the cached ref K/V
as extra keys:

- 5D inputs (temporal) are flattened to 4D and restored on output
- Text processed via `_unpack_context` 竊・`txtfusion` 竊・`txtmlp`
- RoPE positions: text at axis-0=0, image tokens on a y/x grid
- Output: `dit.last` 竊・slice image tokens 竊・patch unpatch 竊・restore original size

### 4.8 `_krea2_ref_fingerprint` 窶・content key for the cache

`ref_latents` tensors are rebuilt every step, so object identity cannot be used.
Shape + sum + sum-of-squares serve as a content-based key.

### 4.9 `_krea2_make_ref_kv_forward` 窶・forward replacement state management

Replaces `diffusion_model.forward`:

1. No `ref_latents` 竊・**delegate to the original forward** (identical to non-openpose behavior)
2. **New-run detection**: sigmas decrease monotonically within a run, so
   `sig > last_sigma` (or matching `sample_sigmas[0]`) clears the cache
3. On cache miss, `_krea2_ref_precompute_kv` computes and stores the K/V
4. `_krea2_forward_with_cached_refs` runs the denoise

### 4.10 `_apply_krea2_openpose_control` 窶・entry point from the node

1. **Block LoRA**: `_krea2_build_block_patches` (existing) builds the 256 keys of
   `diffusion_model.blocks.*` + `diffusion_model.txtfusion.*` as `LoRAAdapter` and
   applies them via `add_patches(strength_patch=strength)`
2. **Ref injection**: `_krea2_fit_ref_image` 竊・`vae.encode` 竊・`_krea2_make_ref_cond_patch`
   registered as `extra_conds` / `extra_conds_shapes` object patches
3. **kv_cache**: with `use_kv_cache=True` (default), `diffusion_model.forward` is replaced
   by `_krea2_make_ref_kv_forward`
4. Missing vae/image 竊・LoRA-only with a warning log (no crash even without ref injection)

### 4.11 Exclusivity Guarantee (important)

| Route | Change | Basis |
|---|---|---|
| zimage | unchanged | `_classify_controlnet_target` branch code identical since 75f99be |
| nunchaku_qwenimage | unchanged | same |
| qwenimage_standard | unchanged | same |
| krea2 depth | **indentation only** | re-indented for the `if/elif` structure (verified via `git diff`) |
| krea2 openpose | new | new functions + routing |

- `use_kv_cache` is an **optional** input (default True), so existing workflows stay
  compatible. It is only referenced by the openpose route and never reaches the other 4.
- `_classify_controlnet_target` checks zimage 竊・nunchaku 竊・qwenimage_standard 竊・krea2 in order.
  `SingleStreamDiT` exists only in `comfy/ldm/krea2/model.py`, so no other model family
  can be misclassified as Krea2.

### 4.12 Verification Results

- Syntax check / import via `python_embeded` succeeded (INPUT_TYPES and signature verified)
- Unit tests: patchify shapes, ref_pack axis-0=1 grid / 5D handling, fingerprint,
  1MP snap, and that the exclusive routes (zimage / nunchaku / qwenimage_standard / depth)
  are unchanged
- **Live INT8 verification**: on HSWQ `moodyKrea2Mix_v50_sci_1off_convrot_int8.safetensors`,
  `patch_weight_to_device` bake confirmed the LoRA delta reaches the weights
  (relative change 7.76%). The HSWQ v7 bake hook also reported
  `int8_baked=256, patches_left=0`.
- Live run log: `route=krea2` 竊・`sub-route: openpose` 竊・`Applied 256 block LoRA patches` 竊・
  `control ref latent shape=(1, 16, 1, 128, 128)` 竊・`kv_cache isolated-ref forward armed`
  竊・12/12 steps completed (no errors).

