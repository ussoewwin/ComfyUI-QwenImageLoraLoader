
from __future__ import annotations

import inspect
import logging
import os
import re
import sys
import types
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

_svdq_from_linear_patched: bool = False
_zimage_fast_disk_patched: bool = False
_ZIMAGE_FAST_DISK_TAG = "_qwen_lora_loader_zimage_fast_disk_patch"
_qwen_apply_rotary_emb_compat_applied: bool = False
_copy_params_patched: bool = False

_ROTARY_SHIM_TAG = "_qwen_lora_loader_rotary_shim"

_NUNCHAKU_QWENIMAGE_APPLY_ROTARY_PATTERNS = (
    re.compile(
        r"from\s+comfy\.ldm\.qwen_image\.model\s+import\s+\([^)]*\bapply_rotary_emb\b",
        re.MULTILINE | re.DOTALL,
    ),
    re.compile(
        r"from\s+comfy\.ldm\.qwen_image\.model\s+import\s+[^\n#]*\bapply_rotary_emb\b",
    ),
)

def _rotary_compat_disabled_by_env() -> bool:
    value = os.environ.get("QWENIMAGE_ROTARY_COMPAT", "").strip().lower()
    return value in ("0", "false", "no", "off", "disable", "disabled")


def _mark_rotary_shim(fn: Callable) -> Callable:
    setattr(fn, _ROTARY_SHIM_TAG, True)
    return fn


def _rotary_shim_installed_on(qwen_image_model) -> bool:
    fn = getattr(qwen_image_model, "apply_rotary_emb", None)
    return fn is not None and getattr(fn, _ROTARY_SHIM_TAG, False)


def _comfyui_has_native_apply_rotary_emb(qwen_image_model) -> bool:
    return (
        getattr(qwen_image_model, "apply_rotary_emb", None) is not None
        and not _rotary_shim_installed_on(qwen_image_model)
    )


def _find_nunchaku_qwenimage_py() -> Optional[str]:
    custom_nodes = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    for folder in ("ComfyUI-nunchaku", "ComfyUI_Nunchaku", "comfyui-nunchaku"):
        path = os.path.join(custom_nodes, folder, "models", "qwenimage.py")
        if os.path.isfile(path):
            return path
    return None


def _nunchaku_qwenimage_still_imports_apply_rotary_emb() -> Optional[bool]:
    """
    True: nunchaku still imports apply_rotary_emb (compat may be needed).
    False: nunchaku source no longer imports it (skip shim).
    None: qwenimage.py not found (apply only if symbol missing).
    """
    path = _find_nunchaku_qwenimage_py()
    if path is None:
        return None

    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        logger.debug("Could not read %s for rotary compat probe: %s", path, exc)
        return None

    for pattern in _NUNCHAKU_QWENIMAGE_APPLY_ROTARY_PATTERNS:
        if pattern.search(text):
            return True
    return False


_ROTARY_TARGET_MODULE = "comfy.ldm.qwen_image.model"
_ROTARY_PARENT_MODULE = "comfy.ldm.qwen_image"
_rotary_hook_installed: bool = False


def _install_rotary_alias(qwen_image_model) -> bool:
    """Alias apply_rope1 onto an already imported comfy.ldm.qwen_image.model module."""
    global _qwen_apply_rotary_emb_compat_applied
    if qwen_image_model is None:
        return False
    if _rotary_shim_installed_on(qwen_image_model):
        _qwen_apply_rotary_emb_compat_applied = True
        return True
    if _comfyui_has_native_apply_rotary_emb(qwen_image_model):
        logger.info(
            "apply_rotary_emb compat skipped: ComfyUI already exports apply_rotary_emb"
        )
        return False

    nunchaku_needs = _nunchaku_qwenimage_still_imports_apply_rotary_emb()
    if nunchaku_needs is False:
        logger.info(
            "apply_rotary_emb compat skipped: ComfyUI-nunchaku no longer imports apply_rotary_emb"
        )
        return False

    from comfy.ldm.flux.math import apply_rope1

    qwen_image_model.apply_rotary_emb = _mark_rotary_shim(apply_rope1)
    _qwen_apply_rotary_emb_compat_applied = True
    logger.info(
        "Patched comfy.ldm.qwen_image.model.apply_rotary_emb -> apply_rope1 "
        "(ComfyUI-nunchaku Qwen Image compat)"
    )
    return True


class _RotaryAliasLoader:
    """Loader wrapper that applies the alias right after the target module executes."""

    def __init__(self, loader) -> None:
        self._loader = loader

    def __getattr__(self, item):
        return getattr(self._loader, item)

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        try:
            _install_rotary_alias(module)
        except Exception as exc:
            logger.error("Failed to apply apply_rotary_emb compat patch: %s", exc)


class _RotaryCompatFinder:
    """Resolve comfy.ldm.qwen_image.model through its parent package without importing it.

    Importing comfy here would pull torch during prestartup, which is exactly the state
    ComfyUI warns about. The finder only wraps the loader, so comfy and torch enter
    sys.modules at the moment the model module is actually imported, as usual.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _ROTARY_TARGET_MODULE or fullname in sys.modules:
            return None
        parent = sys.modules.get(_ROTARY_PARENT_MODULE)
        parent_path = list(getattr(parent, "__path__", None) or [])
        if not parent_path:
            return None
        import importlib.machinery

        spec = importlib.machinery.PathFinder.find_spec(fullname, parent_path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _RotaryAliasLoader(spec.loader)
        return spec


def apply_qwen_image_apply_rotary_emb_compat() -> bool:
    """Arm the apply_rotary_emb compat shim without importing comfy or torch.

    ComfyUI >= 0.24 removed comfy.ldm.qwen_image.model.apply_rotary_emb and
    ComfyUI-nunchaku still imports it. The alias is applied by an import hook: at once
    if the module is already imported, otherwise as soon as it is imported - which is
    still before ComfyUI-nunchaku reads the symbol.
    """
    global _qwen_apply_rotary_emb_compat_applied, _rotary_hook_installed
    if _qwen_apply_rotary_emb_compat_applied:
        return True
    if _rotary_compat_disabled_by_env():
        logger.info(
            "apply_rotary_emb compat skipped (QWENIMAGE_ROTARY_COMPAT is disabled)"
        )
        return False

    loaded = sys.modules.get(_ROTARY_TARGET_MODULE)
    if loaded is not None:
        try:
            return _install_rotary_alias(loaded)
        except Exception as e:
            logger.error("Failed to apply apply_rotary_emb compat patch: %s", e)
            return False

    if _rotary_hook_installed:
        return True
    try:
        sys.meta_path.insert(0, _RotaryCompatFinder())
        _rotary_hook_installed = True
        logger.info(
            "apply_rotary_emb compat armed: alias applies when comfy.ldm.qwen_image.model "
            "is imported (torch is not imported during prestartup)"
        )
        return True
    except Exception as e:
        logger.error("Failed to arm apply_rotary_emb compat hook: %s", e)
        return False


def _torch_device_fallback() -> torch.device:
    import torch

    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _patched_svdqw4a4_from_linear(cls, linear: torch.nn.Linear, **kwargs):
    """
    Replacement for SVDQW4A4Linear.from_linear that works when linear.weight is None
    (ComfyUI disable_weight_init.Linear on Windows + AIMDO before load_model_weights).

    Also avoids ``kwargs.pop("torch_dtype", linear.weight.dtype)``: in Python the default
    is always evaluated, so it crashed even when torch_dtype was passed in kwargs.
    """
    in_features = kwargs.pop("in_features", linear.in_features)
    if "torch_dtype" in kwargs:
        torch_dtype = kwargs.pop("torch_dtype")
    elif linear.weight is not None:
        torch_dtype = linear.weight.dtype
    else:
        raise TypeError(
            "SVDQW4A4Linear.from_linear: linear.weight is None; pass torch_dtype= "
            "(ComfyUI lazy Linear before state dict load)."
        )
    if "device" in kwargs:
        device = kwargs.pop("device")
    elif linear.weight is not None:
        device = linear.weight.device
    else:
        device = _torch_device_fallback()
    return cls(
        in_features=in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
        torch_dtype=torch_dtype,
        device=device,
        **kwargs,
    )


def _make_patched_fuse_to_svdquant_linear(zimage_module: types.ModuleType):
    """Build fuse_to_svdquant_linear with the same lazy-Linear / pop-default fix."""
    from nunchaku.models.linear import SVDQW4A4Linear

    add_comfy_cast_weights_attr = zimage_module.add_comfy_cast_weights_attr

    def fuse_to_svdquant_linear(comfy_linear1: torch.nn.Linear, comfy_linear2: torch.nn.Linear, **kwargs):
        assert comfy_linear1.in_features == comfy_linear2.in_features
        assert comfy_linear1.bias is None and comfy_linear2.bias is None
        if "torch_dtype" in kwargs:
            torch_dtype = kwargs.pop("torch_dtype")
        elif comfy_linear1.weight is not None:
            torch_dtype = comfy_linear1.weight.dtype
        else:
            raise TypeError(
                "fuse_to_svdquant_linear: linear.weight is None; pass torch_dtype= "
                "(ComfyUI lazy Linear before state dict load)."
            )
        if "device" in kwargs:
            device = kwargs.pop("device")
        elif comfy_linear1.weight is not None:
            device = comfy_linear1.weight.device
        else:
            device = _torch_device_fallback()
        svdq_linear = SVDQW4A4Linear(
            comfy_linear1.in_features,
            comfy_linear1.out_features + comfy_linear2.out_features,
            bias=False,
            torch_dtype=torch_dtype,
            device=device,
            **kwargs,
        )
        add_comfy_cast_weights_attr(svdq_linear, comfy_linear1)
        return svdq_linear

    return fuse_to_svdquant_linear


def apply_svdqw4a4_lazy_linear_patch() -> bool:
    """Patch nunchaku SVDQW4A4Linear.from_linear for ComfyUI lazy Linear + eager pop default bug."""
    global _svdq_from_linear_patched
    if _svdq_from_linear_patched:
        return True
    try:
        from nunchaku.models.linear import SVDQW4A4Linear
    except ImportError:
        logger.warning("nunchaku.models.linear.SVDQW4A4Linear not importable; lazy-Linear patch skipped.")
        return False

    SVDQW4A4Linear.from_linear = classmethod(_patched_svdqw4a4_from_linear)
    _svdq_from_linear_patched = True
    logger.info("Patched SVDQW4A4Linear.from_linear for ComfyUI lazy Linear (Windows AIMDO / pre-weight load).")
    return True


def apply_nunchaku_zimage_fuse_lazy_linear_patch() -> bool:
    """
    Patch ComfyUI-nunchaku ``fuse_to_svdquant_linear`` if that module is already loaded
    (same pop-default and weight-None issues as from_linear).
    """
    for _name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (
            hasattr(mod, "fuse_to_svdquant_linear")
            and hasattr(mod, "ComfyNunchakuZImageAttention")
            and hasattr(mod, "add_comfy_cast_weights_attr")
        ):
            continue
        fn = mod.fuse_to_svdquant_linear
        if getattr(fn, "_qwen_lora_loader_lazy_linear_patch", False):
            return True
        patched = _make_patched_fuse_to_svdquant_linear(mod)
        patched._qwen_lora_loader_lazy_linear_patch = True
        mod.fuse_to_svdquant_linear = patched
        logger.info("Patched fuse_to_svdquant_linear in %s for ComfyUI lazy Linear.", getattr(mod, "__name__", "?"))
        return True
    return False


def schedule_nunchaku_zimage_fuse_patch_retries() -> None:
    """
    ComfyUI-QwenImageLoraLoader often loads before ComfyUI-nunchaku (folder name order).
    Then ``models.zimage`` is not in sys.modules yet; retry briefly after startup.
    """
    import threading

    def try_once(attempt: int = 0) -> None:
        fuse_ok = apply_nunchaku_zimage_fuse_lazy_linear_patch()
        if fuse_ok:
            apply_nunchaku_zimage_fast_disk_patch()
        if fuse_ok and _zimage_fast_disk_patched:
            return
        if attempt < 24:
            threading.Timer(0.25, lambda: try_once(attempt + 1)).start()

    threading.Timer(0.05, lambda: try_once(0)).start()


def forward_with_manual_planar_injection(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    timestep_zero_index=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for the transformer block with Manual Planar Injection support for LoRA.
    Monkey-patched into NunchakuQwenImageTransformerBlock by ComfyUI-QwenImageLoraLoader.
    """
    # Get modulation parameters for both streams
    img_mod_params = self.img_mod(temb)  # [B, 6*dim]
    # --- Nunchaku LoRA Patch (Manual Planar Injection) ---
    if hasattr(self.img_mod[1], "_nunchaku_lora_bundle"):
            A, B = self.img_mod[1]._nunchaku_lora_bundle
            # The Linear layer receives SiLU(temb), so we must apply it to LoRA input too
            input_tensor = self.img_mod[0](temb)
            # Cast to LoRA dtype (FP16) before matmul to avoid BFloat16 vs Float16 mismatch
            lora = (input_tensor.to(dtype=A.dtype) @ A.t()) @ B.t()
            img_mod_params = img_mod_params + lora.to(dtype=img_mod_params.dtype, device=img_mod_params.device)
    # -----------------------------------------------------

    txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]
    # --- Nunchaku LoRA Patch (Manual Planar Injection) ---
    if hasattr(self.txt_mod[1], "_nunchaku_lora_bundle"):
            A, B = self.txt_mod[1]._nunchaku_lora_bundle
            # The Linear layer receives SiLU(temb), so we must apply it to LoRA input too
            input_tensor = self.txt_mod[0](temb)
            lora = (input_tensor.to(dtype=A.dtype) @ A.t()) @ B.t()
            txt_mod_params = txt_mod_params + lora.to(dtype=txt_mod_params.dtype, device=txt_mod_params.device)
    # -----------------------------------------------------

    # Nunchaku's mod_params is [B, 6*dim] instead of [B, dim*6]
    img_mod_params = (
        img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(img_mod_params.shape[0], -1)
    )
    txt_mod_params = (
        txt_mod_params.view(txt_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(txt_mod_params.shape[0], -1)
    )

    img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
    txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

    # Process image stream - norm1 + modulation
    img_normed = self.img_norm1(hidden_states)
    img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

    # Process text stream - norm1 + modulation
    txt_normed = self.txt_norm1(encoder_hidden_states)
    txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

    # Joint attention computation (DoubleStreamLayerMegatron logic)
    attn_output = self.attn(
        hidden_states=img_modulated,  # Image stream ("sample")
        encoder_hidden_states=txt_modulated,  # Text stream ("context")
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        image_rotary_emb=image_rotary_emb,
    )

    # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
    img_attn_output, txt_attn_output = attn_output

    # Apply attention gates and add residual (like in Megatron)
    hidden_states = hidden_states + img_gate1 * img_attn_output
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

    # Process image stream - norm2 + MLP
    img_normed2 = self.img_norm2(hidden_states)
    img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
    img_mlp_output = self.img_mlp(img_modulated2)
    hidden_states = hidden_states + img_gate2 * img_mlp_output

    # Process text stream - norm2 + MLP
    txt_normed2 = self.txt_norm2(encoder_hidden_states)
    txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
    txt_mlp_output = self.txt_mlp(txt_modulated2)
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

    return encoder_hidden_states, hidden_states


def _safe_copy_params_into(src: torch.nn.Module, dst: torch.nn.Module, non_blocking: bool = True):
    """
    Safely copy parameters and buffers from src to dst by name matching (with positional fallback),
    and correctly synchronize wtscale attribute without attribute deletion.
    """
    import torch

    with torch.no_grad():
        src_params = dict(src.named_parameters())
        dst_params = dict(dst.named_parameters())
        if src_params and dst_params and set(src_params.keys()) == set(dst_params.keys()):
            for name, pd in dst_params.items():
                pd.copy_(src_params[name], non_blocking=non_blocking)
        else:
            for ps, pd in zip(src.parameters(), dst.parameters()):
                pd.copy_(ps, non_blocking=non_blocking)

        src_buffers = dict(src.named_buffers())
        dst_buffers = dict(dst.named_buffers())
        if src_buffers and dst_buffers and set(src_buffers.keys()) == set(dst_buffers.keys()):
            for name, bd in dst_buffers.items():
                bd.copy_(src_buffers[name], non_blocking=non_blocking)
        else:
            for bs, bd in zip(src.buffers(), dst.buffers()):
                bd.copy_(bs, non_blocking=non_blocking)

        src_modules = dict(src.named_modules())
        dst_modules = dict(dst.named_modules())
        if src_modules and dst_modules and set(src_modules.keys()) == set(dst_modules.keys()):
            for name, md in dst_modules.items():
                ms = src_modules[name]
                if hasattr(ms, "wtscale"):
                    md.wtscale = ms.wtscale
                elif hasattr(md, "wtscale"):
                    precision = getattr(md, "precision", "int4")
                    md.wtscale = 1.0 if precision == "nvfp4" else None
        else:
            for ms, md in zip(src.modules(), dst.modules()):
                if hasattr(ms, "wtscale"):
                    md.wtscale = ms.wtscale
                elif hasattr(md, "wtscale"):
                    precision = getattr(md, "precision", "int4")
                    md.wtscale = 1.0 if precision == "nvfp4" else None


def apply_nunchaku_copy_params_patch() -> bool:
    """
    Patch nunchaku copy_params_into to safely handle wtscale attribute differences
    when reusing buffer_blocks during CPU offloading.
    """
    global _copy_params_patched
    if _copy_params_patched:
        return True
    applied = False
    for mod_name in ("nunchaku.utils", "nunchaku.models.utils"):
        try:
            if mod_name in sys.modules:
                mod = sys.modules[mod_name]
                if hasattr(mod, "copy_params_into"):
                    mod.copy_params_into = _safe_copy_params_into
                    applied = True
            else:
                import importlib
                mod = importlib.import_module(mod_name)
                mod.copy_params_into = _safe_copy_params_into
                applied = True
        except Exception:
            pass
    if applied:
        _copy_params_patched = True
        logger.info("Patched nunchaku.copy_params_into for safe CPU offload buffer parameter copying.")
    return applied


def apply_nunchaku_zimage_fast_disk_patch() -> bool:
    """
    Patch ``ZImageModelPatcher.__init__`` to accept the ``fast_disk`` keyword that
    ComfyUI >= 0.38 ``ModelPatcher.clone()`` always passes.

    Upstream nunchaku (development stopped) has::

        def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False)

    but v0.38 clone calls ``class_(..., weight_inplace_update=..., fast_disk=...)``,
    so every ``model.clone()`` on a nunchaku Z-Image model (e.g. EasyCache node)
    raises ``TypeError: ZImageModelPatcher.__init__() got an unexpected keyword
    argument 'fast_disk'``. Wrapping the subclass __init__ to swallow/forward the
    kwarg keeps nunchaku untouched and is version-safe: if a future build already
    declares ``fast_disk`` this patch no-ops.
    """
    global _zimage_fast_disk_patched
    if _zimage_fast_disk_patched:
        return True

    # Scan sys.modules ONLY. Never importlib.import_module here: ComfyUI loads
    # packs under names like ``custom_nodes.ComfyUI-nunchaku.model_patcher.zimage``,
    # so importing ``model_patcher.zimage`` would create a *second* module
    # instance and patch a class nobody uses. The retry timer handles the case
    # where this pack's __init__ runs before ComfyUI-nunchaku is imported.
    target_class = None
    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue
        candidate = getattr(module, "ZImageModelPatcher", None)
        if isinstance(candidate, type) and (
            "nunchaku" in module_name or module_name.endswith("model_patcher.zimage")
        ):
            target_class = candidate
            logger.info("Found ZImageModelPatcher in %s for fast_disk compat.", module_name)
            break

    if target_class is None:
        return False

    original_init = target_class.__init__
    if getattr(original_init, _ZIMAGE_FAST_DISK_TAG, False):
        _zimage_fast_disk_patched = True
        return True

    try:
        already_supports = "fast_disk" in inspect.signature(original_init).parameters
    except (TypeError, ValueError):
        already_supports = False
    if already_supports:
        _zimage_fast_disk_patched = True
        logger.info("ZImageModelPatcher.__init__ already accepts fast_disk; compat skipped.")
        return True

    def patched_init(self, *args, **kwargs):
        fast_disk = kwargs.pop("fast_disk", None)
        original_init(self, *args, **kwargs)
        # Parent ModelPatcher.__init__ stores fast_disk; set it explicitly so the
        # clone chain (and __del__/pin paths) see a consistent attribute even if
        # the installed ComfyUI parent predates the parameter.
        self.fast_disk = bool(fast_disk) if fast_disk is not None else getattr(self, "fast_disk", False)

    setattr(patched_init, _ZIMAGE_FAST_DISK_TAG, True)
    target_class.__init__ = patched_init
    _zimage_fast_disk_patched = True
    logger.info("Patched ZImageModelPatcher.__init__ for ComfyUI >= 0.38 fast_disk clone kwarg.")
    return True


def apply_nunchaku_patch():
    """
    Apply ComfyUI-nunchaku compatibility patches (LoRA planar injection + lazy Linear fixes + safe copy_params_into).
    Returns True if at least one patch was applied or was already active.
    """
    rotary_compat = apply_qwen_image_apply_rotary_emb_compat()
    lazy_from = apply_svdqw4a4_lazy_linear_patch()
    lazy_fuse = apply_nunchaku_zimage_fuse_lazy_linear_patch()
    fast_disk_ok = apply_nunchaku_zimage_fast_disk_patch()
    copy_params_ok = apply_nunchaku_copy_params_patch()
    if not lazy_fuse or not fast_disk_ok:
        schedule_nunchaku_zimage_fuse_patch_retries()

    planar_ok = False
    try:
        target_class = None

        try:
            from nunchaku.models.qwenimage import NunchakuQwenImageTransformerBlock

            target_class = NunchakuQwenImageTransformerBlock
        except ImportError:
            pass

        if target_class is None:
            for module_name, module in sys.modules.items():
                if "qwenimage" in module_name and hasattr(module, "NunchakuQwenImageTransformerBlock"):
                    target_class = getattr(module, "NunchakuQwenImageTransformerBlock")
                    logger.info("Found NunchakuQwenImageTransformerBlock in %s", module_name)
                    break

        if target_class:
            logger.info("Applying Manual Planar Injection Monkey Patch to NunchakuQwenImageTransformerBlock")
            target_class.forward = forward_with_manual_planar_injection
            planar_ok = True
        else:
            logger.warning(
                "Could not find NunchakuQwenImageTransformerBlock to patch. "
                "Manual Planar Injection logic will not work if the original file is reverted."
            )

    except Exception as e:
        logger.error("Failed to apply Nunchaku planar patch: %s", e)

    return planar_ok or lazy_from or rotary_compat or copy_params_ok or fast_disk_ok

