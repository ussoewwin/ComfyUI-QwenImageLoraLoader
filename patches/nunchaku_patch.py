
import logging
import os
import re
import sys
import types
from typing import Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_svdq_from_linear_patched: bool = False
_qwen_apply_rotary_emb_compat_applied: bool = False

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

_torch_preimport_warning_suppressed: bool = False
_TORCH_PREIMPORT_WARNING_MARKER = "Torch already imported"


def _torch_warning_suppression_disabled_by_env() -> bool:
    value = os.environ.get("QWENIMAGE_SUPPRESS_TORCH_WARNING", "").strip().lower()
    return value in ("0", "false", "no", "off", "disable", "disabled")


class _TorchPreimportWarningFilter(logging.Filter):
    """Drop ComfyUI's cosmetic 'Torch already imported' warning.

    The apply_rotary_emb compat shim must run in prestartup (before ComfyUI-nunchaku
    loads), and installing it requires importing comfy.ldm modules, which import torch.
    ComfyUI main.py then warns that torch entered sys.modules early. cuda_malloc and all
    CUDA env setup already ran before prestartup, so the early import is harmless; this
    filter hides only that single message and then lets every record through.
    """

    def __init__(self) -> None:
        super().__init__()
        self._suppressed = False

    def filter(self, record: logging.LogRecord) -> bool:
        if self._suppressed:
            return True
        try:
            message = record.getMessage()
        except Exception:
            return True
        if _TORCH_PREIMPORT_WARNING_MARKER in message:
            self._suppressed = True
            return False
        return True


def suppress_torch_preimport_warning() -> bool:
    """Install a one-shot root-logger filter hiding the cosmetic torch pre-import warning.

    Must be called during prestartup (before ComfyUI main.py logs the warning).
    Set QWENIMAGE_SUPPRESS_TORCH_WARNING=0 to keep the warning visible.
    """
    global _torch_preimport_warning_suppressed
    if _torch_preimport_warning_suppressed:
        return True
    if _torch_warning_suppression_disabled_by_env():
        logger.info(
            "Torch pre-import warning suppression skipped "
            "(QWENIMAGE_SUPPRESS_TORCH_WARNING is disabled)"
        )
        return False
    try:
        logging.getLogger().addFilter(_TorchPreimportWarningFilter())
        _torch_preimport_warning_suppressed = True
        return True
    except Exception as exc:
        logger.debug("Failed to install torch pre-import warning filter: %s", exc)
        return False


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


def apply_qwen_image_apply_rotary_emb_compat() -> bool:
    """
    ComfyUI >= 0.24 removed comfy.ldm.qwen_image.model.apply_rotary_emb.
    ComfyUI-nunchaku still imports it; alias to comfy.ldm.flux.math.apply_rope1
    only when the import is still required and ComfyUI has not restored the symbol.
    """
    global _qwen_apply_rotary_emb_compat_applied
    if _qwen_apply_rotary_emb_compat_applied:
        return True

    if _rotary_compat_disabled_by_env():
        logger.info(
            "apply_rotary_emb compat skipped (QWENIMAGE_ROTARY_COMPAT is disabled)"
        )
        return False

    try:
        import comfy.ldm.qwen_image.model as qwen_image_model
        from comfy.ldm.flux.math import apply_rope1

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

        qwen_image_model.apply_rotary_emb = _mark_rotary_shim(apply_rope1)
        _qwen_apply_rotary_emb_compat_applied = True
        logger.info(
            "Patched comfy.ldm.qwen_image.model.apply_rotary_emb -> apply_rope1 "
            "(ComfyUI-nunchaku Qwen Image compat)"
        )
        return True
    except Exception as e:
        logger.error("Failed to apply apply_rotary_emb compat patch: %s", e)
        return False


def _torch_device_fallback() -> torch.device:
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
        if apply_nunchaku_zimage_fuse_lazy_linear_patch():
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


def apply_nunchaku_patch():
    """
    Apply ComfyUI-nunchaku compatibility patches (LoRA planar injection + lazy Linear fixes).
    Returns True if at least one patch was applied or was already active.
    """
    rotary_compat = apply_qwen_image_apply_rotary_emb_compat()
    lazy_from = apply_svdqw4a4_lazy_linear_patch()
    lazy_fuse = apply_nunchaku_zimage_fuse_lazy_linear_patch()
    if not lazy_fuse:
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

    return planar_ok or lazy_from or rotary_compat
