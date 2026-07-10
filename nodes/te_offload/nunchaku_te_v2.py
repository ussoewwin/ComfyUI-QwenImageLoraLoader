"""
nodes/te_offload/nunchaku_te_v2.py

V2 text encoder loader nodes for all nunchaku TE architectures.

Adds a single ``offload_after_encode`` toggle that moves the entire encoder
(LLM + ViT for edit model, LLM-only for text-only / Klein) from CUDA to CPU
the moment encode_token_weights() returns, freeing ~4-5 GB of VRAM for the
diffusion model during KSampler passes.

On the *next* encode call the encoder is automatically moved back to CUDA
before the forward pass, so second/third cycle workflows work correctly.

Covers all three nunchaku TE types:
  - NunchakuQwenImageEditEncoderLoaderV2     (Qwen2.5-VL edit, vision-aware)
  - NunchakuQwenImageTextEncoderLoaderV2     (Qwen2.5-VL text-only path)
  - NunchakuQwen3TextEncoderLoaderV2         (FLUX.2 Klein, Qwen3-4B/8B)

Drop-in replacements for the corresponding nunchaku loader nodes.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys

import torch
import comfy.model_management
import folder_paths

logger = logging.getLogger(__name__)


# ===========================================================================
# Nunchaku TE module accessor
# ===========================================================================

_nunchaku_te_mod = None
_nunchaku_te_tried = False


def _get_nunchaku_te_module():
    """
    Lazily obtain nunchaku's ``qwen_text_encoder`` module.

    Strategy:
      1. Check ``sys.modules`` (fastest — module already loaded by ComfyUI).
      2. Load it from disk via ``importlib`` using ``folder_paths.base_path``
         to construct the path robustly without hard-coding drive letters.

    Returns the module object or raises ``RuntimeError`` if not found.
    """
    global _nunchaku_te_mod, _nunchaku_te_tried
    if _nunchaku_te_tried:
        if _nunchaku_te_mod is None:
            raise RuntimeError(
                "[TEv2] ComfyUI-nunchaku not found. "
                "Install it in custom_nodes/ and restart ComfyUI."
            )
        return _nunchaku_te_mod

    _nunchaku_te_tried = True

    # 1. Already in sys.modules?
    for _name, _mod in sys.modules.items():
        if (
            _mod is not None
            and "qwen_text_encoder" in _name
            and hasattr(_mod, "NunchakuQwenImageEditEncoderLoader")
        ):
            _nunchaku_te_mod = _mod
            logger.debug("[TEv2] Found nunchaku TE module in sys.modules: %s", _name)
            return _nunchaku_te_mod

    # 2. Load from disk
    te_path = os.path.join(
        folder_paths.base_path,
        "custom_nodes", "ComfyUI-nunchaku",
        "nodes", "models", "qwen_text_encoder.py",
    )
    if not os.path.exists(te_path):
        raise RuntimeError(
            f"[TEv2] ComfyUI-nunchaku not found at expected path: {te_path}"
        )

    spec = importlib.util.spec_from_file_location("_nunchaku_qwen_te_v2_ref", te_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _nunchaku_te_mod = mod
    logger.debug("[TEv2] Loaded nunchaku TE module from %s", te_path)
    return _nunchaku_te_mod


# ===========================================================================
# Core offload wrapper
# ===========================================================================

def _wrap_clip_with_offload(clip: object, offload_after_encode: bool) -> object:
    """
    Patch ``clip.cond_stage_model.encode_token_weights`` in-place so that
    after each successful encode the nunchaku encoder is moved to CPU and
    CUDA cache is freed.  Before the *next* call, the encoder is moved back
    to CUDA automatically.

    Key subtleties handled:
    * ``nunchaku_encoder`` is stored via ``object.__setattr__``, bypassing
      ``nn.Module`` submodule registration.  ``nn.Module.to()`` ignores it;
      we must call ``encoder.to()`` directly.
    * ``_encoder_device`` on the TE model class records where tensors live.
      Keeping it in sync prevents device-mismatch errors on encode calls.
    * ``_inv_freq_fixed`` (edit encoder only) is a one-shot guard that runs
      ``_move_cpu_tensors_to_device`` once after load.  After a CPU/CUDA
      round-trip, rotary ``original_inv_freq`` ends up on CPU again, so we
      reset the flag so the fix re-runs on the next encode.

    Returns the same ``clip`` object (mutated).
    """
    if not offload_after_encode:
        return clip

    te_model = clip.cond_stage_model

    # Retrieve the encoder reference (bypasses nn.Module submodule tracking)
    encoder = getattr(te_model, "nunchaku_encoder", None)
    if encoder is None:
        logger.warning(
            "[TEv2] nunchaku_encoder not found on %s — offload skipped.",
            type(te_model).__name__,
        )
        return clip

    # Does this TE class use the _inv_freq_fixed safety guard?
    # Present on the edit encoder (Qwen2.5-VL rotary embedding CPU-buffer quirk).
    has_inv_freq_guard = hasattr(te_model, "_inv_freq_fixed")

    # Capture the original bound method once, before we replace it.
    original_encode = te_model.encode_token_weights

    def encode_with_offload(token_weight_pairs):
        compute_device = comfy.model_management.get_torch_device()

        # ── 1. Re-load to CUDA if previously offloaded ─────────────────────
        try:
            current_device = next(encoder.parameters()).device
        except StopIteration:
            # Encoder has no registered parameters (edge case).
            current_device = torch.device("cpu")

        if current_device.type != compute_device.type:
            logger.info(
                "[TE Offload] ↑ %s: CPU → %s",
                type(te_model).__name__,
                compute_device,
            )
            encoder.to(compute_device)
            # Sync _encoder_device so encode_token_weights targets the right device
            object.__setattr__(te_model, "_encoder_device", compute_device)
            # Reset the one-shot inv_freq guard so it runs again after the move
            if has_inv_freq_guard:
                object.__setattr__(te_model, "_inv_freq_fixed", False)

        # ── 2. Run the original encode ──────────────────────────────────────
        result = original_encode(token_weight_pairs)

        # ── 3. Offload encoder to CPU ───────────────────────────────────────
        logger.info("[TE Offload] ↓ %s: CUDA → CPU", type(te_model).__name__)
        encoder.to("cpu")
        object.__setattr__(te_model, "_encoder_device", torch.device("cpu"))
        # Reset guard so it runs cleanly on the *next* re-load to CUDA
        if has_inv_freq_guard:
            object.__setattr__(te_model, "_inv_freq_fixed", False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    # Assign as a plain function on the *instance* (not a method descriptor).
    # When called as `te_model.encode_token_weights(tokens)`, Python looks up
    # the instance dict first, finds our function, and calls it without auto-
    # prepending `self` — exactly what we want.
    te_model.encode_token_weights = encode_with_offload

    logger.info(
        "[TEv2] Post-encode CPU offload patched on %s (inv_freq_guard=%s)",
        type(te_model).__name__,
        has_inv_freq_guard,
    )
    return clip


# ===========================================================================
# Shared filename helper
# ===========================================================================

def _te_filename_list() -> list[str]:
    try:
        # Try nunchaku's own helper (filters to TE checkpoints)
        mod = _get_nunchaku_te_module()
        # nunchaku uses get_filename_list from its utils
        from_nunchaku = getattr(mod, "get_filename_list", None)
        if from_nunchaku is not None:
            return from_nunchaku("text_encoders")
    except Exception:
        pass
    return folder_paths.get_filename_list("text_encoders")


# ===========================================================================
# Base class — shared loading + wrapping logic
# ===========================================================================

class _BaseNunchakuTELoaderV2:
    """
    Shared base for all V2 TE loader nodes.

    Subclasses set ``_NUNCHAKU_LOADER_CLS`` to the name of the corresponding
    nunchaku loader class (string), and ``TITLE`` for the UI display.
    """

    _NUNCHAKU_LOADER_CLS: str  # e.g. "NunchakuQwenImageEditEncoderLoader"

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "Nunchaku"

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _delegate_load(self, model_path: str) -> object:
        """Instantiate nunchaku's loader and return its CLIP output."""
        mod = _get_nunchaku_te_module()
        cls = getattr(mod, self._NUNCHAKU_LOADER_CLS, None)
        if cls is None:
            raise RuntimeError(
                f"[TEv2] {self._NUNCHAKU_LOADER_CLS} not found in "
                f"nunchaku qwen_text_encoder module."
            )
        logger.info("[TEv2] Delegating load to %s", self._NUNCHAKU_LOADER_CLS)
        return cls().load_text_encoder(model_path)[0]

    # ---------------------------------------------------------------------------
    # Node entry point
    # ---------------------------------------------------------------------------

    def load_text_encoder(self, model_path: str, offload_after_encode: bool = True) -> tuple:
        clip = self._delegate_load(model_path)
        clip = _wrap_clip_with_offload(clip, offload_after_encode)
        return (clip,)


# ===========================================================================
# V2 Loader node — Qwen2.5-VL Edit (vision-aware, primary use-case)
# ===========================================================================

class NunchakuQwenImageEditEncoderLoaderV2(_BaseNunchakuTELoaderV2):
    """
    V2 loader for the Nunchaku Qwen2.5-VL edit (multimodal) text encoder.

    Identical to ``NunchakuQwenImageEditEncoderLoader`` in nunchaku, but with
    an ``offload_after_encode`` toggle that frees ~4-5 GB of VRAM after each
    encode call, making that memory available to the diffusion model during
    KSampler passes. The encoder is automatically restored to CUDA before the
    next encode, so multi-KSampler workflows function correctly on every cycle.

    Drop-in replacement — swap this node for the nunchaku loader, enable the
    toggle, and enjoy extra free VRAM during sampling.
    """

    _NUNCHAKU_LOADER_CLS = "NunchakuQwenImageEditEncoderLoader"
    TITLE = "Nunchaku Qwen2.5-VL Edit Encoder Loader V2 (TE Offload)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    _te_filename_list(),
                    {
                        "tooltip": (
                            "Nunchaku-quantised Qwen2.5-VL edit/multimodal encoder "
                            "checkpoint (.safetensors)."
                        )
                    },
                ),
                "offload_after_encode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "After each encode_token_weights call, move the entire text "
                            "encoder (LLM + vision ViT, ~4-5 GB) to CPU and free VRAM. "
                            "The encoder is automatically restored to CUDA before the next "
                            "encode, so multi-KSampler workflows work correctly on every run."
                        ),
                    },
                ),
            }
        }


# ===========================================================================
# V2 Loader node — Qwen2.5-VL text-only
# ===========================================================================

class NunchakuQwenImageTextEncoderLoaderV2(_BaseNunchakuTELoaderV2):
    """
    V2 loader for the Nunchaku Qwen2.5-VL text-only encoder (QwenImage path).

    Adds the same ``offload_after_encode`` toggle as the edit variant.
    """

    _NUNCHAKU_LOADER_CLS = "NunchakuQwenImageTextEncoderLoader"
    TITLE = "Nunchaku Qwen2.5-VL Text Encoder Loader V2 (TE Offload)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    _te_filename_list(),
                    {
                        "tooltip": (
                            "Nunchaku-quantised Qwen2.5-VL text-only encoder "
                            "checkpoint (.safetensors)."
                        )
                    },
                ),
                "offload_after_encode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "After encoding, move the text encoder to CPU and free its VRAM "
                            "for the diffusion model. Encoder restores to CUDA before the next encode."
                        ),
                    },
                ),
            }
        }


# ===========================================================================
# V2 Loader node — Qwen3 / FLUX.2 Klein
# ===========================================================================

class NunchakuQwen3TextEncoderLoaderV2(_BaseNunchakuTELoaderV2):
    """
    V2 loader for the Nunchaku Qwen3 text encoder (FLUX.2 Klein, Qwen3-4B/8B).

    Adds the same ``offload_after_encode`` toggle. Klein does not have the
    Qwen2.5-VL rotary-buffer quirk, so the offload/restore cycle is simpler.
    """

    _NUNCHAKU_LOADER_CLS = "NunchakuQwen3TextEncoderLoader"
    TITLE = "Nunchaku Qwen3 Text Encoder Loader V2 (TE Offload)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    _te_filename_list(),
                    {
                        "tooltip": (
                            "Nunchaku-quantised Qwen3 text encoder checkpoint "
                            "(qwen3-4b or qwen3-8b, .safetensors)."
                        )
                    },
                ),
                "offload_after_encode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "After encoding, move the Qwen3 text encoder to CPU and free VRAM. "
                            "Encoder restores to CUDA automatically before the next encode call."
                        ),
                    },
                ),
            }
        }


# ===========================================================================
# Node registration dicts (consumed by __init__.py)
# ===========================================================================

GENERATED_NODES = {
    "NunchakuQwenImageEditEncoderLoaderV2": NunchakuQwenImageEditEncoderLoaderV2,
    "NunchakuQwenImageTextEncoderLoaderV2": NunchakuQwenImageTextEncoderLoaderV2,
    "NunchakuQwen3TextEncoderLoaderV2": NunchakuQwen3TextEncoderLoaderV2,
}

GENERATED_DISPLAY_NAMES = {
    "NunchakuQwenImageEditEncoderLoaderV2": "Nunchaku Qwen2.5-VL Edit Encoder Loader V2 (TE Offload)",
    "NunchakuQwenImageTextEncoderLoaderV2": "Nunchaku Qwen2.5-VL Text Encoder Loader V2 (TE Offload)",
    "NunchakuQwen3TextEncoderLoaderV2": "Nunchaku Qwen3 Text Encoder Loader V2 (TE Offload)",
}
