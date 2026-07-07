import logging
from pathlib import Path

import comfy.sd
import comfy.utils


logger = logging.getLogger(__name__)


class Krea2ControlNetLoraLoader:
    """
    Apply a Krea2 Control LoRA (.safetensors) to a MODEL_PATCH.

    This node is intentionally separate from existing Qwen/Z-Image LoRA loaders.
    It performs a lightweight key sanity check before patching, to reduce the
    risk of applying an unrelated LoRA by mistake.
    """

    DEFAULT_LORA_PATH = r"D:\USERFILES\StableDiffusion\models\ControlNet\krea2-depth-control-lora.safetensors"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_patch": ("MODEL_PATCH",),
                "lora_path": ("STRING", {"default": cls.DEFAULT_LORA_PATH, "multiline": False}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL_PATCH",)
    FUNCTION = "load_krea2_controlnet_lora"
    CATEGORY = "advanced/loaders/krea2"
    DESCRIPTION = "Load and apply a Krea2 Control LoRA to MODEL_PATCH."

    def _is_krea2_like_lora(self, state_dict: dict) -> bool:
        if not state_dict:
            return False

        keys = list(state_dict.keys())
        has_lora_pair = any(".lora_up." in k or ".lora_A." in k or ".lora.down." in k for k in keys)
        if not has_lora_pair:
            return False

        krea2_tokens = (
            ".blocks.",
            ".txtfusion.",
            ".first.",
            ".last.",
            ".tmlp.",
            ".txtmlp.",
            ".tproj.",
            ".pe_embedder.",
        )
        return any(any(token in k for token in krea2_tokens) for k in keys)

    def load_krea2_controlnet_lora(self, model_patch, lora_path: str, strength_model: float):
        lora_file = Path(lora_path)
        if not lora_file.exists():
            raise FileNotFoundError(f"Krea2 LoRA file not found: {lora_file}")
        if lora_file.suffix.lower() != ".safetensors":
            raise ValueError(f"Krea2 LoRA must be a .safetensors file: {lora_file}")

        logger.info(f"[Krea2ControlNetLoraLoader] Loading LoRA: {lora_file}")
        lora_state_dict = comfy.utils.load_torch_file(str(lora_file), safe_load=True)
        if not isinstance(lora_state_dict, dict) or len(lora_state_dict) == 0:
            raise ValueError(f"Invalid or empty LoRA state dict: {lora_file}")

        if not self._is_krea2_like_lora(lora_state_dict):
            raise ValueError(
                "This file does not look like a Krea2 LoRA (missing expected Krea2/LoRA key patterns). "
                f"Refusing to apply: {lora_file}"
            )

        patched_model_patch, _ = comfy.sd.load_lora_for_models(
            model=model_patch,
            clip=None,
            lora=lora_state_dict,
            strength_model=strength_model,
            strength_clip=0.0,
        )
        if patched_model_patch is None:
            raise RuntimeError(f"Failed to apply Krea2 LoRA: {lora_file}")

        logger.info(f"[Krea2ControlNetLoraLoader] Applied successfully (strength={strength_model})")
        return (patched_model_patch,)

