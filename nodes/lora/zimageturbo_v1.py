"""
This module provides the :class:`NunchakuZImageTurboLoraStackV1` node
for applying LoRA weights to Nunchaku Z-Image-Turbo models within ComfyUI.
rgthree-style UI with dynamic LoRA rows. Uses compose_loras_v2 for perfect mapping.
"""

import logging
import os
import sys

# Fix import path for this module
current_dir = os.path.dirname(os.path.abspath(__file__))
lora_loader_dir = os.path.dirname(os.path.dirname(current_dir))
if lora_loader_dir not in sys.path:
    sys.path.insert(0, lora_loader_dir)

import folder_paths
import comfy

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class any_type(str):
    def __ne__(self, __value: object) -> bool:
        return False


class FlexibleOptionalInputType(dict):
    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return (any_type, {})


class NunchakuZImageTurboLoraStackV1:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku Z-Image-Turbo model with dynamic UI.
    rgthree-style layout. Uses compose_loras_v2 for perfect mapping. Official Nunchaku Z-Image loader only.
    """

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model to apply LoRAs to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model (unchanged, for standard interface compatibility)."}),
                "stack_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Master Switch: Enable or disable the entire LoRA stack processing.",
                    },
                ),
            },
            "optional": FlexibleOptionalInputType(),
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model with all LoRAs applied.", "The CLIP model (unchanged).")
    FUNCTION = "load_lora_stack"
    TITLE = "Nunchaku Z-Image-Turbo LoRA Stack V1"
    CATEGORY = "Nunchaku"
    DESCRIPTION = "Apply multiple LoRAs to Z-Image-Turbo with rgthree-style dynamic UI. Uses compose_loras_v2."

    def load_lora_stack(self, model, clip, stack_enabled=True, **kwargs):
        lora_keys = [key for key in kwargs.keys() if key.startswith("lora_")]
        lora_keys.sort(key=lambda x: int(x.split("_")[-1]))

        lora_count = len(lora_keys)
        loras_to_apply = []

        logger.info(f"[LoRA Stack Status] stack_enabled: {stack_enabled}")
        logger.info(f"[LoRA Stack Status] Processing {lora_count} LoRA slot(s):")

        for i, key in enumerate(lora_keys):
            value = kwargs[key]
            if not isinstance(value, dict):
                continue

            lora_name = value.get("lora_name", "None")
            lora_strength = value.get("lora_strength", 1.0)
            enabled = stack_enabled and value.get("enabled", True)

            status_parts = [f"Slot {i}:"]
            if lora_name and lora_name != "None":
                status_parts.extend([f"'{lora_name}'", f"strength={lora_strength}"])
            else:
                status_parts.append("(no LoRA selected)")

            status_parts.extend([f"stack_enabled={stack_enabled}", f"enabled_{i}={enabled}"])

            if enabled and lora_name and lora_name != "None" and abs(lora_strength) > 1e-5:
                status_parts.append("→ APPLIED ✓")
                loras_to_apply.append((lora_name, lora_strength))
            else:
                status_parts.append("→ SKIPPED ✗")

            logger.info(f"[LoRA Stack Status] {' | '.join(status_parts)}")

        logger.info(f"[LoRA Stack Status] Summary: {len(loras_to_apply)} LoRA(s) will be applied out of {lora_count} slot(s)")

        if not loras_to_apply:
            return (model, clip)

        model_wrapper = model.model.diffusion_model
        model_wrapper_type_name = type(model_wrapper).__name__
        model_wrapper_module = type(model_wrapper).__module__

        if model_wrapper_type_name == "NextDiT" and model_wrapper_module == "comfy.ldm.lumina.model":
            transformer = model_wrapper
        elif hasattr(model_wrapper, "model"):
            transformer = model_wrapper.model
        else:
            logger.error(f"❌ Unsupported model type: {model_wrapper_type_name} from {model_wrapper_module}")
            logger.error("V1 requires NextDiT model. Falling back to standard loader.")
            from comfy.sd import load_lora_for_models
            ret_model, ret_clip = model, clip
            for lora_name, lora_strength in loras_to_apply:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                ret_model, ret_clip = load_lora_for_models(ret_model, ret_clip, lora, lora_strength, lora_strength)
            return (ret_model, ret_clip)

        try:
            if lora_loader_dir not in sys.path:
                sys.path.insert(0, lora_loader_dir)
            from nunchaku_code.lora_qwen import compose_loras_v2
        except ImportError as e:
            logger.error(f"Failed to import compose_loras_v2: {e}")
            from comfy.sd import load_lora_for_models
            ret_model, ret_clip = model, clip
            for lora_name, lora_strength in loras_to_apply:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                ret_model, ret_clip = load_lora_for_models(ret_model, ret_clip, lora, lora_strength, lora_strength)
            return (ret_model, ret_clip)

        lora_configs = []
        for lora_name, lora_strength in loras_to_apply:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            lora_configs.append((lora_path, lora_strength))

        logger.info(f"Applying {len(lora_configs)} LoRA(s) using compose_loras_v2...")
        try:
            compose_loras_v2(transformer, lora_configs, apply_awq_mod=False)
            logger.info(f"✅ Successfully applied {len(lora_configs)} LoRA(s)")
        except Exception as e:
            logger.error(f"❌ Failed to apply LoRAs: {e}")
            from comfy.sd import load_lora_for_models
            ret_model, ret_clip = model, clip
            for lora_name, lora_strength in loras_to_apply:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                ret_model, ret_clip = load_lora_for_models(ret_model, ret_clip, lora, lora_strength, lora_strength)
            return (ret_model, ret_clip)

        return (model, clip)


GENERATED_NODES = {
    "NunchakuZImageTurboLoraStackV1": NunchakuZImageTurboLoraStackV1,
}

GENERATED_DISPLAY_NAMES = {
    "NunchakuZImageTurboLoraStackV1": "Nunchaku Z-Image-Turbo LoRA Stack V1",
}
