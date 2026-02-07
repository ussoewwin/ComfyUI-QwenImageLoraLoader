"""
This module provides the :class:`NunchakuQwenImageLoraStackV1` node
for applying LoRA weights to Nunchaku Qwen Image models within ComfyUI.
The interface completely mimics the Power Lora Loader from rgthree-comfy(https://github.com/rgthree/rgthree-comfy ),
supporting dynamic additions and custom widgets.
"""

import copy
import logging
import os
import sys

# Fix import path for this module
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 2 levels: nodes/lora -> nodes -> ComfyUI-QwenImageLoraLoader
lora_loader_dir = os.path.dirname(os.path.dirname(current_dir))
if lora_loader_dir not in sys.path:
    sys.path.insert(0, lora_loader_dir)
    print(f"[DEBUG] Added to sys.path: {lora_loader_dir}")
    print(f"[DEBUG] wrappers dir exists: {os.path.exists(os.path.join(lora_loader_dir, 'wrappers'))}")
    print(f"[DEBUG] qwenimage.py exists: {os.path.exists(os.path.join(lora_loader_dir, 'wrappers', 'qwenimage.py'))}")

import folder_paths

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
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

class NunchakuQwenImageLoraStackV1:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku Qwen Image model with dynamic UI.
    Built upon V3, pays full homage to the clean and minimalist design of the
    Power Lora Loader from the rgthree-comfy project.
    """

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model to apply LoRAs to."},
                ),
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "disable",
                        "tooltip": "CPU offload setting. 'auto' enables offload when VRAM is low, 'enable' forces offload, 'disable' disables offload.",
                    },
                ),
                "apply_awq_mod": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable manual planar injection for AWQ modulation layers. Fixes noise issues in quantized models. Default is True.",
                    },
                ),
                "stack_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Master Switch: Enable or disable the entire LoRA stack processing.",
                    },
                ),
            },
            "optional": FlexibleOptionalInputType(),
            # Use 'hidden' here to let JS pass dynamic LoRA data
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

        return inputs

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model with all LoRAs applied.",)
    FUNCTION = "load_lora_stack"
    TITLE = "Nunchaku Qwen Image LoRA Stack V1"
    CATEGORY = "Nunchaku"
    DESCRIPTION = "Apply multiple LoRAs to a diffusion model in a single node with dynamic UI control."

    def load_lora_stack(self, model, cpu_offload="disable", apply_awq_mod=True, stack_enabled=True, **kwargs):
        # Dynamic widgets passed from JS will appear in kwargs
        # Named lora_1, lora_2... with values as dictionaries: {'enabled': bool, 'lora_name': str, 'lora_strength': float}
        lora_keys = [key for key in kwargs.keys() if key.startswith("lora_")]
        lora_keys.sort(key=lambda x: int(x.split('_')[-1])) # Ensure numerical sorting

        lora_count = len(lora_keys)

        loras_to_apply = []

        # Log stack_enabled state
        logger.info(f"[LoRA Stack Status] apply_awq_mod: {apply_awq_mod}")
        logger.info(f"[LoRA Stack Status] stack_enabled: {stack_enabled}")
        logger.info(f"[LoRA Stack Status] Processing {lora_count} LoRA slot(s):")

        for i, key in enumerate(lora_keys):
            value = kwargs[key]
            if not isinstance(value, dict):
                continue

            lora_name = value.get("lora_name", "None")
            lora_strength = value.get("lora_strength", 1.0)
            # stack_enabled acts as a Master Switch.
            # If stack_enabled is False, all LoRAs are disabled.
            # If stack_enabled is True, we respect the individual 'enabled' state.
            enabled = stack_enabled and value.get("enabled", True)

            status_parts = []
            status_parts.append(f"Slot {i}:")
            if lora_name and lora_name != "None":
                status_parts.append(f"'{lora_name}'")
                status_parts.append(f"strength={lora_strength}")
            else:
                status_parts.append("(no LoRA selected)")

            status_parts.append(f"stack_enabled={stack_enabled}")
            status_parts.append(f"enabled_{i}={enabled}")

            if enabled and lora_name and lora_name != "None" and abs(lora_strength) > 1e-5:
                status_parts.append("→ APPLIED ✓")
                loras_to_apply.append((lora_name, lora_strength))
            else:
                status_parts.append("→ SKIPPED ✗")

            logger.info(f"[LoRA Stack Status] {' | '.join(status_parts)}")

        logger.info(f"[LoRA Stack Status] Summary: {len(loras_to_apply)} LoRA(s) will be applied out of {lora_count} slot(s)")

        if not loras_to_apply:
            return (model,)

        model_wrapper = model.model.diffusion_model

        # Dynamic import with explicit path manipulation
        import importlib.util

        lora_loader_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if lora_loader_dir not in sys.path:
            sys.path.insert(0, lora_loader_dir)

        spec = importlib.util.spec_from_file_location(
            "wrappers.qwenimage",
            os.path.join(lora_loader_dir, "wrappers", "qwenimage.py"),
        )
        wrappers_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrappers_module)
        ComfyQwenImageWrapper = wrappers_module.ComfyQwenImageWrapper

        from nunchaku import NunchakuQwenImageTransformer2DModel

        # Debug logging
        model_wrapper_type_name = type(model_wrapper).__name__
        model_wrapper_module = type(model_wrapper).__module__
        logger.info(f"🔍 Model wrapper type: '{model_wrapper_type_name}'")
        logger.info(f"🔍 Model wrapper module: {model_wrapper_module}")
        logger.info(f"🔍 Type repr: {repr(type(model_wrapper))}")
        logger.info(f"🔍 Has 'model' attr? {hasattr(model_wrapper, 'model')}")
        logger.info(f"🔍 Has 'loras' attr? {hasattr(model_wrapper, 'loras')}")

        # Check if it's already wrapped
        if hasattr(model_wrapper, "model") and hasattr(model_wrapper, "loras"):
            logger.info("✅ Model is already wrapped (detected via attributes)")
            logger.info(f"📦 Current CPU offload setting: '{model_wrapper.cpu_offload_setting}'")
            if model_wrapper.cpu_offload_setting != cpu_offload:
                logger.info(f"🔄 Updating CPU offload setting from '{model_wrapper.cpu_offload_setting}' to '{cpu_offload}'")
                model_wrapper.cpu_offload_setting = cpu_offload

            # Dynamically update the apply_awq_mod setting.
            if hasattr(model_wrapper, "apply_awq_mod") and model_wrapper.apply_awq_mod != apply_awq_mod:
                logger.info(f"🔄 Updating AWQ mod setting from '{model_wrapper.apply_awq_mod}' to '{apply_awq_mod}'")
                model_wrapper.apply_awq_mod = apply_awq_mod

            transformer = model_wrapper.model
        elif model_wrapper_type_name == "NunchakuQwenImageTransformer2DModel" or model_wrapper_type_name.endswith(
            "NunchakuQwenImageTransformer2DModel"
        ):
            logger.info("🔧 Wrapping NunchakuQwenImageTransformer2DModel with ComfyQwenImageWrapper")
            logger.info(f"📦 Creating ComfyQwenImageWrapper with cpu_offload='{cpu_offload}', apply_awq_mod={apply_awq_mod}")
            wrapped_model = ComfyQwenImageWrapper(
                model_wrapper,
                getattr(model_wrapper, "config", {}),
                None,  # customized_forward
                {},  # forward_kwargs
                cpu_offload,  # cpu_offload_setting
                4.0,  # vram_margin_gb
                apply_awq_mod=apply_awq_mod,
            )
            model.model.diffusion_model = wrapped_model
            model_wrapper = wrapped_model
            transformer = model_wrapper.model
        else:
            logger.error(f"❌ Model type mismatch! Type: {model_wrapper_type_name}, Module: {model_wrapper_module}")
            logger.error("Please use 'Nunchaku Qwen Image DiT Loader'.")
            raise TypeError(f"This LoRA loader only works with Nunchaku Qwen Image models, but got {model_wrapper_type_name}.")

        # Flux-style deepcopy
        saved_config = None
        if hasattr(model, "model") and hasattr(model.model, "model_config"):
            saved_config = model.model.model_config
            model.model.model_config = None

        model_wrapper.model = None
        try:
            ret_model = copy.deepcopy(model)
        finally:
            if saved_config is not None:
                model.model.model_config = saved_config
            model_wrapper.model = transformer

        ret_model_wrapper = ret_model.model.diffusion_model
        if saved_config is not None:
            ret_model.model.model_config = saved_config
        ret_model_wrapper.model = transformer

        ret_model_wrapper.loras = model_wrapper.loras.copy()

        for lora_name, lora_strength in loras_to_apply:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            ret_model_wrapper.loras.append((lora_path, lora_strength))
            logger.debug(f"LoRA added to stack: {lora_name} (strength={lora_strength})")

        logger.info(f"Total LoRAs in stack: {len(ret_model_wrapper.loras)}")
        return (ret_model,)


GENERATED_NODES = {
    "NunchakuQwenImageLoraStackV1": NunchakuQwenImageLoraStackV1,
}

GENERATED_DISPLAY_NAMES = {
    "NunchakuQwenImageLoraStackV1": "Nunchaku Qwen Image LoRA Stack V1",
}
