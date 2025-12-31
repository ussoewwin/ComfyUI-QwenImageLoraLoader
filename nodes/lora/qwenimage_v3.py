"""
This module provides the :class:`NunchakuQwenImageLoraStackV3` node
for applying LoRA weights to Nunchaku Qwen Image models within ComfyUI.
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


class NunchakuQwenImageLoraStackV2:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku Qwen Image model with dynamic UI.
    V3 adds per-slot enable toggles and an all-toggle for ComfyUI Nodes 2.0 UI.
    """

    @classmethod
    def IS_CHANGED(cls, model, lora_count, cpu_offload="disable", toggle_all=True, **kwargs):
        """
        Detect changes to trigger node re-execution.
        Returns a hash of relevant parameters to detect changes.
        """
        import hashlib

        m = hashlib.sha256()
        m.update(str(model).encode())
        m.update(str(lora_count).encode())
        m.update(cpu_offload.encode())
        m.update(str(toggle_all).encode())
        # Hash all LoRA parameters
        for i in range(1, 11):
            m.update(kwargs.get(f"lora_name_{i}", "").encode())
            m.update(str(kwargs.get(f"lora_strength_{i}", 0)).encode())
            m.update(str(kwargs.get(f"enabled_{i}", True)).encode())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(s):
        loras = ["None"] + folder_paths.get_filename_list("loras")

        inputs = {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model to apply LoRAs to."},
                ),
                "lora_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of LoRA slots to process.",
                    },
                ),
                "cpu_offload": (
                    ["auto", "enable", "disable"],
                    {
                        "default": "disable",
                        "tooltip": "CPU offload setting. 'auto' enables offload when VRAM is low, 'enable' forces offload, 'disable' disables offload.",
                    },
                ),
                "toggle_all": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable/disable all LoRAs at once.",
                    },
                ),
            },
            "optional": {},
        }

        # Add all LoRA inputs (up to 10 slots) as optional
        for i in range(1, 11):
            inputs["optional"][f"enabled_{i}"] = (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": f"Enable/disable LoRA {i}.",
                },
            )
            inputs["optional"][f"lora_name_{i}"] = (
                loras,
                {"tooltip": f"The file name of LoRA {i}. Select 'None' to skip this slot."},
            )
            inputs["optional"][f"lora_strength_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": f"Strength for LoRA {i}.",
                },
            )

        return inputs

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model with all LoRAs applied.",)
    FUNCTION = "load_lora_stack"
    TITLE = "Nunchaku Qwen Image LoRA Stack V3"
    CATEGORY = "Nunchaku"
    DESCRIPTION = "Apply multiple LoRAs to a diffusion model in a single node with dynamic UI control."

    def load_lora_stack(self, model, lora_count, cpu_offload="disable", toggle_all=True, **kwargs):
        loras_to_apply = []

        # Log toggle_all state
        logger.info(f"[LoRA Stack Status] toggle_all: {toggle_all}")
        logger.info(f"[LoRA Stack Status] Processing {lora_count} LoRA slot(s):")

        # Process only the number of LoRAs specified by lora_count
        for i in range(1, lora_count + 1):
            lora_name = kwargs.get(f"lora_name_{i}")
            lora_strength = kwargs.get(f"lora_strength_{i}", 1.0)
            enabled_individual = kwargs.get(f"enabled_{i}", True)
            # Check if this LoRA is enabled (considering both toggle_all and individual enabled_<i>)
            # If toggle_all is False, still respect individual enabled_<i> settings (individual override)
            # If toggle_all is True, respect individual enabled_<i> settings
            # Fixed: Allow individual enabled_<i> to work even when toggle_all is False (Issue #42)
            enabled = enabled_individual

            status_parts = []
            status_parts.append(f"Slot {i}:")
            if lora_name and lora_name != "None":
                status_parts.append(f"'{lora_name}'")
                status_parts.append(f"strength={lora_strength}")
            else:
                status_parts.append("(no LoRA selected)")

            status_parts.append(f"toggle_all={toggle_all}")
            status_parts.append(f"enabled_{i}={enabled_individual}")
            status_parts.append(f"final_enabled={enabled}")

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
            transformer = model_wrapper.model
        elif model_wrapper_type_name == "NunchakuQwenImageTransformer2DModel" or model_wrapper_type_name.endswith(
            "NunchakuQwenImageTransformer2DModel"
        ):
            logger.info("🔧 Wrapping NunchakuQwenImageTransformer2DModel with ComfyQwenImageWrapper")
            logger.info(f"📦 Creating ComfyQwenImageWrapper with cpu_offload='{cpu_offload}'")
            wrapped_model = ComfyQwenImageWrapper(
                model_wrapper,
                getattr(model_wrapper, "config", {}),
                None,  # customized_forward
                {},  # forward_kwargs
                cpu_offload,  # cpu_offload_setting
                4.0,  # vram_margin_gb
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
    "NunchakuQwenImageLoraStackV3": NunchakuQwenImageLoraStackV2,
}

GENERATED_DISPLAY_NAMES = {
    "NunchakuQwenImageLoraStackV3": "Nunchaku Qwen Image LoRA Stack V3",
}


