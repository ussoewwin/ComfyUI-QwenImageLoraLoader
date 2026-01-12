"""
This module provides the :class:`NunchakuZImageTurboLoraStackV2` node
for applying LoRA weights to Nunchaku Z-Image-Turbo models within ComfyUI.
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
    print(f"[DEBUG] zimageturbo.py exists: {os.path.exists(os.path.join(lora_loader_dir, 'wrappers', 'zimageturbo.py'))}")

import folder_paths

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuZImageTurboLoraStackV2:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku Z-Image-Turbo model with dynamic UI.
    V3 is for official Nunchaku Z-Image loader only. For unofficial loader, use V2.
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
    TITLE = "Nunchaku Z-Image-Turbo LoRA Stack V3"
    CATEGORY = "Nunchaku"
    DESCRIPTION = "Apply multiple LoRAs to a diffusion model in a single node with dynamic UI control. V3 is for official Nunchaku Z-Image loader only. For unofficial loader, use V2."

    def load_lora_stack(self, model, lora_count, cpu_offload="disable", toggle_all=True, **kwargs):
        """
        Apply multiple LoRAs to a Nunchaku Z-Image-Turbo diffusion model.
        Uses ComfyUI standard load_lora_for_models mechanism (compatible with new ZImageModelPatcher).
        """
        # Import standard ComfyUI LoRA loader
        try:
            from comfy.sd import load_lora_for_models
            import comfy.utils
        except ImportError:
            logger.error("Failed to import standard ComfyUI LoRA loader. Please ensure ComfyUI is properly installed.")
            raise

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
            
            # Log each LoRA slot status
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
        
        # Log summary
        logger.info(f"[LoRA Stack Status] Summary: {len(loras_to_apply)} LoRA(s) will be applied out of {lora_count} slot(s)")

        if not loras_to_apply:
            return (model,)

        # Import mapping debug functions from nunchaku_code.lora_qwen
        try:
            # Use lora_loader_dir that was set at module level
            if lora_loader_dir not in sys.path:
                sys.path.insert(0, lora_loader_dir)
            
            from nunchaku_code.lora_qwen import _classify_and_map_key, _load_lora_state_dict, _detect_lora_format
        except ImportError as e:
            logger.warning(f"Failed to import mapping debug functions: {e}")
            logger.warning("Mapping debug logs will be skipped.")
            _classify_and_map_key = None
            _load_lora_state_dict = None
            _detect_lora_format = None

        # DEBUG: Inspect all keys in the first LoRA (mapping debug logs)
        if loras_to_apply and _classify_and_map_key and _load_lora_state_dict and _detect_lora_format:
            first_lora_name, first_lora_strength = loras_to_apply[0]
            try:
                first_lora_path = folder_paths.get_full_path_or_raise("loras", first_lora_name)
                first_lora_state_dict = _load_lora_state_dict(first_lora_path)
                logger.info(f"--- DEBUG: Inspecting keys for LoRA 1: '{first_lora_name}' (Strength: {first_lora_strength}) ---")
                
                # Check format first (same optimization as v3)
                _first_detection = _detect_lora_format(first_lora_state_dict)
                if _first_detection["has_standard"]:
                    # Standard format (or mixed): Log EVERYTHING as requested.
                    for key in first_lora_state_dict.keys():
                        parsed_res = _classify_and_map_key(key)
                        if parsed_res:
                            group, base_key, comp, ab = parsed_res
                            mapped_name = f"{base_key}.{comp}.{ab}" if comp and ab else (f"{base_key}.{ab}" if ab else base_key)
                            logger.info(f"Key: {key} -> Mapped to: {mapped_name} (Group: {group})")
                        else:
                            logger.warning(f"Key: {key} -> UNMATCHED (Ignored)")
                else:
                    # Unsupported format only: Skip loop to prevent freeze.
                    logger.warning(f"⚠️  Unsupported LoRA format detected (No standard keys).")
                    logger.warning(f"   Skipping detailed key inspection of {len(first_lora_state_dict)} keys to prevent console freeze.")
                    logger.warning(f"   Note: This LoRA will likely have no effect or will be skipped entirely.")
                
                logger.info("--- DEBUG: End key inspection ---")
            except Exception as e:
                logger.warning(f"Failed to inspect LoRA keys for debugging: {e}")

        # Use standard ComfyUI LoRA loading mechanism (compatible with new ZImageModelPatcher)
        # Start with the original model
        ret_model = model

        # Apply each LoRA using standard ComfyUI mechanism
        for lora_name, lora_strength in loras_to_apply:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            logger.debug(f"Loading LoRA: {lora_name} (strength={lora_strength})")
            
            # Load LoRA file
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            
            # Apply LoRA using standard ComfyUI mechanism (MODEL only, no CLIP)
            # load_lora_for_models(model, clip, lora, strength_model, strength_clip)
            ret_model, _ = load_lora_for_models(ret_model, None, lora, lora_strength, 0)
            logger.debug(f"LoRA loaded successfully: {lora_name}")

        logger.info(f"Total LoRAs applied: {len(loras_to_apply)}")

        return (ret_model,)

GENERATED_NODES = {
    "NunchakuZImageTurboLoraStackV3": NunchakuZImageTurboLoraStackV2
}

GENERATED_DISPLAY_NAMES = {
    "NunchakuZImageTurboLoraStackV3": "Nunchaku Z-Image-Turbo LoRA Stack V3"
}
