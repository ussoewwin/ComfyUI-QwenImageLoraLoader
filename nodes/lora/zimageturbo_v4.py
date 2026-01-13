"""
This module provides the :class:`NunchakuZImageTurboLoraStackV4` node
for applying LoRA weights to Nunchaku Z-Image-Turbo models within ComfyUI.
V4 uses compose_loras_v2 for perfect mapping (same as v3).
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

import folder_paths

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuZImageTurboLoraStackV4:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku Z-Image-Turbo model with dynamic UI.
    V4 uses compose_loras_v2 for perfect mapping (same as v3).
    V4 is for official Nunchaku Z-Image loader only.
    """
    @classmethod
    def IS_CHANGED(cls, model, clip, lora_count, toggle_all=True, **kwargs):
        """
        Detect changes to trigger node re-execution.
        Returns a hash of relevant parameters to detect changes.
        """
        import hashlib
        m = hashlib.sha256()
        m.update(str(model).encode())
        m.update(str(clip).encode())
        m.update(str(lora_count).encode())
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
                "clip": (
                    "CLIP",
                    {"tooltip": "The CLIP model to apply LoRAs to."},
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

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model with all LoRAs applied.", "The modified CLIP model (unchanged).")
    FUNCTION = "load_lora_stack"
    TITLE = "Nunchaku Z-Image-Turbo LoRA Stack V4"
    CATEGORY = "Nunchaku"
    DESCRIPTION = "Apply multiple LoRAs to a diffusion model in a single node with dynamic UI control. V4 uses compose_loras_v2 for perfect mapping (same as v3). V4 is for official Nunchaku Z-Image loader only."

    def load_lora_stack(self, model, clip, lora_count, toggle_all=True, **kwargs):
        """
        Apply multiple LoRAs to a Nunchaku Z-Image-Turbo diffusion model using compose_loras_v2.
        This uses the same perfect mapping mechanism as v3.
        CLIP is returned unchanged (for compatibility with standard ComfyUI LoRA loader interface).
        """
        loras_to_apply = []
        
        # Log toggle_all state
        logger.info(f"[LoRA Stack Status] toggle_all: {toggle_all}")
        logger.info(f"[LoRA Stack Status] Processing {lora_count} LoRA slot(s):")
        
        # Process only the number of LoRAs specified by lora_count
        for i in range(1, lora_count + 1):
            lora_name = kwargs.get(f"lora_name_{i}")
            lora_strength = kwargs.get(f"lora_strength_{i}", 1.0)
            enabled_individual = kwargs.get(f"enabled_{i}", True)
            # Check if this LoRA is enabled
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
            return (model, clip)

        # Import mapping debug functions from nunchaku_code.lora_qwen
        try:
            # Use lora_loader_dir that was set at module level
            if lora_loader_dir not in sys.path:
                sys.path.insert(0, lora_loader_dir)
            
            from nunchaku_code.lora_qwen import _classify_and_map_key, _load_lora_state_dict, _detect_lora_format, _log_lora_format_detection
        except ImportError as e:
            logger.warning(f"Failed to import mapping debug functions: {e}")
            logger.warning("Mapping debug logs will be skipped.")
            _classify_and_map_key = None
            _load_lora_state_dict = None
            _detect_lora_format = None
            _log_lora_format_detection = None

        # DEBUG: Inspect all keys in all LoRAs (same as v3 compose_loras_v2)
        # Log format detection and key mapping for all LoRAs with complete statistics
        if loras_to_apply and _classify_and_map_key and _load_lora_state_dict and _detect_lora_format and _log_lora_format_detection:
            from collections import defaultdict
            
            logger.info(f"Composing {len(loras_to_apply)} LoRAs...")
            
            # Cache first LoRA state dict for reuse (same optimization as v3)
            _cached_first_lora_state_dict = None
            
            for idx, (lora_name, lora_strength) in enumerate(loras_to_apply):
                try:
                    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                    
                    # Use cached state dict for first LoRA to avoid duplicate I/O
                    if idx == 0 and _cached_first_lora_state_dict is not None:
                        lora_state_dict = _cached_first_lora_state_dict
                    else:
                        lora_state_dict = _load_lora_state_dict(lora_path)
                        if idx == 0:
                            _cached_first_lora_state_dict = lora_state_dict
                    
                    # LoRA format detection + detailed logging (same as v3 compose_loras_v2)
                    try:
                        detection = _detect_lora_format(lora_state_dict)
                        _log_lora_format_detection(str(lora_name), detection)
                    except Exception:
                        # Safety: never fail due to logging
                        pass
                    
                    # First LoRA: Detailed key inspection (same as v3 compose_loras_v2)
                    if idx == 0:
                        logger.info(f"--- DEBUG: Inspecting keys for LoRA 1 (Strength: {lora_strength}) ---")
                        
                        # Check format first (same optimization as v3)
                        _first_detection = _detect_lora_format(lora_state_dict)
                        if _first_detection["has_standard"]:
                            # Standard format (or mixed): Log EVERYTHING as requested.
                            for key in lora_state_dict.keys():
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
                            logger.warning(f"   Skipping detailed key inspection of {len(lora_state_dict)} keys to prevent console freeze.")
                            logger.warning(f"   Note: This LoRA will likely have no effect or will be skipped entirely.")
                        
                        logger.info("--- DEBUG: End key inspection ---")
                    
                    # Statistics for all LoRAs (same as v3 compose_loras_v2)
                    lora_grouped: dict = defaultdict(dict)
                    lokr_keys_count = 0
                    standard_keys_count = 0
                    qkv_lokr_keys_count = 0
                    unrecognized_keys = []
                    total_keys = len(lora_state_dict)
                    
                    for key, value in lora_state_dict.items():
                        parsed = _classify_and_map_key(key)
                        if parsed is None:
                            unrecognized_keys.append(key)
                            continue
                        
                        group, base_key, comp, ab = parsed
                        if ab in ("lokr_w1", "lokr_w2"):
                            lokr_keys_count += 1
                            # Check if it's QKV format LoKR
                            if group in ("qkv", "add_qkv") and comp is not None:
                                qkv_lokr_keys_count += 1
                        elif ab in ("A", "B"):
                            standard_keys_count += 1
                        
                        if group in ("qkv", "add_qkv", "glu") and comp is not None:
                            # Handle both standard LoRA (A/B) and LoKR (lokr_w1/lokr_w2) formats
                            if ab in ("lokr_w1", "lokr_w2"):
                                lora_grouped[base_key][f"{comp}_{ab}"] = value
                            else:
                                lora_grouped[base_key][f"{comp}_{ab}"] = value
                        else:
                            lora_grouped[base_key][ab] = value
                    
                    # Format classification (same as v3 compose_loras_v2)
                    has_lokr = lokr_keys_count > 0
                    has_standard = standard_keys_count > 0
                    if has_lokr and has_standard:
                        lora_format = "Mixed (LoKR + Standard LoRA)"
                    elif has_lokr:
                        lora_format = "LoKR (QKV format)" if qkv_lokr_keys_count > 0 else "LoKR"
                    elif has_standard:
                        lora_format = "Standard LoRA"
                    else:
                        lora_format = "Unknown/Unsupported"
                    
                    # Process grouped weights for statistics (same as v3 compose_loras_v2)
                    # Note: We don't actually process the weights in v4 (standard loader does that),
                    # but we calculate statistics to match v3's debug output
                    processed_groups = {}
                    special_handled = set()
                    for base_key, lw in lora_grouped.items():
                        if base_key in special_handled:
                            continue
                        
                        # Check if this is LoKR format (lokr_w1, lokr_w2)
                        if "lokr_w1" in lw or "lokr_w2" in lw:
                            continue  # Skip LoKR (not processed)
                        
                        # For statistics only, check if we have A and B (standard LoRA)
                        if "A" in lw and "B" in lw:
                            processed_groups[base_key] = True
                    
                    # Statistics logging (same as v3 compose_loras_v2)
                    logger.info(f"[LoRA {idx + 1} Statistics] '{lora_name}' (Strength: {lora_strength})")
                    logger.info(f"  Total keys: {total_keys}")
                    logger.info(f"  Standard keys: {standard_keys_count}")
                    logger.info(f"  LoKR keys: {lokr_keys_count}")
                    logger.info(f"  QKV LoKR keys: {qkv_lokr_keys_count}")
                    logger.info(f"  Unrecognized keys: {len(unrecognized_keys)}")
                    logger.info(f"  Format: {lora_format}")
                    logger.info(f"  Grouped base keys: {len(lora_grouped)}")
                    logger.debug(f"  Processed module groups: {len(processed_groups)}")
                    
                    if unrecognized_keys:
                        logger.warning(f"  Unrecognized keys (first 10): {unrecognized_keys[:10]}")
                        if len(unrecognized_keys) > 10:
                            logger.warning(f"  ... and {len(unrecognized_keys) - 10} more unrecognized keys")
                    
                    # Warn if no weights would be processed (same as v3 compose_loras_v2)
                    if not processed_groups:
                        if lora_format == "Unknown/Unsupported":
                            logger.error(f"❌ {lora_name}: No weights were processed - LoRA format is unsupported and will be skipped!")
                        else:
                            logger.warning(f"⚠️  {lora_name}: No weights were processed - this LoRA will have no effect!")
                            if lora_grouped:
                                logger.warning(f"   Debug: {len(lora_grouped)} base keys were grouped but none were processed:")
                                for bk, lw in list(lora_grouped.items())[:10]:
                                    keys_in_group = list(lw.keys())
                                    logger.warning(f"     - {bk}: keys={keys_in_group}")
                                if len(lora_grouped) > 10:
                                    logger.warning(f"     ... and {len(lora_grouped) - 10} more grouped keys")
                
                except Exception as e:
                    logger.warning(f"Failed to inspect LoRA {idx + 1} ({lora_name}) keys for debugging: {e}")

        # Apply LoRAs using compose_loras_v2 for perfect mapping (same as v3)
        # Get the underlying NextDiT model from the model patcher
        model_wrapper = model.model.diffusion_model
        
        # Import compose_loras_v2 from nunchaku_code.lora_qwen
        try:
            if lora_loader_dir not in sys.path:
                sys.path.insert(0, lora_loader_dir)
            
            from nunchaku_code.lora_qwen import compose_loras_v2
        except ImportError as e:
            logger.error(f"Failed to import compose_loras_v2: {e}")
            logger.error("Cannot apply LoRAs - falling back to standard loader")
            # Fallback to standard loader if compose_loras_v2 is not available
            from comfy.sd import load_lora_for_models
            ret_model = model
            ret_clip = clip
            for lora_name, lora_strength in loras_to_apply:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                ret_model, ret_clip = load_lora_for_models(ret_model, ret_clip, lora, lora_strength, lora_strength)
            return (ret_model, ret_clip)
        
        # Check if model_wrapper is NextDiT (official Nunchaku loader)
        model_wrapper_type_name = type(model_wrapper).__name__
        model_wrapper_module = type(model_wrapper).__module__
        
        # Get the actual NextDiT model
        if model_wrapper_type_name == "NextDiT" and model_wrapper_module == "comfy.ldm.lumina.model":
            # Official loader: model_wrapper is NextDiT directly
            transformer = model_wrapper
        elif hasattr(model_wrapper, 'model'):
            # Wrapped model (e.g., ComfyZImageTurboWrapper): get underlying model
            transformer = model_wrapper.model
        else:
            logger.error(f"❌ Unsupported model type: {model_wrapper_type_name} from {model_wrapper_module}")
            logger.error("V4 requires NextDiT model. Falling back to standard loader.")
            # Fallback to standard loader
            from comfy.sd import load_lora_for_models
            ret_model = model
            ret_clip = clip
            for lora_name, lora_strength in loras_to_apply:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                ret_model, ret_clip = load_lora_for_models(ret_model, ret_clip, lora, lora_strength, lora_strength)
            return (ret_model, ret_clip)
        
        # Prepare LoRA configs for compose_loras_v2
        lora_configs = []
        for lora_name, lora_strength in loras_to_apply:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            lora_configs.append((lora_path, lora_strength))
        
        # Apply LoRAs using compose_loras_v2 (perfect mapping)
        logger.info(f"Applying {len(lora_configs)} LoRA(s) using compose_loras_v2...")
        try:
            compose_loras_v2(transformer, lora_configs)
            logger.info(f"✅ Successfully applied {len(lora_configs)} LoRA(s) using compose_loras_v2")
        except Exception as e:
            logger.error(f"❌ Failed to apply LoRAs using compose_loras_v2: {e}")
            logger.error("Falling back to standard loader.")
            # Fallback to standard loader
            from comfy.sd import load_lora_for_models
            ret_model = model
            ret_clip = clip
            for lora_name, lora_strength in loras_to_apply:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                ret_model, ret_clip = load_lora_for_models(ret_model, ret_clip, lora, lora_strength, lora_strength)
            return (ret_model, ret_clip)

        # Return model and clip (CLIP unchanged, for compatibility with standard ComfyUI LoRA loader interface)
        return (model, clip)

GENERATED_NODES = {
    "NunchakuZImageTurboLoraStackV4": NunchakuZImageTurboLoraStackV4
}

GENERATED_DISPLAY_NAMES = {
    "NunchakuZImageTurboLoraStackV4": "Nunchaku Z-Image-Turbo LoRA Stack V4"
}