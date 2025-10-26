"""
This module provides the :class:`NunchakuQwenImageLoraLoader` node
for applying LoRA weights to Nunchaku Qwen Image models within ComfyUI.
"""

import copy
import logging
import os

import folder_paths

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuQwenImageLoraLoader:
    """
    Node for loading and applying a LoRA to a Nunchaku Qwen Image model.
    """
    @classmethod
    def IS_CHANGED(s, model, lora_name, lora_strength, *args, **kwargs):
        """
        Detect changes to trigger node re-execution.
        Returns a hash of relevant parameters to detect changes.
        """
        import hashlib
        m = hashlib.sha256()
        m.update(lora_name.encode())
        m.update(str(lora_strength).encode())
        m.update(str(model).encode())
        return m.digest().hex()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRA will be applied to. "
                        "Make sure the model is loaded by `Nunchaku Qwen Image DiT Loader`."
                    },
                ),
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "The file name of the LoRA."},
                ),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "Nunchaku Qwen Image LoRA Loader"
    CATEGORY = "Nunchaku"
    DESCRIPTION = "LoRAs are used to modify the diffusion model, altering the way in which latents are denoised."

    def load_lora(self, model, lora_name: str, lora_strength: float):
        if abs(lora_strength) < 1e-5:
            return (model,)

        model_wrapper = model.model.diffusion_model

        from ...wrappers.qwenimage import ComfyQwenImageWrapper
        from nunchaku import NunchakuQwenImageTransformer2DModel
        
        # Debug logging
        logger.info(f"🔍 Model wrapper type: {type(model_wrapper).__name__}")
        logger.info(f"🔍 Model wrapper module: {type(model_wrapper).__module__}")
        logger.info(f"🔍 Is ComfyQwenImageWrapper? {isinstance(model_wrapper, ComfyQwenImageWrapper)}")
        logger.info(f"🔍 Is NunchakuQwenImageTransformer2DModel? {isinstance(model_wrapper, NunchakuQwenImageTransformer2DModel)}")
        
        # Check if it's already wrapped
        if isinstance(model_wrapper, ComfyQwenImageWrapper):
            # Already wrapped, proceed normally
            transformer = model_wrapper.model
        elif isinstance(model_wrapper, NunchakuQwenImageTransformer2DModel) or model_wrapper.__class__.__name__ == "NunchakuQwenImageTransformer2DModel":
            # Not wrapped yet, need to wrap it first
            logger.info("🔧 Wrapping NunchakuQwenImageTransformer2DModel with ComfyQwenImageWrapper")
            
            # Create wrapper
            wrapped_model = ComfyQwenImageWrapper(
                model_wrapper,
                getattr(model_wrapper, 'config', {}),
                None,  # customized_forward
                {},    # forward_kwargs
                "auto", # cpu_offload_setting
                4.0,   # vram_margin_gb
            )
            
            # Replace the model's diffusion_model with our wrapper
            model.model.diffusion_model = wrapped_model
            model_wrapper = wrapped_model
            transformer = model_wrapper.model
        else:
            logger.error("❌ Model type mismatch! Please use 'Nunchaku Qwen Image DiT Loader'.")
            raise TypeError(f"This LoRA loader only works with Nunchaku Qwen Image models, but got {type(model_wrapper).__name__}.")

        # Flux-style deepcopy
        model_wrapper.model = None
        ret_model = copy.deepcopy(model)
        ret_model_wrapper = ret_model.model.diffusion_model
        model_wrapper.model = transformer
        ret_model_wrapper.model = transformer

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        ret_model_wrapper.loras.append((lora_path, lora_strength))

        logger.info(f"LoRA added: {lora_name} (strength={lora_strength})")
        logger.debug(f"Total LoRAs: {len(ret_model_wrapper.loras)}")

        return (ret_model,)


class NunchakuQwenImageLoraStack:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku Qwen Image model with dynamic UI.
    """
    @classmethod
    def IS_CHANGED(cls, model, lora_count, **kwargs):
        """
        Detect changes to trigger node re-execution.
        Returns a hash of relevant parameters to detect changes.
        """
        import hashlib
        m = hashlib.sha256()
        m.update(str(model).encode())
        m.update(str(lora_count).encode())
        # Hash all LoRA parameters
        for i in range(1, 11):
            m.update(kwargs.get(f"lora_name_{i}", "").encode())
            m.update(str(kwargs.get(f"lora_strength_{i}", 0)).encode())
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
            },
        }

        # Add all LoRA inputs (up to 10 slots)
        for i in range(1, 11):
            inputs["required"][f"lora_name_{i}"] = (
                loras,
                {"tooltip": f"The file name of LoRA {i}. Select 'None' to skip this slot."},
            )
            inputs["required"][f"lora_strength_{i}"] = (
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
    TITLE = "Nunchaku Qwen Image LoRA Stack"
    CATEGORY = "Nunchaku"
    DESCRIPTION = "Apply multiple LoRAs to a diffusion model in a single node with dynamic UI control. v1.0.3"

    def load_lora_stack(self, model, lora_count, **kwargs):
        loras_to_apply = []
        
        # Process only the number of LoRAs specified by lora_count
        for i in range(1, lora_count + 1):
            lora_name = kwargs.get(f"lora_name_{i}")
            lora_strength = kwargs.get(f"lora_strength_{i}", 1.0)
            
            # Skip if lora_name is None or strength is negligible
            if lora_name and lora_name != "None" and abs(lora_strength) > 1e-5:
                loras_to_apply.append((lora_name, lora_strength))

        if not loras_to_apply:
            return (model,)

        model_wrapper = model.model.diffusion_model

        from ...wrappers.qwenimage import ComfyQwenImageWrapper
        from nunchaku import NunchakuQwenImageTransformer2DModel
        
        # Debug logging
        logger.info(f"🔍 Model wrapper type: {type(model_wrapper).__name__}")
        logger.info(f"🔍 Model wrapper module: {type(model_wrapper).__module__}")
        logger.info(f"🔍 Is ComfyQwenImageWrapper? {isinstance(model_wrapper, ComfyQwenImageWrapper)}")
        logger.info(f"🔍 Is NunchakuQwenImageTransformer2DModel? {isinstance(model_wrapper, NunchakuQwenImageTransformer2DModel)}")
        
        # Check if it's already wrapped
        if isinstance(model_wrapper, ComfyQwenImageWrapper):
            # Already wrapped, proceed normally
            transformer = model_wrapper.model
        elif isinstance(model_wrapper, NunchakuQwenImageTransformer2DModel) or model_wrapper.__class__.__name__ == "NunchakuQwenImageTransformer2DModel":
            # Not wrapped yet, need to wrap it first
            logger.info("🔧 Wrapping NunchakuQwenImageTransformer2DModel with ComfyQwenImageWrapper")
            
            # Create wrapper
            wrapped_model = ComfyQwenImageWrapper(
                model_wrapper,
                getattr(model_wrapper, 'config', {}),
                None,  # customized_forward
                {},    # forward_kwargs
                "auto", # cpu_offload_setting
                4.0,   # vram_margin_gb
            )
            
            # Replace the model's diffusion_model with our wrapper
            model.model.diffusion_model = wrapped_model
            model_wrapper = wrapped_model
            transformer = model_wrapper.model
        else:
            logger.error("❌ Model type mismatch! Please use 'Nunchaku Qwen Image DiT Loader'.")
            raise TypeError(f"This LoRA loader only works with Nunchaku Qwen Image models, but got {type(model_wrapper).__name__}.")

        # Flux-style deepcopy
        model_wrapper.model = None
        ret_model = copy.deepcopy(model)
        ret_model_wrapper = ret_model.model.diffusion_model
        model_wrapper.model = transformer
        ret_model_wrapper.model = transformer

        ret_model_wrapper.loras = model_wrapper.loras.copy()

        for lora_name, lora_strength in loras_to_apply:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            ret_model_wrapper.loras.append((lora_path, lora_strength))
            logger.debug(f"LoRA added to stack: {lora_name} (strength={lora_strength})")

        logger.info(f"Total LoRAs in stack: {len(ret_model_wrapper.loras)}")

        return (ret_model,)
