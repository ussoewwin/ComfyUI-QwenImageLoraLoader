from typing import Callable, List, Tuple, Union
from pathlib import Path

import torch
from torch import nn
import comfy.model_management
import logging

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.caching.fbcache import cache_context, create_cache_context
from nunchaku_code.lora_qwen import compose_loras_v2, reset_lora_v2

logger = logging.getLogger(__name__)


class ComfyQwenImageWrapper(nn.Module):
    """
    Wrapper for NunchakuQwenImageTransformer2DModel to support ComfyUI workflows.

    This wrapper separates LoRA composition from the forward pass for maximum efficiency.
    It detects changes to its `loras` attribute and recomposes the underlying model
    lazily when the forward pass is executed.
    """

    def __init__(
            self,
            model: NunchakuQwenImageTransformer2DModel,
            config,
            customized_forward: Callable = None,
            forward_kwargs: dict | None = None,
            cpu_offload_setting: str = "auto",
            vram_margin_gb: float = 4.0
    ):
        super().__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config
        # This list is the authoritative state, modified by LoRA loader nodes
        self.loras: List[Tuple[Union[str, Path, dict], float]] = []
        # This tracks the LoRAs currently composed into the model to detect changes
        self._applied_loras: List[Tuple[Union[str, Path, dict], float]] = None

        self.cpu_offload_setting = cpu_offload_setting
        self.vram_margin_gb = vram_margin_gb

        # Log CPU offload setting on initialization
        logger.info(f"🔧 CPU offload setting: '{cpu_offload_setting}' (VRAM margin: {vram_margin_gb}GB)")

        self.customized_forward = customized_forward
        self.forward_kwargs = forward_kwargs or {}

        self._prev_timestep = None
        self._cache_context = None

        # Reusable tensor caches keyed by (H, W, device, dtype, index, offsets)
        self._img_ids_cache = {}
        # Cache for txt ids keyed by (batch, seq_len, device, dtype)
        self._txt_ids_cache = {}
        # Base linspace caches keyed by (length, device, dtype)
        self._linspace_cache_h = {}
        self._linspace_cache_w = {}

        # Track last seen device to detect CPU/GPU moves that require re-compose
        self._last_device = None

    def to_safely(self, device):
        """Safely move the model to the specified device."""
        if hasattr(self.model, "to_safely"):
            self.model.to_safely(device)
        else:
            self.model.to(device)
        return self

    def forward(
            self,
            x,
            timestep,
            context=None,
            y=None,
            guidance=None,
            control=None,
            transformer_options={},
            **kwargs,
    ):
        """
        Forward pass for the wrapped model.

        Detects changes to the `self.loras` list and recomposes the model
        on-the-fly before inference.
        """
        # Remove guidance, transformer_options, and attention_mask from kwargs
        # These may be added by ComfyUI-EulerDiscreteScheduler or other patches
        # Even though guidance is a parameter of forward(), kwargs may also contain it
        # which could cause issues when passing to _execute_model
        # Note: ref_latents is NOT a parameter of forward(), so we keep it in kwargs for _execute_model to use
        if "guidance" in kwargs:
            # If guidance is in kwargs, use the explicit parameter value (which takes precedence)
            # and remove it from kwargs to avoid duplication
            kwargs.pop("guidance")
        if "transformer_options" in kwargs:
            # Merge transformer_options from kwargs into the parameter if needed
            if isinstance(transformer_options, dict) and isinstance(kwargs["transformer_options"], dict):
                transformer_options = {**transformer_options, **kwargs.pop("transformer_options")}
            else:
                kwargs.pop("transformer_options")
        if "attention_mask" in kwargs:
            kwargs.pop("attention_mask")
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            timestep_float = float(timestep)


        model_is_dirty = (
            not self.loras and # We expect no LoRA
            hasattr(self.model, "_lora_slots") and self.model._lora_slots # But the model actually has LoRA
        )
        
        # Deep comparison of LoRA stacks to detect any changes
        # This ensures we catch changes in weights, paths, or order
        loras_changed = False
        if self._applied_loras is None or len(self._applied_loras) != len(self.loras):
            loras_changed = True
        else:
            for applied, current in zip(self._applied_loras, self.loras):
                if applied != current:
                    loras_changed = True
                    break

        # Detect device transition (e.g., CPU offload on/off) and force re-compose
        try:
            current_device = next(self.model.parameters()).device
        except Exception:
            current_device = None
        device_changed = (self._last_device != current_device)
        
        # Check if the LoRA stack has been changed by a loader node
        if loras_changed or model_is_dirty or device_changed:
            # The compose function handles resetting before applying the new stack
            reset_lora_v2(self.model)
            self._applied_loras = self.loras.copy()
            
            # Reset cache when LoRAs change to prevent stale cache in multi-stage workflows
            # This ensures that when switching between different LoRA sets in different stages,
            # the cache is invalidated and recreated with the new LoRA composition
            if loras_changed:
                self._cache_context = None
                self._prev_timestep = None
                logger.debug("Cache reset due to LoRA change")

            # --- NEW DYNAMIC VRAM CHECK (conditionally applied) ---

            # 1. Check if offload is *already* enabled (from loader setting "enable" or "auto" on low-vram)
            offload_is_on = hasattr(self.model, "offload_manager") and self.model.offload_manager is not None

            # 2. Decide if we *need* to turn it on
            should_enable_offload = offload_is_on

            # 3. Only run the dynamic VRAM check if:
            #    - The user's original setting was "auto"
            #    - Offloading is not *already* on
            #    - We are actually loading new LoRAs
            if self.cpu_offload_setting == "auto" and not offload_is_on and self.loras:
                try:
                    # Use the VRAM margin from the loader node
                    free_vram_gb = comfy.model_management.get_free_memory() / (1024 ** 3)

                    if free_vram_gb < self.vram_margin_gb:
                        logger.info(
                            f"Free VRAM is {free_vram_gb:.2f}GB (below safety margin of {self.vram_margin_gb}GB) and 'cpu_offload' is 'auto'. Force-enabling CPU offload for LoRA composition.")
                        should_enable_offload = True
                    else:
                        logger.info(
                            f"Free VRAM is {free_vram_gb:.2f}GB (>= {self.vram_margin_gb}GB margin). LoRAs will be composed without enabling CPU offload.")

                except Exception as e:
                    logger.error(f"Error during VRAM check for LoRA offloading: {e}. Offload will not be enabled.")
            elif self.cpu_offload_setting == "disable" and not offload_is_on:
                logger.debug("CPU offload is 'disable' and not on. Skipping VRAM check.")
            elif self.cpu_offload_setting == "enable" and offload_is_on:
                logger.debug("CPU offload is 'enable'. Will rebuild offload manager for LoRAs.")

            # --- END NEW VRAM CHECK ---

            # 4. Compose LoRAs. This changes internal tensor shapes.
            compose_loras_v2(self.model, self.loras)

            # Validate composition result; if 0 targets after a crash/transition, retry once
            try:
                has_slots = hasattr(self.model, "_lora_slots") and bool(self.model._lora_slots)
            except Exception:
                has_slots = True
            if self.loras and not has_slots:
                logger.warning("LoRA composition reported 0 target modules. Forcing reset and one retry.")
                try:
                    reset_lora_v2(self.model)
                    compose_loras_v2(self.model, self.loras)
                except Exception as e:
                    logger.error(f"LoRA re-compose retry failed: {e}")

            # 5. Re-build offload manager if it's supposed to be on
            # This block now runs if offload was on *or* if our new check decided to turn it on.
            if should_enable_offload:

                # Store settings if it was already on, otherwise use defaults
                if offload_is_on:
                    manager = self.model.offload_manager
                    offload_settings = {
                        "num_blocks_on_gpu": manager.num_blocks_on_gpu,
                        "use_pin_memory": manager.use_pin_memory,
                    }
                else:
                    # Not previously on, so use defaults from nodes/models/qwenimage.py
                    offload_settings = {
                        "num_blocks_on_gpu": 1,
                        "use_pin_memory": False,  # 'disable' maps to False
                    }
                    logger.info("Building new CPU offload manager due to LoRA VRAM check.")

                # Step 1: Completely disable and clear any old offloader (safe to call even if off)
                self.model.set_offload(False)

                # Step 2: Re-enable offloading with the correct settings
                # This builds the manager based on the *newly composed* tensor shapes.
                self.model.set_offload(True, **offload_settings)

            # --- END MODIFIED SECTION ---

            # Update last known device signature after (re)composition
            self._last_device = current_device

        # Caching logic
        use_caching = getattr(self.model, "residual_diff_threshold_multi", 0) != 0 or getattr(self.model, "_is_cached",
                                                                                              False)
        if use_caching:
            cache_invalid = self._prev_timestep is None or self._prev_timestep < timestep_float + 1e-5
            if cache_invalid:
                self._cache_context = create_cache_context()
            self._prev_timestep = timestep_float

            with cache_context(self._cache_context):
                out = self._execute_model(x, timestep, context, guidance, control, transformer_options, **kwargs)
        else:
            out = self._execute_model(x, timestep, context, guidance, control, transformer_options, **kwargs)

        if isinstance(out, tuple):
            out = out[0]

        if x.ndim == 5 and out.ndim == 4:
            out = out.unsqueeze(2)

        return out

    def _execute_model(self, x, timestep, context, guidance, control, transformer_options, **kwargs):
        """Helper function to run the model's forward pass."""
        # Get ref_latents from kwargs before removing it (it's not a parameter of forward())
        ref_latents_value = kwargs.pop("ref_latents", None)
        
        # Double-check: Remove guidance, transformer_options, and attention_mask from kwargs
        # This is a defensive measure in case any code path adds these to kwargs after forward() method
        # Even though forward() already removes them, this ensures _execute_model is always safe
        kwargs.pop("guidance", None)
        kwargs.pop("transformer_options", None)
        kwargs.pop("attention_mask", None)
        
        model_device = next(self.model.parameters()).device

        # Move input tensors to the model's device
        if x.device != model_device:
            x = x.to(model_device)
        if context is not None and context.device != model_device:
            context = context.to(model_device)

        # Keep original input shape check
        input_is_5d = x.ndim == 5
        if input_is_5d:
            x = x.squeeze(2)

        if self.customized_forward:
            with torch.inference_mode():
                # Remove guidance, ref_latents, transformer_options, and attention_mask from forward_kwargs to avoid duplicate argument error
                # These are passed explicitly to customized_forward
                # Note: kwargs is already cleaned above, so we only need to clean forward_kwargs
                forward_kwargs_cleaned = {k: v for k, v in self.forward_kwargs.items() if k not in ("guidance", "ref_latents", "transformer_options", "attention_mask")}
                # Create a copy of transformer_options and remove guidance and ref_latents if present
                transformer_options_cleaned = dict(transformer_options) if transformer_options else {}
                transformer_options_cleaned.pop("guidance", None)
                transformer_options_cleaned.pop("ref_latents", None)
                
                return self.customized_forward(
                    self.model,
                    hidden_states=x,
                    encoder_hidden_states=context,
                    timestep=timestep,
                    guidance=guidance if self.config.get("guidance_embed", False) else None,
                    ref_latents=ref_latents_value,  # ref_latents from kwargs (already extracted above)
                    control=control,
                    transformer_options=transformer_options_cleaned,
                    **forward_kwargs_cleaned,
                    **kwargs,  # kwargs is already cleaned above (guidance, ref_latents, transformer_options, attention_mask removed)
                )
        else:
            with torch.inference_mode():
                # Check if input tensor needs dimension adjustment
                if x.ndim == 4:
                    # Add time dimension for 5D tensor (bs, c, t, h, w)
                    x = x.unsqueeze(2)
                
                # Prepare values to pass as positional arguments
                # Note: ref_latents_value was already extracted from kwargs at the start of _execute_model
                guidance_value = guidance if self.config.get("guidance_embed", False) else None
                
                # Remove guidance, transformer_options, and attention_mask from kwargs to avoid duplicate argument error
                # These are passed as positional arguments to match QwenImageTransformer2DModel.forward signature
                # Signature: forward(self, x, timestep, context, attention_mask=None, guidance=None, ref_latents=None, transformer_options={}, **kwargs)
                # Note: ref_latents was already removed from kwargs at the start of _execute_model
                kwargs_cleaned = {k: v for k, v in kwargs.items() if k not in ("guidance", "transformer_options", "attention_mask")}
                # Create a copy of transformer_options and remove guidance and ref_latents if present
                transformer_options_cleaned = dict(transformer_options) if transformer_options else {}
                transformer_options_cleaned.pop("guidance", None)
                transformer_options_cleaned.pop("ref_latents", None)
                
                # Include control in kwargs if provided
                # If control is in kwargs, use it; otherwise use the control parameter
                final_kwargs = dict(kwargs_cleaned)
                if control is not None:
                    final_kwargs["control"] = control
                # Note: control is not in the forward signature, so it should be in **kwargs
                
                # Final cleanup: Remove guidance, ref_latents, transformer_options, and attention_mask from final_kwargs
                # This is necessary because comfy/ldm/qwen_image/model.py's forward() passes these as positional arguments
                # to WrapperExecutor.execute(), and if they are also in **kwargs, _forward() will receive them twice
                # This prevents the error: TypeError: got multiple values for argument 'guidance'
                final_kwargs_cleaned = {k: v for k, v in final_kwargs.items() if k not in ("guidance", "ref_latents", "transformer_options", "attention_mask")}
                
                return self.model(
                    x,
                    timestep,
                    context,
                    None,  # attention_mask
                    guidance_value,  # guidance as positional argument
                    ref_latents_value,  # ref_latents as positional argument
                    transformer_options_cleaned,  # transformer_options
                    **final_kwargs_cleaned,
                )
