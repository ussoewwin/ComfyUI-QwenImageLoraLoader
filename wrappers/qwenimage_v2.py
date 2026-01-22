from typing import Callable, List, Tuple, Union
from pathlib import Path

import torch
from torch import nn
import comfy.model_management
import logging

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.caching.fbcache import cache_context, create_cache_context
from nunchaku_code.lora_qwen import compose_loras_v2_v2, reset_lora_v2

logger = logging.getLogger(__name__)


class ComfyQwenImageWrapperV2(nn.Module):
    """
    V2 node-specific wrapper for NunchakuQwenImageTransformer2DModel to support ComfyUI workflows.
    
    This wrapper uses compose_loras_v2_v2 to ensure complete isolation from v1/v3 nodes.
    It separates LoRA composition from the forward pass for maximum efficiency.
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
            vram_margin_gb: float = 4.0,
            apply_awq_mod: bool | str | None = None,
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
        logger.info(f"[V2 Node] 🔧 CPU offload setting: '{cpu_offload_setting}' (VRAM margin: {vram_margin_gb}GB)")

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

        # AWQ modulation control (None means "follow global env setting")
        self.apply_awq_mod = apply_awq_mod

    def to_safely(self, device):
        """Safely move the model to the specified device."""
        if self.model is None:
            return self
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
        on-the-fly before inference using v2-specific compose function.
        """
        # Remove guidance, transformer_options, and attention_mask from kwargs
        # These may be added by ComfyUI-EulerDiscreteScheduler or other patches
        # Even though guidance is a parameter of forward(), kwargs may also contain it
        # which could cause issues when passing to _execute_model
        # Note: ref_latents is NOT a parameter of forward(), so we keep it in kwargs for _execute_model to use
        if "guidance" in kwargs:
            # Preserve legacy guidance if it only arrives via kwargs (environment-dependent),
            # then remove it to avoid duplication downstream.
            if guidance is None:
                guidance = kwargs.get("guidance")
            kwargs.pop("guidance")
        if "transformer_options" in kwargs:
            # Merge transformer_options from kwargs into the parameter if needed
            if isinstance(transformer_options, dict) and isinstance(kwargs["transformer_options"], dict):
                transformer_options = {**transformer_options, **kwargs.pop("transformer_options")}
            else:
                kwargs.pop("transformer_options")
        # Preserve legacy guidance if it only arrives via transformer_options (environment-dependent).
        # We will later remove it from transformer_options when actually calling the model.
        if guidance is None and isinstance(transformer_options, dict) and "guidance" in transformer_options:
            guidance = transformer_options.get("guidance")
        if "attention_mask" in kwargs:
            kwargs.pop("attention_mask")
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep_float = timestep.item()
            else:
                timestep_float = timestep.flatten()[0].item()
        else:
            timestep_float = float(timestep)

        # Guard against None model (can happen during GC/unload)
        if self.model is None:
            raise RuntimeError("Model has been unloaded or garbage collected. Cannot perform forward pass.")

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
                logger.debug("[V2 Node] Cache reset due to LoRA change")

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
                            f"[V2 Node] Free VRAM is {free_vram_gb:.2f}GB (below safety margin of {self.vram_margin_gb}GB) and 'cpu_offload' is 'auto'. Force-enabling CPU offload for LoRA composition.")
                        should_enable_offload = True
                    else:
                        logger.info(
                            f"[V2 Node] Free VRAM is {free_vram_gb:.2f}GB (>= {self.vram_margin_gb}GB margin). LoRAs will be composed without enabling CPU offload.")

                except Exception as e:
                    logger.error(f"[V2 Node] Error during VRAM check for LoRA offloading: {e}. Offload will not be enabled.")
            elif self.cpu_offload_setting == "disable" and not offload_is_on:
                logger.debug("[V2 Node] CPU offload is 'disable' and not on. Skipping VRAM check.")
            elif self.cpu_offload_setting == "enable" and offload_is_on:
                logger.debug("[V2 Node] CPU offload is 'enable'. Will rebuild offload manager for LoRAs.")

            # --- END NEW VRAM CHECK ---

            # 4. Compose LoRAs using v2-specific function. This changes internal tensor shapes.
            # Returns True if successful (supported format), False if unsupported (skipped).
            logger.info(f"[V2 Node] 🔧 Forward pass: calling compose_loras_v2_v2 with apply_awq_mod={self.apply_awq_mod} (type: {type(self.apply_awq_mod).__name__})")
            is_supported_format = compose_loras_v2_v2(self.model, self.loras, apply_awq_mod=self.apply_awq_mod)

            # Validate composition result; if 0 targets after a crash/transition, retry once
            # But ONLY if the format was supported. If unsupported, retrying is pointless.
            if is_supported_format:
                try:
                    has_slots = hasattr(self.model, "_lora_slots") and bool(self.model._lora_slots)
                except Exception:
                    has_slots = True
                if self.loras and not has_slots:
                    logger.warning("[V2 Node] LoRA composition reported 0 target modules. Forcing reset and one retry.")
                    try:
                        reset_lora_v2(self.model)
                        compose_loras_v2_v2(self.model, self.loras, apply_awq_mod=self.apply_awq_mod)
                    except Exception as e:
                        logger.error(f"[V2 Node] LoRA re-compose retry failed: {e}")
            else:
                 logger.warning("[V2 Node] Skipping retry because LoRA format is unsupported.")

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
                    logger.info("[V2 Node] Building new CPU offload manager due to LoRA VRAM check.")

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
        
        # Extract additional_t_cond from kwargs before removing it (needed for both customized_forward and direct model call)
        # ComfyUI v0.6.0+ uses additional_t_cond instead of guidance
        additional_t_cond_value = kwargs.pop("additional_t_cond", None)

        # CRITICAL FIX for Issue #43 (ComfyUI-nunchaku):
        # - ComfyUI-nunchaku's _forward() has a bug where it calls:
        #     time_text_embed(timestep, guidance, hidden_states)
        #   when guidance is not None, causing:
        #     TypeError: QwenTimestepProjEmbeddings.forward() takes 3 positional arguments but 4 were given
        # - We must ensure "guidance" never reaches ComfyUI-nunchaku as a non-None value.
        #
        # However, we still want to preserve the original *meaning* of legacy "guidance" inputs:
        # if additional_t_cond is missing but guidance is provided (and the model supports guidance embedding),
        # treat guidance as additional_t_cond.
        #
        # So we *convert* guidance -> additional_t_cond_value (fallback), then force guidance=None.
        if additional_t_cond_value is None and guidance is not None and self.config.get("guidance_embed", False):
            additional_t_cond_value = guidance
        guidance = None
        
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
                # IMPORTANT: ComfyUI v0.6.0+ uses additional_t_cond instead of guidance
                # Use additional_t_cond_value extracted above, or fallback to guidance if available
                customized_additional_t_cond = additional_t_cond_value
                if customized_additional_t_cond is None and guidance is not None and self.config.get("guidance_embed", False):
                    customized_additional_t_cond = guidance
                
                # Remove guidance, additional_t_cond, ref_latents, transformer_options, and attention_mask from forward_kwargs to avoid duplicate argument error
                # These are passed explicitly to customized_forward
                # Note: kwargs is already cleaned above, so we only need to clean forward_kwargs
                forward_kwargs_cleaned = {k: v for k, v in self.forward_kwargs.items() if k not in ("guidance", "additional_t_cond", "ref_latents", "transformer_options", "attention_mask")}
                # Create a copy of transformer_options and remove guidance and ref_latents if present
                transformer_options_cleaned = dict(transformer_options) if transformer_options else {}
                transformer_options_cleaned.pop("guidance", None)
                transformer_options_cleaned.pop("ref_latents", None)
                
                return self.customized_forward(
                    self.model,
                    hidden_states=x,
                    encoder_hidden_states=context,
                    timestep=timestep,
                    additional_t_cond=customized_additional_t_cond,  # additional_t_cond (ComfyUI v0.6.0+)
                    ref_latents=ref_latents_value,  # ref_latents from kwargs (already extracted above)
                    control=control,
                    transformer_options=transformer_options_cleaned,
                    **forward_kwargs_cleaned,
                    **kwargs,  # kwargs is already cleaned above (guidance, additional_t_cond, ref_latents, transformer_options, attention_mask removed)
                )
        else:
            with torch.inference_mode():
                # Check if input tensor needs dimension adjustment
                if x.ndim == 4:
                    # Add time dimension for 5D tensor (bs, c, t, h, w)
                    x = x.unsqueeze(2)
                
                # Prepare values to pass as positional arguments
                # Note: ref_latents_value and additional_t_cond_value were already extracted from kwargs at the start of _execute_model
                # IMPORTANT: ComfyUI v0.6.0+ QwenImageTransformer2DModel.forward signature is:
                # forward(self, x, timestep, context, attention_mask=None, ref_latents=None, additional_t_cond=None, transformer_options={}, **kwargs)
                # Note: There is NO 'guidance' parameter in the forward signature!
                # guidance is handled internally via additional_t_cond

                # Use additional_t_cond_value extracted above, or fallback to guidance if available
                model_additional_t_cond = additional_t_cond_value
                if model_additional_t_cond is None and guidance is not None and self.config.get("guidance_embed", False):
                    model_additional_t_cond = guidance

                # Remove guidance, transformer_options, and attention_mask from kwargs to avoid duplicate argument error
                kwargs_cleaned = {k: v for k, v in kwargs.items() if k not in ("guidance", "transformer_options", "attention_mask")}

                # Create a copy of transformer_options and remove guidance and ref_latents if present
                transformer_options_cleaned = dict(transformer_options) if transformer_options else {}
                transformer_options_cleaned.pop("guidance", None)
                transformer_options_cleaned.pop("ref_latents", None)

                # Include control in kwargs if provided
                final_kwargs = dict(kwargs_cleaned)
                if control is not None:
                    final_kwargs["control"] = control

                # Final cleanup: Remove any remaining problematic keys
                final_kwargs_cleaned = {k: v for k, v in final_kwargs.items() if k not in ("guidance", "ref_latents", "transformer_options", "attention_mask", "additional_t_cond")}

                return self.model(
                    x,
                    timestep,
                    context,
                    None,  # attention_mask
                    ref_latents_value,  # ref_latents as positional argument
                    model_additional_t_cond,  # additional_t_cond as positional argument
                    transformer_options_cleaned,  # transformer_options
                    **final_kwargs_cleaned,
                )
