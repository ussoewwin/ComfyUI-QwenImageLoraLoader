from typing import Callable, List, Tuple, Union
from pathlib import Path

import torch
from torch import nn
import comfy.model_management
import logging

from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku.caching.fbcache import cache_context, create_cache_context
from nunchaku_code.lora_qwen import compose_loras_v2, reset_lora_v2

logger = logging.getLogger(__name__)


class ComfyZImageTurboWrapper(nn.Module):
    """
    Wrapper for NunchakuZImageTransformer2DModel to support ComfyUI workflows.

    This wrapper separates LoRA composition from the forward pass for maximum efficiency.
    It detects changes to its `loras` attribute and recomposes the underlying model
    lazily when the forward pass is executed.
    """

    def __init__(
            self,
            model: NunchakuZImageTransformer2DModel,
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

    def _get_model_safe(self):
        """Safely get model attribute from _modules or __dict__ of nn.Module.
        
        In nn.Module subclasses, when self.model = nn.Module() is set, the model attribute
        is stored in the _modules dictionary. Access via __getattr__ can cause issues,
        so we retrieve it directly from the dictionary.
        """
        # First check __dict__ (regular Python attributes)
        if 'model' in self.__dict__:
            return self.__dict__['model']
        # Next check _modules (nn.Module child modules)
        modules = self.__dict__.get('_modules', {})
        if modules and 'model' in modules:
            return modules['model']
        return None

    def to_safely(self, device):
        """Safely move the model to the specified device."""
        # Use _get_model_safe to bypass nn.Module's attribute system
        model = self._get_model_safe()
        if model is None:
            return self
        if hasattr(model, "to_safely"):
            model.to_safely(device)
        else:
            model.to(device)
        return self

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model (e.g., layers, config, etc.)."""
        # Special handling for direct access to 'model' attribute
        # nn.Module stores child modules in _modules, so normal attribute access won't find them
        if name == 'model':
            # Try to get from _modules
            modules = self.__dict__.get('_modules', {})
            if modules and 'model' in modules:
                return modules['model']
            # Try to get from __dict__
            if 'model' in self.__dict__:
                return self.__dict__['model']
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'model'")
        
        # Delegate attributes that don't exist on this wrapper to the wrapped model (NextDiT)
        # This allows ZImageModelPatcher and other code to access NextDiT attributes like 'layers'
        # Use _get_model_safe to bypass nn.Module's attribute system
        model = self._get_model_safe()
        if model is not None:
            try:
                return getattr(model, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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
        Forward pass for the wrapped Z-Image-Turbo model.

        Detects changes to the `self.loras` list and recomposes the model
        on-the-fly before inference.
        """
        # Remove guidance, transformer_options, and attention_mask from kwargs
        if "guidance" in kwargs:
            kwargs.pop("guidance")
        if "transformer_options" in kwargs:
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

        # Guard against None model (can happen during GC/unload)
        if self.model is None:
            raise RuntimeError("Model has been unloaded or garbage collected. Cannot perform forward pass.")

        model_is_dirty = (
            not self.loras and # We expect no LoRA
            hasattr(self.model, "_lora_slots") and self.model._lora_slots # But the model actually has LoRA
        )
        
        # Deep comparison of LoRA stacks to detect any changes
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
            
            # Reset cache when LoRAs change
            if loras_changed:
                self._cache_context = None
                self._prev_timestep = None
                logger.debug("Cache reset due to LoRA change")

            # Dynamic VRAM check for CPU offload
            offload_is_on = hasattr(self.model, "offload_manager") and self.model.offload_manager is not None
            should_enable_offload = offload_is_on

            if self.cpu_offload_setting == "auto" and not offload_is_on and self.loras:
                try:
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

            # 4. Compose LoRAs. This changes internal tensor shapes.
            # Returns True if successful (supported format), False if unsupported (skipped).
            is_supported_format = compose_loras_v2(self.model, self.loras)

            # Validate composition result; if 0 targets after a crash/transition, retry once
            # But ONLY if the format was supported. If unsupported, retrying is pointless.
            try:
                has_slots = hasattr(self.model, "_lora_slots") and bool(self.model._lora_slots)
            except Exception:
                has_slots = True
            
            if self.loras and not has_slots and is_supported_format:
                logger.warning("LoRA composition reported 0 target modules (supported format). Forcing reset and one retry.")
                try:
                    reset_lora_v2(self.model)
                    compose_loras_v2(self.model, self.loras)
                except Exception as e:
                    logger.error(f"LoRA re-compose retry failed: {e}")
            elif self.loras and not has_slots and not is_supported_format:
                logger.info("Skipping retry because LoRA format is unsupported.")

            # Re-build offload manager if it's supposed to be on
            if should_enable_offload:
                if offload_is_on:
                    manager = self.model.offload_manager
                    offload_settings = {
                        "num_blocks_on_gpu": manager.num_blocks_on_gpu,
                        "use_pin_memory": manager.use_pin_memory,
                    }
                else:
                    offload_settings = {
                        "num_blocks_on_gpu": 1,
                        "use_pin_memory": False,
                    }
                    logger.info("Building new CPU offload manager due to LoRA VRAM check.")

                # Step 1: Completely disable and clear any old offloader
                if hasattr(self.model, "set_offload"):
                    self.model.set_offload(False)
                    # Step 2: Re-enable offloading with the correct settings
                    self.model.set_offload(True, **offload_settings)

            # Update last known device signature after (re)composition
            self._last_device = current_device

        # Caching logic
        use_caching = getattr(self.model, "residual_diff_threshold_multi", 0) != 0 or getattr(self.model, "_is_cached", False)
        if use_caching:
            cache_invalid = self._prev_timestep is None or self._prev_timestep < timestep_float + 1e-5
            if cache_invalid:
                self._cache_context = create_cache_context()
            self._prev_timestep = timestep_float

            with cache_context(self._cache_context):
                out = self._execute_model(x, timestep, context, y, guidance, control, transformer_options, **kwargs)
        else:
            out = self._execute_model(x, timestep, context, y, guidance, control, transformer_options, **kwargs)

        if isinstance(out, tuple):
            out = out[0]

        # Check if x is a list (Z-Image-Turbo format) or tensor
        if isinstance(x, list):
            # For list format, check output dimensions differently
            if isinstance(out, list) and len(out) > 0:
                # Output is already in list format, no need to adjust
                pass
        else:
            # For tensor format, check dimensions
            if x.ndim == 5 and out.ndim == 4:
                out = out.unsqueeze(2)

        return out

    def _execute_model(self, x, timestep, context, y, guidance, control, transformer_options, **kwargs):
        """Helper function to run the Z-Image-Turbo model's forward pass."""
        # Get ref_latents from kwargs before removing it
        ref_latents_value = kwargs.pop("ref_latents", None)
        
        # Remove guidance, transformer_options, and attention_mask from kwargs
        kwargs.pop("guidance", None)
        kwargs.pop("transformer_options", None)
        kwargs.pop("attention_mask", None)
        
        model_device = next(self.model.parameters()).device

        # Z-Image-Turbo model base already converts x to list format
        # Check if x is already a list (from model base) or still a tensor
        if isinstance(x, list):
            # x is already in list format from model base
            # Filter out None elements and move each tensor to the model's device
            x_list = []
            for img in x:
                if img is not None:
                    if img.device != model_device:
                        x_list.append(img.to(model_device))
                    else:
                        x_list.append(img)
            # Check if x_list is empty after filtering
            if len(x_list) == 0:
                raise ValueError("x_list is empty after filtering None elements - cannot process empty batch")
            x = x_list
        else:
            # x is still a tensor, move it to the model's device
            if x.device != model_device:
                x = x.to(model_device)
            
            # Keep original input shape check for tensor format
            input_is_5d = x.ndim == 5
            if input_is_5d:
                x = x.squeeze(2)
        
        # Move context to the model's device if it's a tensor
        if context is not None and not isinstance(context, list):
            if context.device != model_device:
                context = context.to(model_device)

        if self.customized_forward:
            with torch.inference_mode():
                forward_kwargs_cleaned = {k: v for k, v in self.forward_kwargs.items() if k not in ("guidance", "ref_latents", "transformer_options", "attention_mask")}
                transformer_options_cleaned = dict(transformer_options) if transformer_options else {}
                transformer_options_cleaned.pop("guidance", None)
                transformer_options_cleaned.pop("ref_latents", None)
                
                return self.customized_forward(
                    self.model,
                    hidden_states=x,
                    encoder_hidden_states=context,
                    timestep=timestep,
                    guidance=guidance if self.config.get("guidance_embed", False) else None,
                    ref_latents=ref_latents_value,
                    control=control,
                    transformer_options=transformer_options_cleaned,
                    **forward_kwargs_cleaned,
                    **kwargs,
                )
        else:
            with torch.inference_mode():
                # Check for NextDiT signature first (forward(x, timesteps, context, num_tokens, ...))
                import inspect
                forward_sig = inspect.signature(self.model.forward)
                forward_params = set(forward_sig.parameters.keys())
                
                # Branch 1: NextDiT signature (ComfyUI Lumina2 / NextDiT)
                if "context" in forward_params and "num_tokens" in forward_params:
                    # NextDiT forward signature: forward(x, timesteps, context, num_tokens, attention_mask=None, **kwargs)
                    # Process x for NextDiT format
                    if isinstance(x, list):
                        # Stack list of tensors into a single tensor
                        x_stack = torch.stack([t for t in x if t is not None], dim=0)
                        if x_stack.ndim == 5 and x_stack.shape[2] == 1:
                            x_stack = x_stack.squeeze(2)  # Remove frame dimension if F=1
                        x_tensor = x_stack
                    else:
                        x_tensor = x
                    
                    # Extract num_tokens from kwargs if not provided
                    num_tokens_value = kwargs.pop("num_tokens", None)
                    if num_tokens_value is None and isinstance(context, torch.Tensor) and context.ndim >= 2:
                        num_tokens_value = context.shape[1]
                    
                    # Extract attention_mask from kwargs
                    attention_mask_value = kwargs.pop("attention_mask", None)
                    
                    # Check if forward method supports **kwargs
                    supports_kwargs = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in forward_sig.parameters.values()
                    )
                    
                    forward_kwargs = {}
                    if "attention_mask" in forward_params:
                        forward_kwargs["attention_mask"] = attention_mask_value
                    
                    if supports_kwargs:
                        # Ensure patches (e.g. double_block from DiffSynth/ControlNet) reach NextDiT
                        forward_kwargs["transformer_options"] = transformer_options
                        if control is not None:
                            forward_kwargs["control"] = control
                    else:
                        if "transformer_options" in forward_params:
                            forward_kwargs["transformer_options"] = transformer_options
                        if "control" in forward_params:
                            forward_kwargs["control"] = control
                    
                    # NextDiT expects context and num_tokens as required args
                    return self.model(
                        x_tensor,
                        timestep,
                        context=context,
                        num_tokens=num_tokens_value,
                        **forward_kwargs,
                        **kwargs,
                    )
                
                # Branch 2: Z-Image-Turbo forward signature: forward(x, t, cap_feats, patch_size=2, f_patch_size=1, return_dict=True)
                # x: List[torch.Tensor] with shape (C, F, H, W) for each image
                # t: normalized timestep in [0, 1]
                # cap_feats: List[torch.Tensor] or None
                
                # x is already in list format from model base or converted above
                if isinstance(x, list):
                    x_list_raw = x
                    x_list = []
                    for img in x_list_raw:
                        if img is not None and img.numel() > 0:
                            # Handle standard ComfyUI latent batch size 1: (1, C, H, W) -> remove batch dim -> (C, H, W)
                            if img.dim() == 4 and img.shape[0] == 1:
                                img = img.squeeze(0)

                            # Ensure frame dimension: (C, H, W) -> (C, 1, H, W)
                            if img.dim() == 3:
                                img = img.unsqueeze(1)

                            x_list.append(img)
                else:
                    # Convert x to list format if it's still a tensor
                    x_list = []
                    if x.shape[0] > 0:  # Check batch size
                        for i in range(x.shape[0]):
                            img = x[i]  # (C, H, W) or (C, F, H, W)
                            if img.numel() > 0:  # Check if tensor is not empty
                                if img.dim() == 3:
                                    # Add frame dimension: (C, H, W) -> (C, 1, H, W)
                                    img = img.unsqueeze(1)
                                x_list.append(img)
                
                # Check if x_list is empty (should not happen, but handle gracefully)
                if len(x_list) == 0:
                    raise ValueError("x_list is empty - cannot process empty batch. x type: {}, x shape: {}".format(type(x), getattr(x, 'shape', 'N/A')))
                
                # Normalize timestep: ComfyUI passes sigma [1 -> 0], Z-Image-Turbo expects [0 -> 1]
                # For now, assume timestep is already in the correct format from model base
                # If not, we may need to convert: t_zimage = 1.0 - t_normalized
                t_zimage = timestep
                
                # Convert cap_feats (context) to list format if needed
                # ComfyUI passes c_concat as context (arg 3) and c_crossattn as y (arg 4)
                # Z-Image needs caption features. If context (c_concat) is empty, try y (c_crossattn).
                # Also check kwargs for explicit 'cap_feats' passed by some wrappers (e.g. zimage.py)
                cap_feats = kwargs.pop("cap_feats", None)
                
                if cap_feats is None:
                    cap_feats = context
                    if cap_feats is None and y is not None:
                        cap_feats = y
                
                if cap_feats is not None:
                    if not isinstance(cap_feats, list):
                        # Split batch into list of tensors
                        if cap_feats.shape[0] > 0:  # Check batch size
                            cap_feats = [cap_feats[i] for i in range(cap_feats.shape[0])]
                        else:
                            cap_feats = []
                else:
                    cap_feats = []
                
                # Prepare kwargs for Z-Image-Turbo
                zimage_kwargs = {}
                if "patch_size" in kwargs:
                    zimage_kwargs["patch_size"] = kwargs["patch_size"]
                if "f_patch_size" in kwargs:
                    zimage_kwargs["f_patch_size"] = kwargs["f_patch_size"]
                zimage_kwargs["return_dict"] = False
                
                # =============================================================================
                # DEBUG LOGGING (MUTED) - Uncomment to diagnose input tensor issues
                # =============================================================================
                # This debug block logs the shapes of input tensors for Z-Image-Turbo forward pass.
                # It is specifically active when using NunchakuZImageTurboLoraStackV2 (zimageturbo_v2.py)
                # which wraps the model with ComfyZImageTurboWrapper.
                #
                # Output format:
                #   - x_list: List of latent tensors, shape [C, F, H, W] = [channels, frames, height, width]
                #   - cap_feats: Caption features from text encoder, shape [tokens, embed_dim]
                #
                # Note: Logs appear twice per step due to CFG (Classifier-Free Guidance) which runs
                # both positive and negative conditioning through the model.
                #
                # To enable: Remove the '#' prefix from the logger.info lines below.
                # =============================================================================
                # logger.info(f"DEBUG: x_list len={len(x_list) if isinstance(x_list, list) else 'Not List'}")
                # if isinstance(x_list, list) and len(x_list) > 0:
                #      logger.info(f"DEBUG: x_list[0] shape={x_list[0].shape}")
                # logger.info(f"DEBUG: cap_feats len={len(cap_feats) if isinstance(cap_feats, list) else 'Not List'}")
                # if isinstance(cap_feats, list) and len(cap_feats) > 0:
                #     logger.info(f"DEBUG: cap_feats[0] shape={cap_feats[0].shape}")
                
                # Check for mismatch which causes zip to truncate
                if isinstance(x_list, list) and isinstance(cap_feats, list):
                    if len(x_list) != len(cap_feats):
                        logger.warning(f"WARNING: Mismatch in lengths! x_list={len(x_list)}, cap_feats={len(cap_feats)}. This will cause truncation in zip()!")
                        if len(cap_feats) == 0:
                             raise ValueError("cap_feats is empty! Model requires caption features (y/c_crossattn).")

                # Call Z-Image-Turbo forward with correct signature
                # Pass control and transformer_options to allow Model Patcher (double_block patches) to work
                # forward_sig and forward_params are already defined above (NextDiT branch check)
                
                zimage_kwargs_clean = {k: v for k, v in zimage_kwargs.items() if k not in ('control', 'transformer_options')}
                
                # Build kwargs based on what the forward method accepts
                forward_kwargs = zimage_kwargs_clean.copy()
                if 'control' in forward_params:
                    forward_kwargs['control'] = control
                if 'transformer_options' in forward_params:
                    forward_kwargs['transformer_options'] = transformer_options
                
                model_output = self.model(
                    x_list,
                    t_zimage,
                    cap_feats=cap_feats,
                    **forward_kwargs
                )
                
                # Extract list from tuple: (x,) -> x
                if isinstance(model_output, tuple):
                    model_output = model_output[0]
                
                # Convert output from List[torch.Tensor] to single tensor (B, C, H, W)
                if isinstance(model_output, list):
                    # Stack list of tensors: each tensor is (C, F, H, W)
                    model_output = torch.stack(model_output, dim=0)  # (B, C, F, H, W)
                    # Remove frame dimension F=1: (B, C, F, H, W) -> (B, C, H, W)
                    if model_output.shape[2] == 1:
                        model_output = model_output.squeeze(2)
                
                return model_output

