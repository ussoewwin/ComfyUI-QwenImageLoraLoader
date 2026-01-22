
import torch
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def forward_with_manual_planar_injection(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    timestep_zero_index=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for the transformer block with Manual Planar Injection support for LoRA.
    Monkey-patched into NunchakuQwenImageTransformerBlock by ComfyUI-QwenImageLoraLoader.
    """
    # Get modulation parameters for both streams
    img_mod_params = self.img_mod(temb)  # [B, 6*dim]
    # --- Nunchaku LoRA Patch (Manual Planar Injection) ---
    if hasattr(self.img_mod[1], "_nunchaku_lora_bundle"):
            A, B = self.img_mod[1]._nunchaku_lora_bundle
            # The Linear layer receives SiLU(temb), so we must apply it to LoRA input too
            input_tensor = self.img_mod[0](temb)
            # Cast to LoRA dtype (FP16) before matmul to avoid BFloat16 vs Float16 mismatch
            lora = (input_tensor.to(dtype=A.dtype) @ A.t()) @ B.t()
            img_mod_params = img_mod_params + lora.to(dtype=img_mod_params.dtype, device=img_mod_params.device)
    # -----------------------------------------------------

    txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]
    # --- Nunchaku LoRA Patch (Manual Planar Injection) ---
    if hasattr(self.txt_mod[1], "_nunchaku_lora_bundle"):
            A, B = self.txt_mod[1]._nunchaku_lora_bundle
            # The Linear layer receives SiLU(temb), so we must apply it to LoRA input too
            input_tensor = self.txt_mod[0](temb)
            lora = (input_tensor.to(dtype=A.dtype) @ A.t()) @ B.t()
            txt_mod_params = txt_mod_params + lora.to(dtype=txt_mod_params.dtype, device=txt_mod_params.device)
    # -----------------------------------------------------

    # Nunchaku's mod_params is [B, 6*dim] instead of [B, dim*6]
    img_mod_params = (
        img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(img_mod_params.shape[0], -1)
    )
    txt_mod_params = (
        txt_mod_params.view(txt_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(txt_mod_params.shape[0], -1)
    )

    img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
    txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

    # Process image stream - norm1 + modulation
    img_normed = self.img_norm1(hidden_states)
    img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

    # Process text stream - norm1 + modulation
    txt_normed = self.txt_norm1(encoder_hidden_states)
    txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

    # Joint attention computation (DoubleStreamLayerMegatron logic)
    attn_output = self.attn(
        hidden_states=img_modulated,  # Image stream ("sample")
        encoder_hidden_states=txt_modulated,  # Text stream ("context")
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        image_rotary_emb=image_rotary_emb,
    )

    # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
    img_attn_output, txt_attn_output = attn_output

    # Apply attention gates and add residual (like in Megatron)
    hidden_states = hidden_states + img_gate1 * img_attn_output
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

    # Process image stream - norm2 + MLP
    img_normed2 = self.img_norm2(hidden_states)
    img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
    img_mlp_output = self.img_mlp(img_modulated2)
    hidden_states = hidden_states + img_gate2 * img_mlp_output

    # Process text stream - norm2 + MLP
    txt_normed2 = self.txt_norm2(encoder_hidden_states)
    txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
    txt_mlp_output = self.txt_mlp(txt_modulated2)
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

    return encoder_hidden_states, hidden_states


def apply_nunchaku_patch():
    try:
        # Try importing from the expected package structure
        # Assuming ComfyUI-nunchaku is loaded as a custom node and its models directory handles imports
        # We need to find the NunchakuQwenImageTransformerBlock class
        
        # Strategy: Look for the module in sys.modules first, or try standard import paths
        import sys
        
        # Check if we can import directly if it's in python path
        # Or if we have to go through custom_nodes
        
        target_class = None
        
        # 1. Try importing nunchaku.models.qwenimage (if standard install or adjusted path)
        try:
            from nunchaku.models.qwenimage import NunchakuQwenImageTransformerBlock
            target_class = NunchakuQwenImageTransformerBlock
        except ImportError:
            pass
            
        # 2. Try importing via custom_nodes path mechanism if above failed
        # This is strictly environment dependent but common in Comfy
        if target_class is None:
             # Often custom nodes are not importable as top level packages unless they add themselves
             # We can try to walk sys.modules to find it
             for module_name, module in sys.modules.items():
                 if "qwenimage" in module_name and hasattr(module, "NunchakuQwenImageTransformerBlock"):
                     target_class = getattr(module, "NunchakuQwenImageTransformerBlock")
                     logger.info(f"Found NunchakuQwenImageTransformerBlock in {module_name}")
                     break

        if target_class:
            logger.info("Applying Manual Planar Injection Monkey Patch to NunchakuQwenImageTransformerBlock")
            target_class.forward = forward_with_manual_planar_injection
            return True
        else:
            logger.warning("Could not find NunchakuQwenImageTransformerBlock to patch. Manual Planar Injection logic will not work if the original file is reverted.")
            return False

    except Exception as e:
        logger.error(f"Failed to apply Nunchaku patch: {e}")
        return False
