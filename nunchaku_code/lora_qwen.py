import functools
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors import safe_open

# It's assumed these functions from your project are available for import.
# If not, you'll need to provide their definitions.
from nunchaku.lora.flux.nunchaku_converter import (
    pack_lowrank_weight,
    reorder_adanorm_lora_up,
    unpack_lowrank_weight,
)

logger = logging.getLogger(__name__)

# --- Centralized & Optimized Key Mapping ---
# This structure is faster to process and easier to maintain than a long if/elif chain.
# --- CORRECTED Centralized & Optimized Key Mapping ---
# --- Centralized & Optimized Key Mapping ---
# This version correctly tokenizes all module paths.
KEY_MAPPING = [
    # Fused QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]qkv$"), r"\1.\2.attn.to_qkv", "qkv", None),
    # Decomposed QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._](q|k|v)$"), r"\1.\2.attn.to_qkv", "qkv",
     lambda m: m.group(3).upper()),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._](q|k|v)[._]proj$"), r"\1.\2.attn.to_qkv", "qkv",
     lambda m: m.group(3).upper()),
    # Fused Add_QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]add[._]qkv[._]proj$"), r"\1.\2.attn.add_qkv_proj",
     "add_qkv", None),
    # Decomposed Add_QKV (Double Block)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]add[._](q|k|v)[._]proj$"), r"\1.\2.attn.add_qkv_proj",
     "add_qkv",
     lambda m: m.group(3).upper()),
    # Fused QKV (Single Block)
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._]qkv$"), r"\1.\2.attn.to_qkv", "qkv", None),
    # Decomposed QKV (Single Block)
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._](q|k|v)$"), r"\1.\2.attn.to_qkv", "qkv",
     lambda m: m.group(3).upper()),
    # Output Projections
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]out[._]proj[._]context$"), r"\1.\2.attn.to_add_out", "regular",
     None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]add[._]out$"), r"\1.\2.attn.to_add_out",
     "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]out[._]proj$"), r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]out$"), r"\1.\2.attn.to_out.0", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]attn[._]to[._]out[._]0$"), r"\1.\2.attn.to_out.0", "regular",
     None),
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]attn[._]to[._]out$"), r"\1.\2.attn.to_out", "regular",
     None),

    # Feed-Forward / MLP Layers (Standard)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff[._]net[._]0(?:[._]proj)?$"), r"\1.\2.mlp_fc1", "regular",
     None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff[._]net[._]2$"), r"\1.\2.mlp_fc2", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff_context[._]net[._]0(?:[._]proj)?$"),
     r"\1.\2.mlp_context_fc1", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]ff_context[._]net[._]2$"), r"\1.\2.mlp_context_fc2", "regular",
     None),

    # --- THIS IS THE CORRECTED SECTION ---
    # Feed-Forward / MLP Layers (img/txt)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](0)[._](proj)$"), r"\1.\2.\3.\4.\5.\6",
     "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](0)$"), r"\1.\2.\3.\4.\5", "regular",
     None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mlp)[._](net)[._](2)$"), r"\1.\2.\3.\4.\5", "regular",
     None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](0)[._](proj)$"), r"\1.\2.\3.\4.\5.\6",
     "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](0)$"), r"\1.\2.\3.\4.\5", "regular",
     None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mlp)[._](net)[._](2)$"), r"\1.\2.\3.\4.\5", "regular",
     None),
    # Mod Layers (img/txt)
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](img_mod)[._](1)$"), r"\1.\2.\3.\4", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._](txt_mod)[._](1)$"), r"\1.\2.\3.\4", "regular", None),
    # ------------------------------------

    # Single Block Projections
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]proj[._]out$"), r"\1.\2.proj_out", "single_proj_out",
     None),
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]proj[._]mlp$"), r"\1.\2.mlp_fc1", "regular", None),
    # Normalization Layers
    (re.compile(r"^(single_transformer_blocks)[._](\d+)[._]norm[._]linear$"), r"\1.\2.norm.linear", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]norm1[._]linear$"), r"\1.\2.norm1.linear", "regular", None),
    (re.compile(r"^(transformer_blocks)[._](\d+)[._]norm1_context[._]linear$"), r"\1.\2.norm1_context.linear",
     "regular", None),

    # Mappings for top-level diffusion_model modules
    (re.compile(r"^(img_in)$"), r"\1", "regular", None),
    (re.compile(r"^(txt_in)$"), r"\1", "regular", None),
    (re.compile(r"^(proj_out)$"), r"\1", "regular", None),
    (re.compile(r"^(norm_out)[._](linear)$"), r"\1.\2", "regular", None),
    (re.compile(r"^(time_text_embed)[._](timestep_embedder)[._](linear_1)$"), r"\1.\2.\3", "regular", None),
    (re.compile(r"^(time_text_embed)[._](timestep_embedder)[._](linear_2)$"), r"\1.\2.\3", "regular", None),
]
_RE_LORA_SUFFIX = re.compile(r"\.(?P<tag>lora(?:[._](?:A|B|down|up)))(?:\.[^.]+)*\.weight$")
_RE_LOKR_SUFFIX = re.compile(r"\.(?P<tag>lokr_w[12])(?:\.[^.]+)*$")
_RE_ALPHA_SUFFIX = re.compile(r"\.(?:alpha|lora_alpha)(?:\.[^.]+)*$")


# --- Helper Functions ---
def _rename_layer_underscore_layer_name(old_name: str) -> str:
    """
    Converts specific model layer names by replacing underscore patterns
    with dot notation using an ordered set of regex rules.
    """

    # Rules are ordered from most specific to most general
    # to prevent a general rule from incorrectly matching part
    # of a more specific pattern.
    rules = [
        # Case: transformer_blocks_8_attn_to_out_0 -> transformer_blocks.8.attn.to_out.0
        (r'_(\d+)_attn_to_out_(\d+)', r'.\1.attn.to_out.\2'),

        # Case: transformer_blocks_8_img_mlp_net_0_proj -> transformer_blocks.8.img_mlp.net.0.proj
        (r'_(\d+)_img_mlp_net_(\d+)_proj', r'.\1.img_mlp.net.\2.proj'),

        # Case: transformer_blocks_8_txt_mlp_net_0_proj -> transformer_blocks.8.txt_mlp.net.0.proj
        (r'_(\d+)_txt_mlp_net_(\d+)_proj', r'.\1.txt_mlp.net.\2.proj'),

        # Case: transformer_blocks_8_img_mlp_net_2 -> transformer_blocks.8.img_mlp.net.2
        (r'_(\d+)_img_mlp_net_(\d+)', r'.\1.img_mlp.net.\2'),

        # Case: transformer_blocks_8_txt_mlp_net_2 -> transformer_blocks.8.txt_mlp.net.2
        (r'_(\d+)_txt_mlp_net_(\d+)', r'.\1.txt_mlp.net.\2'),

        # Case: transformer_blocks_8_img_mod_1 -> transformer_blocks.8.img_mod.1
        (r'_(\d+)_img_mod_(\d+)', r'.\1.img_mod.\2'),

        # Case: transformer_blocks_8_txt_mod_1 -> transformer_blocks.8.txt_mod.2
        (r'_(\d+)_txt_mod_(\d+)', r'.\1.txt_mod.\2'),

        # General 'attn' case: transformer_blocks_8_attn_... -> transformer_blocks.8.attn....
        # This catches add_k_proj, add_q_proj, to_k, etc.
        (r'_(\d+)_attn_', r'.\1.attn.'),
    ]

    new_name = old_name
    for pattern, replacement in rules:
        # Apply the substitution. If the pattern doesn't match,
        # re.sub simply returns the original string.
        new_name = re.sub(pattern, replacement, new_name)

    return new_name

def _classify_and_map_key(key: str) -> Optional[Tuple[str, str, Optional[str], str]]:
    """
    Efficiently classifies a LoRA key using the centralized KEY_MAPPING.
    The implementation is new and optimized, but the name and signature are preserved.
    """
    k = key
    if k.startswith("transformer."):
        k = k[len("transformer."):]
    if k.startswith("diffusion_model."):
        k = k[len("diffusion_model."):]
    if k.startswith("lora_unet_"):
        k = k[len("lora_unet_"):]
        k = _rename_layer_underscore_layer_name(k)


    base = None
    ab = None

    m = _RE_LORA_SUFFIX.search(k)
    if m:
        tag = m.group("tag")
        base = k[: m.start()]
        if "lora_A" in tag or tag.endswith(".A") or "down" in tag:
            ab = "A"
        elif "lora_B" in tag or tag.endswith(".B") or "up" in tag:
            ab = "B"
    else:
        # Check for LoKR format (lokr_w1, lokr_w2)
        m = _RE_LOKR_SUFFIX.search(k)
        if m:
            tag = m.group("tag")
            base = k[: m.start()]
            if tag == "lokr_w1":
                ab = "lokr_w1"
            elif tag == "lokr_w2":
                ab = "lokr_w2"
        else:
            m = _RE_ALPHA_SUFFIX.search(k)
        if m:
            ab = "alpha"
            base = k[: m.start()]

    if base is None or ab is None:
        return None  # Not a recognized LoRA key format

    for pattern, template, group, comp_fn in KEY_MAPPING:
        match = pattern.match(base)
        if match:
            final_key = match.expand(template)
            component = comp_fn(match) if comp_fn else None
            return group, final_key, component, ab

    return None


def _is_indexable_module(m):
    """Checks if a module is a list-like container."""
    return isinstance(m, (nn.ModuleList, nn.Sequential, list, tuple))


def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Traverse a path like 'a.b.3.c' to find and return a module."""
    if not name: return model
    module = model
    for part in name.split("."):
        if not part: continue

        # Prioritize hasattr check. This works for:
        # 1. Regular attributes ('attn', 'img_mod')
        # 2. Numerically-named children in nn.Sequential/nn.ModuleDict ('0', '1', '2')
        if hasattr(module, part):
            module = getattr(module, part)
        # Fallback to indexing for ModuleList (which fails hasattr for numeric keys)
        elif part.isdigit() and _is_indexable_module(module):
            try:
                module = module[int(part)]
            except (IndexError, TypeError):
                logger.warning(f"Failed to index module {name} with part {part}")
                return None
        # All attempts failed
        else:
            return None
    return module

def _resolve_module_name(model: nn.Module, name: str) -> Tuple[str, Optional[nn.Module]]:
    """Resolve a name string path to a module, attempting fallback paths."""
    m = _get_module_by_name(model, name)
    if m is not None:
        return name, m

    if name.endswith(".attn.to_out.0"):
        alt = name[:-2]
        m = _get_module_by_name(model, alt)
        if m is not None: return alt, m
    elif name.endswith(".attn.to_out"):
        alt = name + ".0"
        m = _get_module_by_name(model, alt)
        if m is not None: return alt, m

    mapping = {
        ".ff.net.0.proj": ".mlp_fc1", ".ff.net.2": ".mlp_fc2",
        ".ff_context.net.0.proj": ".mlp_context_fc1", ".ff_context.net.2": ".mlp_context_fc2",
    }
    for src, dst in mapping.items():
        if src in name:
            alt = name.replace(src, dst)
            m = _get_module_by_name(model, alt)
            if m is not None: return alt, m

    logger.debug(f"[MISS] Module not found: {name}")
    return name, None


def _load_lora_state_dict(lora_state_dict_or_path: Union[str, Path, Dict[str, torch.Tensor]]) -> Dict[
    str, torch.Tensor]:
    """Load LoRA state dict from path or return existing dict."""
    if isinstance(lora_state_dict_or_path, (str, Path)):
        path = Path(lora_state_dict_or_path)
        if path.suffix == ".safetensors":
            state_dict = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            return state_dict
        else:
            return torch.load(path, map_location="cpu")
    return lora_state_dict_or_path


def _convert_lokr_to_lora(lokr_weights: Dict[str, torch.Tensor], module_in_features: Optional[int] = None, module_out_features: Optional[int] = None) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert LoKR format (lokr_w1, lokr_w2) to standard LoRA format (A, B).
    
    Args:
        lokr_weights: Dictionary containing lokr_w1, lokr_w2, and optionally alpha
        module_in_features: Optional module in_features for Strategy 5 (square matrix case)
        module_out_features: Optional module out_features for Strategy 5b (non-square w2 case)
    """
    w1 = lokr_weights.get("lokr_w1")
    w2 = lokr_weights.get("lokr_w2")
    
    if w1 is None or w2 is None:
        return None, None, None
    
    # FACT: Keep original references for device casting
    original_w1 = w1
    original_w2 = w2
    
    # LoKR uses Kronecker product approximation: ΔW = w2 @ w1
    # Standard LoRA uses: ΔW = B @ A
    # We need to convert w2 @ w1 to B @ A format
    
    # w1 and w2 shapes can vary, handle common cases
    # Typically: w1 is [rank1, in_features] or [in_features, rank1]
    #            w2 is [out_features, rank2] or [rank2, out_features]
    
    # Ensure w1 and w2 are 2D
    if w1.ndim != 2 or w2.ndim != 2:
        logger.warning(f"LoKR weights must be 2D, got w1.shape={w1.shape}, w2.shape={w2.shape}")
        return None, None, None
    
    # Standard LoRA format: A is [rank, in_features], B is [out_features, rank]
    # For w2 @ w1 to equal B @ A, we need:
    # - w2 @ w1 dimensions: [out_features, rank2] @ [rank1, in_features] = [out_features, in_features]
    # - B @ A dimensions: [out_features, rank] @ [rank, in_features] = [out_features, in_features]
    # So we need rank1 == rank2 == rank
    
    # Check if w2 @ w1 is valid
    # Standard LoRA format: A is [rank, in_features], B is [out_features, rank]
    # For w2 @ w1 to equal B @ A, we need: w2 @ w1 dimensions to match
    # w2 @ w1: [out_features, rank2] @ [rank1, in_features] = [out_features, in_features]
    
    # Try to align dimensions by transposing if needed (use clones to avoid modifying originals)
    w1_aligned = w1.clone() if isinstance(w1, torch.Tensor) else w1
    w2_aligned = w2.clone() if isinstance(w2, torch.Tensor) else w2
    
    # Standard LoRA format: A is [rank, in_features], B is [out_features, rank]
    # For w2 @ w1 to equal B @ A, we need: w2.shape[1] == w1.shape[0]
    
    # Try different combinations to align dimensions
    original_w1, original_w2 = w1_aligned, w2_aligned
    
    # Strategy 1: Try if w2 @ w1 is already valid
    if w2_aligned.shape[1] == w1_aligned.shape[0]:
        pass  # Already aligned
    # Strategy 2: w2 might be transposed: [rank, out_features] -> [out_features, rank]
    elif w2_aligned.shape[0] == w1_aligned.shape[0]:
        w2_aligned = w2_aligned.T
    # Strategy 3: w1 might be transposed: [in_features, rank] -> [rank, in_features]
    elif w1_aligned.shape[1] == w2_aligned.shape[1]:
        w1_aligned = w1_aligned.T
    # Strategy 4: Both might need transposing
    elif w2_aligned.shape[0] == w1_aligned.shape[1]:
        w2_aligned = w2_aligned.T
        w1_aligned = w1_aligned.T
    # Strategy 5: Special case for square matrices (e.g., SNOFs format)
    # FACT-BASED: Requires actual module_in_features, no estimation
    # Handles:
    # - Case 5a: w2 is [out, out] and w1 is [rank, rank] where out != rank (both square)
    # - Case 5b: w2 is [out_features, larger_dim] and w1 is [rank, rank] (w1 square, w2 not)
    elif (w2_aligned.shape[0] == w2_aligned.shape[1] and w1_aligned.shape[0] == w1_aligned.shape[1]) or \
         (w1_aligned.shape[0] == w1_aligned.shape[1] and w2_aligned.shape[1] != w1_aligned.shape[0] and 
          w2_aligned.shape[0] != w1_aligned.shape[0] and module_in_features is not None):
        # FACT: w1 is square matrix [rank, rank]
        rank_dim = w1_aligned.shape[0]  # FACT: w1 is [rank, rank]
        
        # Determine if w2 is square or not
        is_w2_square = w2_aligned.shape[0] == w2_aligned.shape[1]
        
        if is_w2_square:
            out_dim = w2_aligned.shape[0]    # FACT: w2 is [out, out]
        else:
            out_dim = w2_aligned.shape[0]    # FACT: w2 is [out_features, larger_dim]
            w2_cols = w2_aligned.shape[1]    # FACT: w2 has more columns than rank
        
        # FACT: module_in_features is REQUIRED for Strategy 5 - no conversion without it
        if module_in_features is None:
            logger.debug(f"LoKR Strategy 5 requires module_in_features (w2.shape={original_w2.shape}, w1.shape={original_w1.shape})")
            return None, None, None
        
        if is_w2_square and out_dim >= rank_dim:
            # FACT: Extract [out_features, rank] from w2 by taking first rank_dim columns
            w2_aligned = w2_aligned[:, :rank_dim]  # [out_features, rank]
            
            # FACT: We need to convert w1=[rank, rank] to A=[rank, in_features]
            # FACT: Use SVD on w2 to decompose it: w2 = U @ S @ V^T
            # FACT: This gives us information about how to expand w1
            try:
                # FACT: SVD requires float32 (BFloat16 not supported on CPU)
                # Convert to float32 for SVD, then convert back to original dtype
                original_dtype = original_w2.dtype
                original_device = original_w2.device
                
                w2_f32 = original_w2.float() if original_dtype != torch.float32 else original_w2
                w1_f32 = w1_aligned.float() if original_dtype != torch.float32 else w1_aligned
                
                # FACT: SVD decomposition of original w2
                U, S, Vh = torch.linalg.svd(w2_f32, full_matrices=False)
                # FACT: Extract top rank components
                U_rank = U[:, :rank_dim]  # [out, rank]
                S_rank = S[:rank_dim]  # [rank]
                Vh_rank = Vh[:rank_dim, :]  # [rank, out]
                
                # FACT: Use sqrt of singular values for decomposition
                sqrt_S = torch.sqrt(S_rank + 1e-8)  # [rank]
                
                # FACT: Build B = [out_features, rank] from U and sqrt(S)
                w2_temp = U_rank @ torch.diag(sqrt_S)  # [out, rank]
                
                # FACT: If module_out_features is provided and different, adjust w2
                if module_out_features is not None and w2_temp.shape[0] != module_out_features:
                    if module_out_features > w2_temp.shape[0]:
                        # Pad with zeros
                        pad_size = module_out_features - w2_temp.shape[0]
                        w2_aligned = torch.cat([w2_temp, torch.zeros(pad_size, rank_dim, dtype=w2_temp.dtype, device=w2_temp.device)], dim=0)
                        logger.debug(f"LoKR Strategy 5a: Expanded w2 from {w2_temp.shape} to {w2_aligned.shape} to match module_out_features={module_out_features}")
                    else:
                        # Truncate
                        w2_aligned = w2_temp[:module_out_features, :]
                        logger.debug(f"LoKR Strategy 5a: Truncated w2 from {w2_temp.shape} to {w2_aligned.shape} to match module_out_features={module_out_features}")
                else:
                    w2_aligned = w2_temp
                
                # FACT: For A, we need [rank, in_features]
                # FACT: Use SVD components to expand w1: w1 @ (sqrt(S) @ Vh^T) gives [rank, out]
                A_svd = torch.diag(sqrt_S) @ Vh_rank  # [rank, out]
                w1_expanded = w1_f32 @ A_svd  # [rank, rank] @ [rank, out] = [rank, out]
                
                # Convert back to original dtype and device
                w2_aligned = w2_aligned.to(dtype=original_dtype, device=original_device)
                w1_expanded = w1_expanded.to(dtype=original_dtype, device=original_device)
                
                # FACT: Now we have [rank, out] but need [rank, in_features]
                # FACT: If in_features == out, use directly
                if module_in_features == out_dim:
                    w1_aligned = w1_expanded  # [rank, in_features] where in_features = out
                elif module_in_features > out_dim:
                    # FACT: Need to expand from [rank, out] to [rank, in_features] where in_features > out
                    # FACT: Use Kronecker product approach or zero padding
                    # For now, use zero padding as a fallback (may need refinement)
                    pad_size = module_in_features - out_dim
                    w1_aligned = torch.cat([w1_expanded, torch.zeros(rank_dim, pad_size, dtype=w1_expanded.dtype, device=w1_expanded.device)], dim=1)
                    logger.debug(f"LoKR Strategy 5: Expanded w1 from [rank, out]={w1_expanded.shape} to [rank, in_features]={w1_aligned.shape} using zero padding")
                else:
                    # FACT: in_features < out (unusual but possible) - truncate
                    w1_aligned = w1_expanded[:, :module_in_features]
                    logger.debug(f"LoKR Strategy 5: Truncated w1 from [rank, out]={w1_expanded.shape} to [rank, in_features]={w1_aligned.shape}")
            except Exception as e:
                logger.warning(f"LoKR SVD decomposition failed for Strategy 5: {e}, cannot convert")
                return None, None, None
        elif not is_w2_square:
            # Case 5b: w2 is not square [out_features, larger_dim], w1 is [rank, rank]
            # Need to reduce w2 to [out_features, rank] and expand w1 to [rank, in_features]
            try:
                # FACT: SVD requires float32 (BFloat16 not supported on CPU)
                original_dtype = original_w2.dtype
                original_device = original_w2.device
                
                w2_f32 = original_w2.float() if original_dtype != torch.float32 else original_w2
                w1_f32 = w1_aligned.float() if original_dtype != torch.float32 else w1_aligned
                
                # FACT: Standard LoKR conversion approach for non-square w2
                # w2 = [out_features, larger_dim], w1 = [rank, rank]
                # Goal: Convert to B = [out_features, rank], A = [rank, in_features]
                # 
                # Standard approach: Use SVD on w2^T to get decomposition
                # w2^T = U @ S @ Vh^T where U is [larger_dim, rank], S is [rank], Vh is [rank, out_features]
                # Then: B = Vh^T, and use w1 to expand to A
                
                # Use SVD on w2^T to decompose: [larger_dim, out_features] = U @ S @ Vh^T
                w2_T_f32 = w2_f32.T  # [larger_dim, out_features]
                U, S, Vh = torch.linalg.svd(w2_T_f32, full_matrices=False)
                
                # FACT: Build B = [out_features, rank] from Vh
                # Take top rank components
                Vh_rank = Vh[:rank_dim, :]  # [rank, out_features]
                S_rank = S[:rank_dim]  # [rank]
                
                # B = Vh^T with scaling: [out_features, rank]
                sqrt_S = torch.sqrt(S_rank + 1e-8)  # [rank]
                w2_temp = (Vh_rank.T) @ torch.diag(sqrt_S)  # [out_features, rank] @ [rank, rank] = [out_features, rank]
                
                # FACT: If module_out_features is provided and different, adjust w2
                if module_out_features is not None and w2_temp.shape[0] != module_out_features:
                    if module_out_features > w2_temp.shape[0]:
                        # Pad with zeros
                        pad_size = module_out_features - w2_temp.shape[0]
                        w2_aligned = torch.cat([w2_temp, torch.zeros(pad_size, rank_dim, dtype=w2_temp.dtype, device=w2_temp.device)], dim=0)
                        logger.debug(f"LoKR Strategy 5b: Expanded w2 from {w2_temp.shape} to {w2_aligned.shape} to match module_out_features={module_out_features}")
                    else:
                        # Truncate
                        w2_aligned = w2_temp[:module_out_features, :]
                        logger.debug(f"LoKR Strategy 5b: Truncated w2 from {w2_temp.shape} to {w2_aligned.shape} to match module_out_features={module_out_features}")
                else:
                    w2_aligned = w2_temp
                
                # FACT: Expand w1 from [rank, rank] to [rank, in_features]
                # Use w1 @ (U @ sqrt(S)) to expand: [rank, rank] @ [larger_dim, rank]^T = [rank, larger_dim]
                U_rank = U[:, :rank_dim]  # [larger_dim, rank]
                w1_expanded_via_u = w1_f32 @ (U_rank @ torch.diag(sqrt_S)).T  # [rank, rank] @ [rank, larger_dim] = [rank, larger_dim]
                
                # Now expand from larger_dim to in_features
                if module_in_features == w2_cols:
                    w1_aligned = w1_expanded_via_u  # [rank, in_features] where in_features = larger_dim
                elif module_in_features > w2_cols:
                    # Pad with zeros (standard approach for dimension expansion)
                    pad_size = module_in_features - w2_cols
                    w1_aligned = torch.cat([w1_expanded_via_u, torch.zeros(rank_dim, pad_size, dtype=w1_expanded_via_u.dtype, device=w1_expanded_via_u.device)], dim=1)
                    logger.debug(f"LoKR Strategy 5b: Expanded w1 from [rank, larger_dim]={w1_expanded_via_u.shape} to [rank, in_features]={w1_aligned.shape} using zero padding")
                else:
                    # Truncate
                    w1_aligned = w1_expanded_via_u[:, :module_in_features]
                    logger.debug(f"LoKR Strategy 5b: Truncated w1 from [rank, larger_dim]={w1_expanded_via_u.shape} to [rank, in_features]={w1_aligned.shape}")
                
                # Convert back to original dtype and device
                w2_aligned = w2_aligned.to(dtype=original_dtype, device=original_device)
                w1_aligned = w1_aligned.to(dtype=original_dtype, device=original_device)
            except Exception as e:
                logger.warning(f"LoKR SVD decomposition failed for Strategy 5b: {e}, cannot convert")
                return None, None, None
        else:
            logger.warning(f"LoKR Strategy 5: w2 smaller than w1. w2.shape={original_w2.shape}, w1.shape={original_w1.shape}")
            return None, None, None
    # Strategy 6: Both w1 and w2 are non-square, but module information is available
    # This handles cases like w2=[2048, 384], w1=[9, 8] where standard strategies fail
    # FACT: For non-standard LoKR, we compute the Kronecker product and decompose using SVD
    # This is the only way to preserve the full interaction between w1 and w2
    elif module_in_features is not None and module_out_features is not None:
        # FACT: Check if Kronecker product dimensions match the target module
        # Try kron(w1, w2) which is standard for many implementations (e.g. Lycoris)
        # Shape would be [w1.shape[0]*w2.shape[0], w1.shape[1]*w2.shape[1]]
        k_rows_12 = w1.shape[0] * w2.shape[0]
        k_cols_12 = w1.shape[1] * w2.shape[1]
        
        # Try kron(w2, w1)
        k_rows_21 = w2.shape[0] * w1.shape[0]
        k_cols_21 = w2.shape[1] * w1.shape[1]
        
        match_12 = (k_rows_12 == module_out_features and k_cols_12 == module_in_features)
        match_21 = (k_rows_21 == module_out_features and k_cols_21 == module_in_features)
        
        if not match_12 and not match_21:
             # Try transposed matches (if weights are for [in, out])
             match_12_T = (k_cols_12 == module_out_features and k_rows_12 == module_in_features)
             match_21_T = (k_cols_21 == module_out_features and k_rows_21 == module_in_features)
             
             if not match_12_T and not match_21_T:
                 logger.warning(f"LoKR Strategy 6: Kron product dimensions mismatch module. w1={w1.shape}, w2={w2.shape}, module=({module_out_features}, {module_in_features})")
                 return None, None, None

        # Determine which order to use
        use_w2_first = False
        transpose_result = False
        
        if match_21:
            use_w2_first = True
        elif match_12:
            use_w2_first = False
        elif match_21_T:
            use_w2_first = True
            transpose_result = True
        elif match_12_T:
            use_w2_first = False
            transpose_result = True
            
        # Perform construction and SVD
        # Move to CPU and float32 for stability and memory
        w1_cpu = w1.to(device="cpu", dtype=torch.float32)
        w2_cpu = w2.to(device="cpu", dtype=torch.float32)
        
        try:
            if use_w2_first:
                # kron(w2, w1)
                full_W = torch.kron(w2_cpu, w1_cpu)
            else:
                # kron(w1, w2)
                full_W = torch.kron(w1_cpu, w2_cpu)
            
            if transpose_result:
                full_W = full_W.T
                
            # SVD
            # Choose rank: large enough to capture info, small enough for efficiency
            # Standard Qwen LoRA rank is often 16-64. LoKR can have high effective rank.
            # Let's use a safe rank like 64, or min(dim)/4, capped at 128.
            target_rank = min(full_W.shape)
            target_rank = min(target_rank, 64) # Cap at 64 for now to match typical LoRA sizes
            
            U, S, Vh = torch.linalg.svd(full_W, full_matrices=False)
            
            U_k = U[:, :target_rank]
            S_k = S[:target_rank]
            Vh_k = Vh[:target_rank, :]
            
            # Decompose into B @ A
            # B = U * sqrt(S)
            # A = sqrt(S) * Vh
            sqrt_S = torch.sqrt(S_k)
            B_cpu = U_k @ torch.diag(sqrt_S)
            A_cpu = torch.diag(sqrt_S) @ Vh_k
            
            # Cast back
            w2_aligned = B_cpu.to(device=original_w2.device, dtype=original_w2.dtype)
            w1_aligned = A_cpu.to(device=original_w1.device, dtype=original_w1.dtype)
            
            logger.debug(f"LoKR Strategy 6: SVD decomposition successful. Rank={target_rank}. Shape {full_W.shape} -> B={w2_aligned.shape}, A={w1_aligned.shape}")
            
        except Exception as e:
            logger.warning(f"LoKR Strategy 6: SVD failed: {e}")
            return None, None, None
            
    else:
        logger.warning(f"LoKR dimension mismatch: w2.shape={original_w2.shape}, w1.shape={original_w1.shape}. Module information required for conversion.")
        return None, None, None
    
    # Verify final dimensions are correct
    if w2_aligned.shape[1] != w1_aligned.shape[0]:
        logger.warning(f"LoKR dimension alignment failed: w2.shape={w2_aligned.shape}, w1.shape={w1_aligned.shape}")
        return None, None, None
    
    # After alignment, w2 should be [out_features, rank] and w1 should be [rank, in_features]
    # So A = w1, B = w2
    A = w1_aligned  # [rank, in_features]
    B = w2_aligned  # [out_features, rank]
    
    alpha = lokr_weights.get("alpha")
    
    return A, B, alpha


def _fuse_qkv_lora(qkv_weights: Dict[str, torch.Tensor], model: Optional[nn.Module] = None, base_key: Optional[str] = None) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fuse Q/K/V LoRA weights into a single QKV tensor.
    
    Args:
        qkv_weights: Dictionary containing Q/K/V LoRA weights
        model: Optional model instance to inspect module structure
        base_key: Optional base key to resolve module for Strategy 5
    """
    # Check if this is LoKR format for QKV
    q_lokr = qkv_weights.get("Q_lokr_w1") is not None and qkv_weights.get("Q_lokr_w2") is not None
    k_lokr = qkv_weights.get("K_lokr_w1") is not None and qkv_weights.get("K_lokr_w2") is not None
    v_lokr = qkv_weights.get("V_lokr_w1") is not None and qkv_weights.get("V_lokr_w2") is not None
    
    # For Strategy 5 (square matrix case), get actual module structure (FACT-BASED, NO ESTIMATION)
    qkv_in_features = None
    qkv_out_features_per_component = None
    if (q_lokr or k_lokr or v_lokr) and model is not None and base_key is not None:
        # Get actual QKV module to inspect its structure
        resolved_name, module = _resolve_module_name(model, base_key)
        if module is not None:
            if hasattr(module, 'in_features'):
                # FACT: For QKV modules, each Q/K/V shares the same in_features as the unified module
                # FACT: Confirmed by error logs: add_qkv_proj has in_features=3072
                # Use the actual module.in_features value (NO ESTIMATION)
                qkv_in_features = module.in_features
            if hasattr(module, 'out_features'):
                # FACT: QKV module has combined out_features = 3 * individual_out_features
                # FACT: Each Q/K/V has out_features = module.out_features / 3
                # Example: module.out_features=9216 means each Q/K/V has out_features=3072
                qkv_out_features_per_component = module.out_features // 3
            logger.debug(f"QKV module {resolved_name}: in_features={qkv_in_features}, out_features_per_component={qkv_out_features_per_component}")
    
    if q_lokr or k_lokr or v_lokr:
        # --- EXPERIMENTAL LoKR CODE START ---
        # The following code block is for experimental LoKR support.
        # Currently, it does not work correctly with Nunchaku and produces noise.
        # It has been disabled to prevent issues.
        """
        q_lora = _convert_lokr_to_lora({
            "lokr_w1": qkv_weights.get("Q_lokr_w1"),
            "lokr_w2": qkv_weights.get("Q_lokr_w2"),
            "alpha": qkv_weights.get("Q_alpha")
        }, module_in_features=qkv_in_features, module_out_features=qkv_out_features_per_component) if q_lokr else (qkv_weights.get("Q_A"), qkv_weights.get("Q_B"), qkv_weights.get("Q_alpha"))
        
        k_lora = _convert_lokr_to_lora({
            "lokr_w1": qkv_weights.get("K_lokr_w1"),
            "lokr_w2": qkv_weights.get("K_lokr_w2"),
            "alpha": qkv_weights.get("K_alpha")
        }, module_in_features=qkv_in_features, module_out_features=qkv_out_features_per_component) if k_lokr else (qkv_weights.get("K_A"), qkv_weights.get("K_B"), qkv_weights.get("K_alpha"))
        
        v_lora = _convert_lokr_to_lora({
            "lokr_w1": qkv_weights.get("V_lokr_w1"),
            "lokr_w2": qkv_weights.get("V_lokr_w2"),
            "alpha": qkv_weights.get("V_alpha")
        }, module_in_features=qkv_in_features, module_out_features=qkv_out_features_per_component) if v_lokr else (qkv_weights.get("V_A"), qkv_weights.get("V_B"), qkv_weights.get("V_alpha"))
        
        A_q, B_q, alpha_q = q_lora
        A_k, B_k, alpha_k = k_lora
        A_v, B_v, alpha_v = v_lora
        
        if not all([A_q is not None, B_q is not None, A_k is not None, B_k is not None, A_v is not None, B_v is not None]):
            return None, None, None
        """
        logger.warning(f"Skipping LoKR weights for QKV: LoKR support is currently experimental and disabled due to compatibility issues (produces noise).")
        return None, None, None
        # --- EXPERIMENTAL LoKR CODE END ---
        
        # Continue with standard QKV fusion
    else:
        # Standard LoRA format
        required_keys = ["Q_A", "Q_B", "K_A", "K_B", "V_A", "V_B"]
        if not all(k in qkv_weights for k in required_keys):
            return None, None, None

        A_q, A_k, A_v = qkv_weights["Q_A"], qkv_weights["K_A"], qkv_weights["V_A"]
        B_q, B_k, B_v = qkv_weights["Q_B"], qkv_weights["K_B"], qkv_weights["V_B"]
        
        alpha_q, alpha_k, alpha_v = qkv_weights.get("Q_alpha"), qkv_weights.get("K_alpha"), qkv_weights.get("V_alpha")

    if not (A_q.shape == A_k.shape == A_v.shape):
        logger.warning(f"Q/K/V LoRA A dimensions mismatch: {A_q.shape}, {A_k.shape}, {A_v.shape}")
        return None, None, None

    # FACT: Check if B dimensions are consistent
    if not (B_q.shape[1] == B_k.shape[1] == B_v.shape[1]):
        logger.warning(f"Q/K/V LoRA B rank mismatch: {B_q.shape[1]}, {B_k.shape[1]}, {B_v.shape[1]}")
        return None, None, None
    
    # FACT: For QKV modules, get actual module out_features to ensure correct fusion
    qkv_out_features = None
    if model is not None and base_key is not None:
        resolved_name, module = _resolve_module_name(model, base_key)
        if module is not None and hasattr(module, 'out_features'):
            # FACT: QKV module has combined out_features = 3 * individual_out_features
            qkv_out_features = module.out_features
            logger.debug(f"QKV module {resolved_name}: using actual module out_features={qkv_out_features} for fusion")

    alpha_fused = None
    if alpha_q is not None and alpha_k is not None and alpha_v is not None and (
            alpha_q.item() == alpha_k.item() == alpha_v.item()):
        alpha_fused = alpha_q

    A_fused = torch.cat([A_q, A_k, A_v], dim=0)

    r = B_q.shape[1]
    
    # FACT: For QKV fusion, use actual module out_features if available
    # Otherwise, use sum of individual out_features (standard case)
    if qkv_out_features is not None:
        # FACT: QKV module has out_features = 3 * individual_out_features
        # Each Q/K/V should have out_features = qkv_out_features / 3
        expected_out_per_component = qkv_out_features // 3
        out_q, out_k, out_v = B_q.shape[0], B_k.shape[0], B_v.shape[0]
        
        # Verify or adjust individual B shapes if needed
        if out_q == expected_out_per_component and out_k == expected_out_per_component and out_v == expected_out_per_component:
            # All correct, use module out_features for fusion
            B_fused = torch.zeros(qkv_out_features, 3 * r, dtype=B_q.dtype, device=B_q.device)
            B_fused[:out_q, :r] = B_q
            B_fused[out_q: out_q + out_k, r: 2 * r] = B_k
            B_fused[out_q + out_k:, 2 * r:] = B_v
        else:
            # B shapes don't match expected - log warning and use standard fusion
            logger.warning(f"Q/K/V B out_features mismatch: Q={out_q}, K={out_k}, V={out_v}, expected={expected_out_per_component} each. Using actual shapes.")
            B_fused = torch.zeros(out_q + out_k + out_v, 3 * r, dtype=B_q.dtype, device=B_q.device)
            B_fused[:out_q, :r] = B_q
            B_fused[out_q: out_q + out_k, r: 2 * r] = B_k
            B_fused[out_q + out_k:, 2 * r:] = B_v
    else:
        # Standard fusion when module info not available
        out_q, out_k, out_v = B_q.shape[0], B_k.shape[0], B_v.shape[0]
        B_fused = torch.zeros(out_q + out_k + out_v, 3 * r, dtype=B_q.dtype, device=B_q.device)
        B_fused[:out_q, :r] = B_q
        B_fused[out_q: out_q + out_k, r: 2 * r] = B_k
        B_fused[out_q + out_k:, 2 * r:] = B_v

    return A_fused, B_fused, alpha_fused


def _handle_proj_out_split(lora_dict: Dict[str, Dict[str, torch.Tensor]], base_key: str, model: nn.Module) -> Tuple[
    Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]], List[str]]:
    """Split single-block proj_out LoRA into two branches."""
    result, consumed = {}, []
    m = re.search(r"single_transformer_blocks\.(\d+)", base_key)
    if not m or base_key not in lora_dict:
        return result, consumed

    block_idx = m.group(1)
    block = _get_module_by_name(model, f"single_transformer_blocks.{block_idx}")
    if block is None:
        return result, consumed

    A_full, B_full, alpha = lora_dict[base_key].get("A"), lora_dict[base_key].get("B"), lora_dict[base_key].get("alpha")
    if A_full is None or B_full is None:
        return result, consumed

    attn_to_out = getattr(getattr(block, "attn", None), "to_out", None)
    mlp_fc2 = getattr(block, "mlp_fc2", None)
    if attn_to_out is None or mlp_fc2 is None or not hasattr(attn_to_out, "in_features") or not hasattr(mlp_fc2,
                                                                                                        "in_features"):
        return result, consumed

    attn_in, mlp_in = attn_to_out.in_features, mlp_fc2.in_features
    if A_full.shape[1] != attn_in + mlp_in:
        logger.warning(f"{base_key}: A_full shape mismatch {A_full.shape} vs expected in_features {attn_in + mlp_in}")
        return result, consumed

    A_attn, A_mlp = A_full[:, :attn_in], A_full[:, attn_in:]
    result[f"single_transformer_blocks.{block_idx}.attn.to_out"] = (A_attn, B_full.clone(), alpha)
    result[f"single_transformer_blocks.{block_idx}.mlp_fc2"] = (A_mlp, B_full.clone(), alpha)
    consumed.append(base_key)
    return result, consumed


def _apply_lora_to_module(module: nn.Module, A: torch.Tensor, B: torch.Tensor, module_name: str,
                          model: nn.Module) -> None:
    """Helper to append combined LoRA weights to a module."""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"{module_name}: A/B must be 2D, got {A.shape}, {B.shape}")
    # FACT: A must match module.in_features exactly (no padding/estimation)
    if A.shape[1] != module.in_features:
        raise ValueError(f"{module_name}: A shape {A.shape} mismatch with in_features={module.in_features}")
    if B.shape[0] != module.out_features:
        raise ValueError(f"{module_name}: B shape {B.shape} mismatch with out_features={module.out_features}")

    pd, pu = module.proj_down.data, module.proj_up.data
    pd = unpack_lowrank_weight(pd, down=True)
    pu = unpack_lowrank_weight(pu, down=False)

    base_rank = pd.shape[0] if pd.shape[1] == module.in_features else pd.shape[1]

    if pd.shape[1] == module.in_features:  # [rank, in]
        new_proj_down = torch.cat([pd, A], dim=0)
        axis_down = 0
    else:  # [in, rank]
        new_proj_down = torch.cat([pd, A.T], dim=1)
        axis_down = 1

    new_proj_up = torch.cat([pu, B], dim=1)

    module.proj_down.data = pack_lowrank_weight(new_proj_down, down=True)
    module.proj_up.data = pack_lowrank_weight(new_proj_up, down=False)
    module.rank = base_rank + A.shape[0]

    if not hasattr(model, "_lora_slots"):
        model._lora_slots = {}
    slot = model._lora_slots.setdefault(module_name, {"base_rank": base_rank, "appended": 0, "axis_down": axis_down})
    slot["appended"] += A.shape[0]


# --- Main Public API ---

def compose_loras_v2(
        model: torch.nn.Module,
        lora_configs: List[Tuple[Union[str, Path, Dict[str, torch.Tensor]], float]],
) -> None:
    """
    Resets and composes multiple LoRAs into the model with individual strengths.
    """
    logger.info(f"Composing {len(lora_configs)} LoRAs...")
    reset_lora_v2(model)

    aggregated_weights: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # 1. Aggregate weights from all LoRAs
    for lora_path_or_dict, strength in lora_configs:
        lora_name = lora_path_or_dict if isinstance(lora_path_or_dict, str) else "dict"
        lora_state_dict = _load_lora_state_dict(lora_path_or_dict)

        lora_grouped: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
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
                
            if group in ("qkv", "add_qkv") and comp is not None:
                # Handle both standard LoRA (A/B) and LoKR (lokr_w1/lokr_w2) formats
                if ab in ("lokr_w1", "lokr_w2"):
                    lora_grouped[base_key][f"{comp}_{ab}"] = value
                else:
                lora_grouped[base_key][f"{comp}_{ab}"] = value
            else:
                lora_grouped[base_key][ab] = value

        # Detect and log LoRA format
        has_lokr = lokr_keys_count > 0
        has_standard = standard_keys_count > 0
        unrecognized_count = len(unrecognized_keys)
        
        if has_lokr and has_standard:
            lora_format = "Mixed (LoKR + Standard LoRA)"
        elif has_lokr:
            if qkv_lokr_keys_count > 0:
                lora_format = "LoKR (QKV format)"
            else:
                lora_format = "LoKR"
        elif has_standard:
            lora_format = "Standard LoRA"
        else:
            lora_format = "Unknown/Unsupported"
        
        logger.info(f"📦 LoRA: {lora_name} | Format: {lora_format} | Strength: {strength:.3f}")
        if has_lokr:
            logger.debug(f"   LoKR keys: {lokr_keys_count} (QKV: {qkv_lokr_keys_count}), Standard keys: {standard_keys_count}")
        
        # Warn about unrecognized keys
        if unrecognized_count > 0:
            unrecognized_ratio = unrecognized_count / total_keys * 100
            if unrecognized_ratio >= 50:
                logger.warning(f"⚠️  {lora_name}: {unrecognized_count}/{total_keys} keys ({unrecognized_ratio:.1f}%) are unrecognized/unsupported format!")
                # Show sample unrecognized keys
                sample_keys = unrecognized_keys[:5]
                for sample_key in sample_keys:
                    logger.warning(f"   Unrecognized key example: {sample_key}")
                if unrecognized_count > 5:
                    logger.warning(f"   ... and {unrecognized_count - 5} more unrecognized keys")
            else:
                logger.debug(f"   {unrecognized_count}/{total_keys} keys ({unrecognized_ratio:.1f}%) are unrecognized (likely metadata or non-LoRA keys)")
        
        # Warn if format is completely unknown
        if lora_format == "Unknown/Unsupported":
            logger.warning(f"⚠️  {lora_name}: No recognized LoRA format detected! All {total_keys} keys were unrecognized.")
            if unrecognized_keys:
                logger.warning(f"   Sample unrecognized keys:")
                for sample_key in unrecognized_keys[:10]:
                    logger.warning(f"     - {sample_key}")
                if len(unrecognized_keys) > 10:
                    logger.warning(f"     ... and {len(unrecognized_keys) - 10} more")

        # Process grouped weights for this LoRA
        processed_groups = {}
        special_handled = set()
        for base_key, lw in lora_grouped.items():
            if base_key in special_handled:
                continue

            # Check if this is LoKR format (lokr_w1, lokr_w2)
            if "lokr_w1" in lw or "lokr_w2" in lw:
                # --- EXPERIMENTAL LoKR CODE START ---
                # The following code block is for experimental LoKR support.
                # Currently, it does not work correctly with Nunchaku and produces noise.
                # It has been disabled to prevent issues.
                """
                # Try to get module info early for Strategy 5/6 (square matrix case and non-standard cases)
                resolved_name, module = _resolve_module_name(model, base_key)
                module_in_features = module.in_features if module is not None and hasattr(module, 'in_features') else None
                module_out_features = module.out_features if module is not None and hasattr(module, 'out_features') else None
                A, B, alpha = _convert_lokr_to_lora(lw, module_in_features=module_in_features, module_out_features=module_out_features)
                if A is not None and B is not None:
                    processed_groups[base_key] = (A, B, alpha)
                else:
                    logger.warning(f"Failed to convert LoKR weights for {base_key}, skipping")
                """
                logger.warning(f"Skipping LoKR weights for {base_key}: LoKR support is currently experimental and disabled due to compatibility issues (produces noise). Please convert LoKR to standard LoRA first.")
                continue
                # --- EXPERIMENTAL LoKR CODE END ---

            if "qkv" in base_key:
                # Pass model and base_key to _fuse_qkv_lora for actual module inspection
                A, B, alpha = (lw.get("A"), lw.get("B"), lw.get("alpha")) if "A" in lw else _fuse_qkv_lora(lw, model=model, base_key=base_key)
            elif ".proj_out" in base_key and "single_transformer_blocks" in base_key:
                split_map, consumed_keys = _handle_proj_out_split(lora_grouped, base_key, model)
                processed_groups.update(split_map)
                special_handled.update(consumed_keys)
                continue
            else:
                A, B, alpha = lw.get("A"), lw.get("B"), lw.get("alpha")

            if A is not None and B is not None:
                processed_groups[base_key] = (A, B, alpha)

        # Warn if no weights were processed for this LoRA
        if not processed_groups:
            if lora_format == "Unknown/Unsupported":
                logger.error(f"❌ {lora_name}: No weights were processed - LoRA format is unsupported and will be skipped!")
            else:
                logger.warning(f"⚠️  {lora_name}: No weights were processed - this LoRA will have no effect!")
                # Debug: show what keys were grouped but not processed
                if lora_grouped:
                    logger.warning(f"   Debug: {len(lora_grouped)} base keys were grouped but none were processed:")
                    for base_key, lw in list(lora_grouped.items())[:10]:
                        keys_in_group = list(lw.keys())
                        logger.warning(f"     - {base_key}: keys={keys_in_group}")
                    if len(lora_grouped) > 10:
                        logger.warning(f"     ... and {len(lora_grouped) - 10} more grouped keys")
        else:
            logger.debug(f"   {lora_name}: Processed {len(processed_groups)} module groups")
        for module_key, (A, B, alpha) in processed_groups.items():
            aggregated_weights[module_key].append(
                {"A": A, "B": B, "alpha": alpha, "strength": strength, "source": lora_name})

    # 2. Apply aggregated weights to the model
    applied_modules_count = 0

    for module_name, parts in aggregated_weights.items():
        resolved_name, module = _resolve_module_name(model, module_name)
        if module is None:
            logger.debug(f"[MISS] Module not found: {module_name} (resolved: {resolved_name})")
            continue
        if not (hasattr(module, "proj_down") and hasattr(module, "proj_up")):
            logger.debug(f"[MISS] Module found but missing proj_down/proj_up: {resolved_name}")
            continue

        all_A = []
        all_B_scaled = []
        for part in parts:
            A, B, alpha, strength = part["A"], part["B"], part["alpha"], part["strength"]
            r_lora = A.shape[0]
            scale_alpha = alpha.item() if alpha is not None else float(r_lora)
            scale = strength * (scale_alpha / max(1.0, float(r_lora)))

            if ".norm1.linear" in resolved_name or ".norm1_context.linear" in resolved_name:
                B = reorder_adanorm_lora_up(B, splits=6)
            elif ".single_transformer_blocks." in resolved_name and ".norm.linear" in resolved_name:
                B = reorder_adanorm_lora_up(B, splits=3)

            all_A.append(A.to(dtype=module.proj_down.dtype, device=module.proj_down.device))
            all_B_scaled.append((B * scale).to(dtype=module.proj_up.dtype, device=module.proj_up.device))

        if not all_A:
            continue

        final_A = torch.cat(all_A, dim=0)
        final_B = torch.cat(all_B_scaled, dim=1)

        _apply_lora_to_module(module, final_A, final_B, resolved_name, model)
        applied_modules_count += 1

    total_loras = len(lora_configs)
    # Always output the existing log message
    logger.info(f"Applied LoRA compositions to {applied_modules_count} modules.")
    
    # Add additional error message if needed (but keep existing log)
    if total_loras > 0 and applied_modules_count == 0:
        logger.error(f"❌ No LoRA modules were applied! {total_loras} LoRA(s) were loaded but none matched the model structure.")
        logger.error("   This may indicate:")
        logger.error("   - Unsupported LoRA format (check format warnings above)")
        logger.error("   - LoRA for a different model architecture")
        logger.error("   - Corrupted or incompatible LoRA file(s)")

def update_lora_params_v2(
        model: torch.nn.Module,
        lora_state_dict_or_path: Union[str, Path, Dict[str, torch.Tensor]],
        strength: float = 1.0,
) -> None:
    """Loads and applies a single LoRA to the model (convenience wrapper)."""
    logger.info(f"Loading single LoRA with strength {strength}.")
    compose_loras_v2(model, [(lora_state_dict_or_path, strength)])


def set_lora_strength_v2(model: nn.Module, strength: float) -> None:
    """Adjusts the overall strength of all applied LoRAs as a global multiplier."""
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        logger.warning("No LoRA weights loaded, cannot set strength.")
        return

    old_strength = getattr(model, "_lora_strength", 1.0)
    scale_factor = strength / old_strength if old_strength != 0 else 0

    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None or info.get("appended", 0) <= 0:
            continue

        base_rank, appended = info["base_rank"], info["appended"]
        with torch.no_grad():
            module.proj_up.data[:, base_rank: base_rank + appended] *= scale_factor

    model._lora_strength = strength
    logger.info(f"LoRA global strength updated to {strength}.")


def reset_lora_v2(model: nn.Module) -> None:
    """Removes all appended LoRA weights from the model."""
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        return

    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None:
            continue

        base_rank = info["base_rank"]
        with torch.no_grad():
            pd = unpack_lowrank_weight(module.proj_down.data, down=True)
            pu = unpack_lowrank_weight(module.proj_up.data, down=False)

            if info.get("axis_down", 0) == 0:  # [rank, in]
                pd_reset = pd[:base_rank, :].clone()
            else:  # [in, rank]
                pd_reset = pd[:, :base_rank].clone()
            pu_reset = pu[:, :base_rank].clone()

            module.proj_down.data = pack_lowrank_weight(pd_reset, down=True)
            module.proj_up.data = pack_lowrank_weight(pu_reset, down=False)
            module.rank = base_rank

    model._lora_slots.clear()
    model._lora_strength = 1.0
    logger.info("All LoRA weights have been reset from the model.")