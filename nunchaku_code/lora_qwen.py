import functools
import logging
import os
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

# Safety switch:
# QwenImage's modulation linears (`img_mod.1` / `txt_mod.1`) are extremely sensitive because their
# output is reshaped into shift/scale/gate parameters. With AWQ quantization (AWQW4A16Linear),
# applying LoRA here often results in severe noise. Default to skipping these two layers.
# Users can override by setting env var `QWENIMAGE_LORA_APPLY_AWQ_MOD=1`.
_APPLY_AWQ_MOD = str(os.getenv("QWENIMAGE_LORA_APPLY_AWQ_MOD", "0")).strip().lower() in ("1", "true", "yes", "y", "on")


def _detect_lora_format(lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Detect LoRA formats based on key patterns.
    Returns a dict containing detected format flags and sample keys.

    Formats:
    - Standard LoRA: lora_up/lora_down, lora_A/lora_B, and dot variants.
    - LoKR (Lycoris): lokr_w1/lokr_w2
    - LoHa: hada_w1/hada_w2/hada_t1/hada_t2
    - IA3: ia3_w or '.ia3.' patterns
    """
    keys = list(lora_state_dict.keys())

    standard_patterns = (
        ".lora_up.weight",
        ".lora_down.weight",
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora.up.weight",
        ".lora.down.weight",
        ".lora.A.weight",
        ".lora.B.weight",
    )

    def _sample(match_fn, limit: int = 8) -> List[str]:
        out = []
        for k in keys:
            if match_fn(k):
                out.append(k)
                if len(out) >= limit:
                    break
        return out

    has_standard = any(p in k for k in keys for p in standard_patterns)
    has_lokr = any(".lokr_w1" in k or ".lokr_w2" in k for k in keys)
    has_loha = any(".hada_w1" in k or ".hada_w2" in k or ".hada_t1" in k or ".hada_t2" in k for k in keys)
    has_ia3 = any(".ia3." in k or ".ia3_w" in k or k.endswith(".ia3.weight") for k in keys)

    return {
        "has_standard": has_standard,
        "has_lokr": has_lokr,
        "has_loha": has_loha,
        "has_ia3": has_ia3,
        "sample_standard": _sample(lambda k: any(p in k for p in standard_patterns)),
        "sample_lokr": _sample(lambda k: ".lokr_w1" in k or ".lokr_w2" in k),
        "sample_loha": _sample(lambda k: ".hada_w1" in k or ".hada_w2" in k or ".hada_t1" in k or ".hada_t2" in k),
        "sample_ia3": _sample(lambda k: ".ia3." in k or ".ia3_w" in k or k.endswith(".ia3.weight")),
        "total_keys": len(keys),
    }


def _log_lora_format_detection(lora_name: str, detection: Dict[str, Any]) -> None:
    sep = "=" * 80
    logger.info(sep)
    logger.info(f"LoRA Format Detection: {lora_name}")
    logger.info(sep)
    logger.info("Detected Formats:")

    has_standard = detection["has_standard"]
    has_lokr = detection["has_lokr"]
    has_loha = detection["has_loha"]
    has_ia3 = detection["has_ia3"]

    if has_standard:
        logger.info("  ✅ Standard LoRA (Rank-Decomposed)")
    if has_lokr:
        logger.info("  ❌ LoKR (Lycoris) - Not Supported")
    if has_loha:
        logger.info("  ❌ LoHa - Not Supported")
    if has_ia3:
        logger.info("  ❌ IA3 - Not Supported")

    if not (has_standard or has_lokr or has_loha or has_ia3):
        logger.info("  ❌ Unknown/Unsupported (no known LoRA keys detected)")

    if has_standard:
        logger.info("")
        logger.info("✅ Standard LoRA Details:")
        logger.info("   Supported weight keys:")
        logger.info("   - lora_up.weight / lora_down.weight")
        logger.info("   - lora.up.weight / lora.down.weight")
        logger.info("   - lora_A.weight / lora_B.weight")
        logger.info("   - lora.A.weight / lora.B.weight")
        logger.info("   These are the standard formats produced by Kohya-ss, Diffusers, and most training scripts.")

    if has_lokr:
        logger.info("")
        logger.info("❌ LoKR (Lycoris) - Not Supported")
        logger.info("   Issue: LoRAs in LoKR format (created by Lycoris) are not supported.")
        logger.info("   Important Note: This limitation applies specifically to Nunchaku quantization models.")
        logger.info("   LoKR format LoRAs may work with standard (non-quantized) Qwen Image models, but this node is designed for Nunchaku models only.")
        logger.info("   LoKR weights are automatically skipped when detected (experimental conversion code is disabled).")
        logger.info("   Converting to Standard LoRA using SVD approximation (via external tools or scripts) has also been tested")
        logger.info("   and found to result in noise/artifacts when applied to Nunchaku quantization models.")
        logger.info("   Conclusion: At this time, we have not found a way to successfully apply LoKR weights to Nunchaku models.")
        logger.info("   Please use Standard LoRA formats.")
        sample = detection.get("sample_lokr") or []
        if sample:
            logger.info(f"   Sample LoKR keys found: {sample}")

    if has_loha:
        logger.info("")
        logger.info("❌ LoHa - Not Supported")
        logger.info("   Issue: LoRAs in LoHa format are not supported for Nunchaku quantization models in this loader.")
        logger.info("   Please convert LoHa to Standard LoRA format before use.")
        sample = detection.get("sample_loha") or []
        if sample:
            logger.info(f"   Sample LoHa keys found: {sample}")

    if has_ia3:
        logger.info("")
        logger.info("❌ IA3 - Not Supported")
        logger.info("   Issue: IA3 format is not supported for Nunchaku models in this loader.")
        logger.info("   Please use Standard LoRA formats.")
        sample = detection.get("sample_ia3") or []
        if sample:
            logger.info(f"   Sample IA3 keys found: {sample}")

    logger.info(sep)

# --- Centralized & Optimized Key Mapping ---
# This structure is faster to process and easier to maintain than a long if/elif chain.
# --- CORRECTED Centralized & Optimized Key Mapping ---
# --- Centralized & Optimized Key Mapping ---
# This version correctly tokenizes all module paths.

# Active mapping override (used for runtime model-structure switching, e.g. NextDiT).
_ACTIVE_KEY_MAPPING = None

# --- NextDiT (ComfyUI Lumina2) mappings for Z-Image-Turbo official loader ---
# These are required because NextDiT uses:
# - layers.N.attention.qkv / layers.N.attention.out
# - layers.N.feed_forward.w1/w2/w3 (unpatched) OR layers.N.feed_forward.w13 + w2 (nunchaku-patched)
ZIMAGE_NEXTDIT_UNPATCHED_KEY_MAPPING = [
    (re.compile(r"^(layers)[._](\d+)[._]attention[._]to[._]([qkv])$"), r"\1.\2.attention.qkv", "qkv", lambda m: m.group(3).upper()),
    (re.compile(r"^(layers)[._](\d+)[._]attention[._]to[._]out(?:[._]0)?$"), r"\1.\2.attention.out", "regular", None),
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._](w1|w2|w3)$"), r"\1.\2.feed_forward.\3", "regular", None),
]

ZIMAGE_NEXTDIT_NUNCHAKU_PATCHED_KEY_MAPPING = [
    (re.compile(r"^(layers)[._](\d+)[._]attention[._]to[._]([qkv])$"), r"\1.\2.attention.qkv", "qkv", lambda m: m.group(3).upper()),
    (re.compile(r"^(layers)[._](\d+)[._]attention[._]to[._]out(?:[._]0)?$"), r"\1.\2.attention.out", "regular", None),
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._](w1|w3)$"), r"\1.\2.feed_forward.w13", "glu", lambda m: m.group(3)),
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._]w2$"), r"\1.\2.feed_forward.w2", "regular", None),
]

KEY_MAPPING = [
    # Z-Image QKV Fusion
    # Matches: layers.X.attention.to_q/k/v -> layers.X.attention.to_qkv
    (re.compile(r"^(layers)[._](\d+)[._]attention[._]to[._]([qkv])$"), r"\1.\2.attention.to_qkv", "qkv", lambda m: m.group(3).upper()),

    # Z-Image MLP Fusion (GLU: w1=gate, w3=up -> fused into ff.net.0.proj)
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._](w1|w3)$"), r"\1.\2.feed_forward.net.0.proj", "glu", lambda m: m.group(3)),
    # Z-Image MLP Output (w2 -> ff.net.2)
    (re.compile(r"^(layers)[._](\d+)[._]feed_forward[._]w2$"), r"\1.\2.feed_forward.net.2", "regular", None),

    # Z-Image (ZXIT) Generic Mapping
    # Matches: layers.0.xxx -> layers.0.xxx
    # This covers other layers.
    (re.compile(r"^(layers)[._](\d+)[._](.*)$"), r"\1.\2.\3", "regular", None),

    # Fused QKV (Double Block)
    # ... existing mappings ...
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

    mapping_to_use = _ACTIVE_KEY_MAPPING if _ACTIVE_KEY_MAPPING is not None else KEY_MAPPING

    for pattern, template, group, comp_fn in mapping_to_use:
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

    # Z-Image-Turbo (NextDiT) fallback mappings
    # NextDiT uses: layers.N.attention.qkv (not .to_qkv)
    if ".attention.to_qkv" in name:
        alt = name.replace(".attention.to_qkv", ".attention.qkv")
        m = _get_module_by_name(model, alt)
        if m is not None: return alt, m
    # NextDiT uses: layers.N.attention.out (not .to_out.0)
    if ".attention.to_out.0" in name:
        alt = name.replace(".attention.to_out.0", ".attention.out")
        m = _get_module_by_name(model, alt)
        if m is not None: return alt, m
    # NextDiT feed_forward: .net.0.proj -> .w13 (for GLU) or .w1/.w3 (unpatched)
    if ".feed_forward.net.0.proj" in name:
        # Try w13 first (nunchaku-patched)
        alt = name.replace(".feed_forward.net.0.proj", ".feed_forward.w13")
        m = _get_module_by_name(model, alt)
        if m is not None: return alt, m
        # Try w1 (unpatched, for gate)
        alt = name.replace(".feed_forward.net.0.proj", ".feed_forward.w1")
        m = _get_module_by_name(model, alt)
        if m is not None: return alt, m
    # NextDiT feed_forward: .net.2 -> .w2
    if ".feed_forward.net.2" in name:
        alt = name.replace(".feed_forward.net.2", ".feed_forward.w2")
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


def _fuse_glu_lora(glu_weights: Dict[str, torch.Tensor]) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fuse GLU LoRA weights (gate/w1 and up/w3) into a single tensor for SwiGLU projection.
    
    Args:
        glu_weights: 'w1' (gate) and 'w3' (up) LoRA weights
        
    Returns:
        Fused A, B, alpha
    """
    # w1 is usually gate, w3 is value (up) in LLaMA-like SwiGLU
    # Target module (ff.net[0].proj) has out_features = w1.out + w3.out
    
    if "w1_A" not in glu_weights or "w3_A" not in glu_weights:
        # If one is missing, we could theoretically support partial application,
        # but for now let's require both for simplicity or return None
        return None, None, None

    A_w1, B_w1 = glu_weights["w1_A"], glu_weights["w1_B"]
    A_w3, B_w3 = glu_weights["w3_A"], glu_weights["w3_B"]
    
    alpha_w1 = glu_weights.get("w1_alpha")
    alpha_w3 = glu_weights.get("w3_alpha") # w3 is 'up' or 'value'

    # Check consistency
    if A_w1.shape[0] != A_w3.shape[0]: # in_features should match
         logger.warning(f"GLU LoRA in_features mismatch: {A_w1.shape} vs {A_w3.shape}")
         return None, None, None

    r1 = B_w1.shape[1]
    r3 = B_w3.shape[1]
    
    # Fused A: Concatenate A_w1 and A_w3 (Rank becomes r1 + r3)
    # A shape: (rank, in_features) -> (r1+r3, in)
    A_fused = torch.cat([A_w1, A_w3], dim=0)
    
    # Fused B: Block diagonal
    # B shape: (out_features, rank)
    # Target out_features = out_w1 + out_w3
    out1 = B_w1.shape[0]
    out3 = B_w3.shape[0]
    
    B_fused = torch.zeros(out1 + out3, r1 + r3, dtype=B_w1.dtype, device=B_w1.device)
    B_fused[:out1, :r1] = B_w1
    B_fused[out1:, r1:] = B_w3
    
    # Alpha: If different, we might need to verify logic.
    # Usually they are same. If not, we rely on the fact that scaling is done BEFORE fusion if we were being careful,
    # but here 'compose_loras' applies scale later.
    # If alphas differ, we technically can't use a single scalar alpha for the whole fused layer if rank is used for scaling.
    # But usually alpha is constant.
    alpha_fused = alpha_w1
    if alpha_w1 is not None and alpha_w3 is not None and alpha_w1.item() != alpha_w3.item():
         logger.warning("GLU LoRA alphas differ. Using w1 alpha.")
    
    return A_fused, B_fused, alpha_fused


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
    if module is None:
        raise ValueError(f"{module_name}: Module is None, cannot apply LoRA")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"{module_name}: A/B must be 2D, got {A.shape}, {B.shape}")
    # FACT: A must match module.in_features exactly (no padding/estimation)
    if not hasattr(module, "in_features"):
        raise ValueError(f"{module_name}: Module does not have in_features attribute")
    if A.shape[1] != module.in_features:
        raise ValueError(f"{module_name}: A shape {A.shape} mismatch with in_features={module.in_features}")
    if B.shape[0] != module.out_features:
        raise ValueError(f"{module_name}: B shape {B.shape} mismatch with out_features={module.out_features}")

    # Handle AWQ quantized linear layers (e.g. AWQW4A16Linear) by injecting LoRA in forward path.
    # NOTE: We avoid importing the class directly; name/path may differ across environments.
    if (
        module.__class__.__name__ == "AWQW4A16Linear"
        and hasattr(module, "qweight")
        and hasattr(module, "wscales")
        and hasattr(module, "wzeros")
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
    ):
        # Save original forward once
        if not hasattr(module, "_lora_original_forward"):
            try:
                module._lora_original_forward = module.forward
            except Exception:
                module._lora_original_forward = None

        # Attach LoRA tensors on the module
        if not hasattr(module, "_nunchaku_lora_bundle"):
            module._nunchaku_lora_bundle = []
        module._nunchaku_lora_bundle.append((A, B))
        module._is_modulation_layer = True # Flag to ensure forward knows it's being patched

        def _awq_lora_forward(x, *args, **kwargs):
            orig = getattr(module, "_lora_original_forward", None)
            if orig is None:
                # Fall back, but don't crash (safety)
                return module.forward(x, *args, **kwargs)
            
            # 1. Run original forward (quantized)
            out = orig(x, *args, **kwargs)

            # 2. Add LoRA contribution
            # Check for bundle
            if not hasattr(module, "_nunchaku_lora_bundle") or not module._nunchaku_lora_bundle:
                 return out

            dtype = out.dtype
            device = out.device
            in_features = module.in_features
            
            # Prepare input: flatten to [N, in_features]
            x_in = x
            if not torch.is_tensor(x_in):
                return out
            
            # Reshape input efficiently
            x_flat = x_in.reshape(-1, in_features)

            for (A_local, B_local) in module._nunchaku_lora_bundle:
                # A_local: [rank, in], B_local: [out, rank]
                # Cast to correct device/dtype
                A_local = A_local.to(device=device, dtype=dtype)
                B_local = B_local.to(device=device, dtype=dtype)

                # Compute LoRA term: (X @ A.T) @ B.T
                # [N, in] @ [in, rank] -> [N, rank]
                lora_mid = x_flat @ A_local.transpose(0, 1)
                # [N, rank] @ [rank, out] -> [N, out]
                lora_term = lora_mid @ B_local.transpose(0, 1)

                # Reshape back to output shape
                try:
                    lora_term = lora_term.reshape(out.shape)
                    out = out + lora_term
                except Exception:
                     # Fallback for modulation layer specific reshaping if needed
                     pass

            return out

        # Patch forward
        if module.forward.__code__ != _awq_lora_forward.__code__:
             module.forward = _awq_lora_forward

        if not hasattr(model, "_lora_slots"):
            model._lora_slots = {}
        # Track for reset
        model._lora_slots[module_name] = {"type": "awq_w4a16"}
        return

    # Handle Nunchaku LoRA-ready modules
    if hasattr(module, "proj_down") and hasattr(module, "proj_up"):
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
        slot = model._lora_slots.setdefault(module_name, {"base_rank": base_rank, "appended": 0, "axis_down": axis_down, "type": "nunchaku"})
        slot["appended"] += A.shape[0]

    # Handle Standard nn.Linear (Fallback)
    elif isinstance(module, nn.Linear):
        if not hasattr(model, "_lora_slots"):
            model._lora_slots = {}
        
        # Initialize slot and backup original weight if not exists
        if module_name not in model._lora_slots:
            # Backup original weight to CPU to save VRAM
            model._lora_slots[module_name] = {
                "type": "linear",
                "original_weight": module.weight.detach().cpu().clone()
            }
        
        # Calculate Delta: B @ A
        # B: [out, rank], A: [rank, in] -> Delta: [out, in]
        delta = B @ A
        if delta.shape != module.weight.shape:
             raise ValueError(f"{module_name}: LoRA delta shape {delta.shape} mismatch with linear weight {module.weight.shape}")
        
        # Apply to weight
        module.weight.data.add_(delta.to(dtype=module.weight.dtype, device=module.weight.device))
    
    else:
        # Should be caught by caller, but safety check
        raise ValueError(f"{module_name}: Unsupported module type {type(module)}")


# --- Main Public API ---

# Fallback for _APPLY_AWQ_MOD if it's not exported or we want to read env var directly
# (It is exported as _APPLY_AWQ_MOD in line 28 of lora_qwen.py, but usually private)
DEFAULT_APPLY_AWQ_MOD_ENV = str(os.getenv("QWENIMAGE_LORA_APPLY_AWQ_MOD", "0")).strip().lower() in ("1", "true", "yes", "y", "on")

def _load_lora_state_dict_robust(path_or_dict: Union[str, Dict]) -> Dict:
    """Helper to unify loading from path or returning dict directly."""
    import comfy.utils
    if isinstance(path_or_dict, dict):
        return path_or_dict
    
    # Handle string/Path
    path = str(path_or_dict)
    try:
        if os.path.exists(path):
            logger.debug(f"Loading LoRA from path: {path}")
            return comfy.utils.load_torch_file(path)
        else:
            logger.error(f"LoRA path does not exist: {path}")
    except Exception as e:
        logger.error(f"Failed to load LoRA from {path}: {e}")
    
    return {}

def compose_loras_v2(
        model: torch.nn.Module,
        lora_configs: List[Tuple[Union[str, Path, Dict[str, torch.Tensor]], float]],
        apply_awq_mod: Union[bool, str] = "auto",
) -> bool:
    """
    Resets and composes multiple LoRAs into the model with individual strengths.
    """
    logger.info(f"Composing {len(lora_configs)} LoRAs (Direct Fix V6)...")
    reset_lora_v2(model)
    _first_detection = None

    # --- 1. Z-Image / NextDiT Handling REMOVED (Reverting to original successful state) ---
    global _ACTIVE_KEY_MAPPING
    # Original state: do not modify _ACTIVE_KEY_MAPPING based on model structure
    # _ACTIVE_KEY_MAPPING = None 


    # --- 2. Environment Variable / Argument Logic for AWQ Mod ---
    should_apply_awq_mod = False
    if isinstance(apply_awq_mod, str):
        if apply_awq_mod.lower() == "auto":
             should_apply_awq_mod = DEFAULT_APPLY_AWQ_MOD_ENV
        else:
             should_apply_awq_mod = apply_awq_mod.strip().lower() in ("1", "true", "yes", "y", "on")
    elif isinstance(apply_awq_mod, bool):
        # Explicit bool value - use it directly (V2 node passes bool explicitly)
        should_apply_awq_mod = apply_awq_mod
    else:
        # None or other -> default to False (safety first)
        should_apply_awq_mod = False
    
    if should_apply_awq_mod:
        logger.info("⚠️  AWQ Modulation Layer LoRA Injection ENABLED (via override).")

    # --- 3. Debug Inspection (First LoRA) ---
    if lora_configs:
        first_lora_path_or_dict, first_lora_strength = lora_configs[0]
        first_lora_state_dict = _load_lora_state_dict_robust(first_lora_path_or_dict)
        
        # Simple logging of format detection
        _first_detection = _detect_lora_format(first_lora_state_dict)
        _log_lora_format_detection(str(first_lora_path_or_dict)[:50], _first_detection)

    aggregated_weights: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # --- 4. Main Loading Loop ---
    for idx, (lora_path_or_dict, strength) in enumerate(lora_configs):
        if abs(strength) < 1e-5:
            continue
        
        lora_name = str(lora_path_or_dict)
        if isinstance(lora_path_or_dict, dict):
            lora_name = f"lora_{idx}"
        
        lora_state_dict = _load_lora_state_dict_robust(lora_path_or_dict)
        if not lora_state_dict:
            continue

        detection = _detect_lora_format(lora_state_dict)
        if not detection["has_standard"]:
            logger.warning(f"Skipping unsupported LoRA: {lora_name}")
            continue

        # Group keys by base module
        lora_grouped = defaultdict(dict)
        for key, value in lora_state_dict.items():
            parsed_res = _classify_and_map_key(key)
            if parsed_res:
                group, base_key, comp, ab = parsed_res
                if comp and ab:
                    lora_grouped[base_key][f"{comp}_{ab}"] = value
                elif ab:
                    lora_grouped[base_key][ab] = value
                else:
                    logger.debug(f"Skipping unmappable key part: {key}")

        # Process groups
        processed_groups = {}
        special_handled = set()
        
        for base_key, lw in lora_grouped.items():
            if base_key in special_handled:
                continue

            # Standard Logic Copied from Original
            if "lokr_w1" in lw or "lokr_w2" in lw:
                continue # LoKR skipped

            A, B, alpha = None, None, None
            
            if "qkv" in base_key:
                A, B, alpha = (lw.get("A"), lw.get("B"), lw.get("alpha")) if "A" in lw else _fuse_qkv_lora(lw, model=model, base_key=base_key)
            elif "w1_A" in lw or "w3_A" in lw:
                A, B, alpha = _fuse_glu_lora(lw)
            elif ".proj_out" in base_key and "single_transformer_blocks" in base_key:
                split_map, consumed_keys = _handle_proj_out_split(lora_grouped, base_key, model)
                processed_groups.update(split_map)
                special_handled.update(consumed_keys)
                continue
            else:
                A, B, alpha = lw.get("A"), lw.get("B"), lw.get("alpha")

            if A is not None and B is not None:
                processed_groups[base_key] = (A, B, alpha)

        for module_key, (A, B, alpha) in processed_groups.items():
            aggregated_weights[module_key].append(
                {"A": A, "B": B, "alpha": alpha, "strength": strength, "source": lora_name}
            )

    # --- 5. Application Loop (with AWQ Fix) ---
    applied_modules_count = 0
    skipped_modules_count = 0
    mod_layer_applied_count = 0
    
    all_A = [] # Initialization for safety

    for module_name_key, weight_list in aggregated_weights.items():
        # Correctly resolving module (handling tuple return if needed from _resolve_module_name, but it returns tuple (name, mod) or just (name, mod) in signature, wait.)
        # _resolve_module_name defined in this file returns Tuple[str, Optional[nn.Module]]
        # Wait, if module is None, it returns None? line 416 returns name, m. line 409 returns None.
        # Ah, the definition of resolve_module_name:
        # def _resolve_module_name(model, name) -> Tuple[str, Optional[nn.Module]]:
        
        resolved_name, module = _resolve_module_name(model, module_name_key)
        
        # Skip if module not found
        if module is None:
            logger.warning(f"[MISS] Module not found: {module_name_key} (resolved: {resolved_name})")
            skipped_modules_count += 1
            continue
        
        all_A = []
        all_B_scaled = []

        for w in weight_list:
            A, B, alpha = w["A"], w["B"], w["alpha"]
            strength = w["strength"]
            source = w["source"] 

            rank = A.shape[0]
            if B.shape[1] != rank:
                continue

            scale = strength
            if alpha is not None:
                scale *= (alpha / rank)
            
            # --- AWQ Modulation Layer Check using String Check ---
            # Safety check: ensure module is not None before accessing attributes
            if module is None:
                logger.warning(f"[SKIP] Module is None for {module_name_key} in weight processing, skipping")
                continue
                
            is_awq_w4a16 = (
                module.__class__.__name__ == "AWQW4A16Linear"
                and hasattr(module, "qweight")
            )
            is_modulation_layer = "img_mod" in resolved_name or "txt_mod" in resolved_name
            
            if is_awq_w4a16 and is_modulation_layer:
                if should_apply_awq_mod:
                     # For modulation layers, we force FP16/CUDA usually (same as module weight)
                     # But AWQ module doesn't have .weight usually. has .qweight.
                     target_device = module.qweight.device
                     target_dtype = torch.float16
                     
                     all_A.append(A.to(dtype=target_dtype, device=target_device))
                     all_B_scaled.append((B * scale).to(dtype=target_dtype, device=target_device))
                     # mod_layer_applied_count += 1  <-- Removed to avoid double counting per LoRA
                else:
                    skipped_modules_count += 1
            elif is_awq_w4a16:
                 # Standard AWQ (not modulation)
                     target_device = module.qweight.device
                     target_dtype = torch.float16
                     all_A.append(A.to(dtype=target_dtype, device=target_device))
                     all_B_scaled.append((B * scale).to(dtype=target_dtype, device=target_device))
            
            else:
                # Standard Linear / Nunchaku
                if hasattr(module, "proj_down"):
                     target_dtype = module.proj_down.dtype
                     target_device = module.proj_down.device
                elif hasattr(module, "weight"):
                     target_dtype = module.weight.dtype
                     target_device = module.weight.device
                else:
                     target_dtype = torch.float16
                     target_device = "cuda"

                all_A.append(A.to(dtype=target_dtype, device=target_device))
                all_B_scaled.append((B * scale).to(dtype=target_dtype, device=target_device))


        if not all_A:
            continue
            
        final_A = torch.cat(all_A, dim=0)
        final_B = torch.cat(all_B_scaled, dim=1)
            
        # Standard application via _apply_lora_to_module
        # Now supporting manual bundle for AWQ
        # Re-define flags for the module scope
        # Double-check module is not None (safety)
        if module is None:
            logger.warning(f"[SKIP] Module is None for {module_name_key}, skipping LoRA application")
            continue
            
        is_awq_w4a16 = (module.__class__.__name__ == "AWQW4A16Linear" and hasattr(module, "qweight"))
        is_modulation_layer = "img_mod" in resolved_name or "txt_mod" in resolved_name

        # Special handling for AWQ modulation layers:
        # We do NOT patch the forward loop. Instead, we attach the weights to the module
        # and manually compute/add the LoRA term in the TransformerBlock forward pass (models/qwenimage.py).
        # This avoids all layout/transpose issues by operating on the clean Planar output.
        if is_awq_w4a16 and is_modulation_layer:
            logger.info(f"[AWQ_MOD] {resolved_name}: Storing LoRA weights for manual Planar injection")
            
            mod = module
            
            # Store as a tuple on the module
            # final_A: [rank, in], final_B: [out, rank]
            mod._nunchaku_lora_bundle = (final_A, final_B)
            
            # Remove any existing patch if present (from previous runs)
            if hasattr(mod, "_lora_original_forward"):
                mod.forward = mod._lora_original_forward
                del mod._lora_original_forward
            
            # Tag it
            mod._is_modulation_layer = True
                
            mod_layer_applied_count += 1
        else:
            _apply_lora_to_module(module, final_A, final_B, resolved_name, model)
            # If AWQ but not special modulation handling (or fallback), set flag
            if is_awq_w4a16 and is_modulation_layer:
                module._is_modulation_layer = True
            
            logger.info(f"[APPLY] LoRA applied to: {resolved_name}")
            applied_modules_count += 1

    logger.info(f"Sampled LoRA composition complete. Applied: {applied_modules_count}, Mod-patched: {mod_layer_applied_count}, Skipped: {skipped_modules_count}")
    return True

def update_lora_params_v2(
        model: torch.nn.Module,
        lora_state_dict_or_path: Union[str, Path, Dict[str, torch.Tensor]],
        strength: float = 1.0,
) -> None:
    """Loads and applies a single LoRA to the model (convenience wrapper)."""
    logger.info(f"Loading single LoRA with strength {strength}.")
    compose_loras_v2(model, [(lora_state_dict_or_path, strength)])


def compose_loras_v2_v2(
        model: torch.nn.Module,
        lora_configs: List[Tuple[Union[str, Path, Dict[str, torch.Tensor]], float]],
        apply_awq_mod: bool | str | None = None,
) -> bool:
    """
    V2 node-specific LoRA composition function with AWQ modulation control.
    
    This function is identical to compose_loras_v2 except it accepts an apply_awq_mod
    parameter that overrides the global _APPLY_AWQ_MOD environment variable.
    
    Args:
        model: The model to apply LoRAs to
        lora_configs: List of (lora_path_or_dict, strength) tuples
        apply_awq_mod: If True/False, overrides global _APPLY_AWQ_MOD setting.
                       If None or "auto", uses global _APPLY_AWQ_MOD setting.
                       If string "0"/"false"/"no"/"off" or "1"/"true"/"yes"/"on", parsed as bool.
    
    Returns:
        bool: True if the LoRA format is supported and processed, False otherwise.
    """
    # Determine effective AWQ modulation setting:
    # - If apply_awq_mod is explicitly provided, use it
    # - Otherwise, fall back to global _APPLY_AWQ_MOD
    if isinstance(apply_awq_mod, str):
        apply_awq_mod_flag = apply_awq_mod.lower() not in ("0", "false", "no", "n", "off", "auto", "")
    elif isinstance(apply_awq_mod, bool):
        apply_awq_mod_flag = apply_awq_mod
    else:
        # None or other -> use global setting
        apply_awq_mod_flag = _APPLY_AWQ_MOD
    
    logger.info(f"[V2 Node] Composing {len(lora_configs)} LoRAs with apply_awq_mod={apply_awq_mod} (effective: {apply_awq_mod_flag})...")
    
    # Call compose_loras_v2 with the determined flag
    # Pass as bool explicitly to override default "auto" behavior
    return compose_loras_v2(model, lora_configs, apply_awq_mod=apply_awq_mod_flag)


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


def reset_lora_v2(model: nn.Module) -> None:
    """Removes all appended LoRA weights from the model."""
    if not hasattr(model, "_lora_slots") or not model._lora_slots:
        return

    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None:
            continue

    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None:
            continue

        module_type = info.get("type", "nunchaku") # Default to nunchaku for backward compatibility logic

        if module_type == "nunchaku":
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

        elif module_type == "linear":
            if "original_weight" in info:
                # Restore original weight
                with torch.no_grad():
                    module.weight.data.copy_(info["original_weight"].to(module.weight.device))

        elif module_type == "awq_w4a16":
            # Restore original forward and remove attached LoRA tensors
            if hasattr(module, "_lora_original_forward"):
                try:
                    module.forward = module._lora_original_forward
                except Exception:
                    # Safety: never fail reset
                    pass
            
            # Clean up attributes
            for attr in ("_lora_A", "_lora_B", "_lora_original_forward", "_nunchaku_lora_bundle", "_is_modulation_layer"):
                if hasattr(module, attr):
                    try:
                        delattr(module, attr)
                    except Exception:
                        pass
            
    model._lora_slots.clear()
    model._lora_strength = 1.0
    logger.info("All LoRA weights have been reset from the model.")