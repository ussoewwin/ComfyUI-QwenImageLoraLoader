#!/usr/bin/env python3
"""Analyze SNOFs LoRA file structure"""
import sys
import re
from pathlib import Path
from typing import Optional, Tuple
from collections import defaultdict

# Key mapping from nunchaku_code/lora_qwen.py
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
_RE_ALPHA_SUFFIX = re.compile(r"\.(?:alpha|lora_alpha)(?:\.[^.]+)*$")

def _rename_layer_underscore_layer_name(old_name: str) -> str:
    rules = [
        (r'_(\d+)_attn_to_out_(\d+)', r'.\1.attn.to_out.\2'),
        (r'_(\d+)_img_mlp_net_(\d+)_proj', r'.\1.img_mlp.net.\2.proj'),
        (r'_(\d+)_txt_mlp_net_(\d+)_proj', r'.\1.txt_mlp.net.\2.proj'),
        (r'_(\d+)_img_mlp_net_(\d+)', r'.\1.img_mlp.net.\2'),
        (r'_(\d+)_txt_mlp_net_(\d+)', r'.\1.txt_mlp.net.\2'),
        (r'_(\d+)_img_mod_(\d+)', r'.\1.img_mod.\2'),
        (r'_(\d+)_txt_mod_(\d+)', r'.\1.txt_mod.\2'),
        (r'_(\d+)_attn_', r'.\1.attn.'),
    ]
    new_name = old_name
    for pattern, replacement in rules:
        new_name = re.sub(pattern, replacement, new_name)
    return new_name

def _classify_and_map_key(key: str) -> Optional[Tuple[str, str, Optional[str], str]]:
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
        m = _RE_ALPHA_SUFFIX.search(k)
        if m:
            ab = "alpha"
            base = k[: m.start()]

    if base is None or ab is None:
        return None

    for pattern, template, group, comp_fn in KEY_MAPPING:
        match = pattern.match(base)
        if match:
            final_key = match.expand(template)
            component = comp_fn(match) if comp_fn else None
            return group, final_key, component, ab

    return None

lora_path = r"D:\USERFILES\GitHub\Qwen_Snofs_1_3.safetensors"

print(f"Analyzing LoRA file: {lora_path}")
print("=" * 100)

try:
    from safetensors import safe_open
    
    with safe_open(lora_path, framework="pt") as f:
        keys = list(f.keys())
        print(f"\nTotal keys: {len(keys)}")
        
        # Test key mapping
        print("\n" + "=" * 100)
        print("Key Mapping Analysis:")
        print("=" * 100)
        
        mapped_keys = []
        unmapped_keys = []
        mapped_details = defaultdict(list)
        
        for key in keys:
            parsed = _classify_and_map_key(key)
            if parsed is None:
                unmapped_keys.append(key)
            else:
                group, base_key, comp, ab = parsed
                mapped_keys.append(key)
                mapped_details[base_key].append((key, group, comp, ab))
        
        print(f"\n✓ Mapped keys: {len(mapped_keys)}")
        print(f"✗ Unmapped keys: {len(unmapped_keys)}")
        print(f"  Mapping rate: {len(mapped_keys)/len(keys)*100:.1f}%")
        
        if unmapped_keys:
            print("\n" + "=" * 100)
            print("UNMAPPED KEYS (First 50):")
            print("=" * 100)
            for i, key in enumerate(unmapped_keys[:50], 1):
                print(f"{i:3d}. {key}")
            
            # Analyze unmapped key patterns
            print("\n" + "=" * 100)
            print("Unmapped Key Pattern Analysis:")
            print("=" * 100)
            
            unmapped_prefixes = defaultdict(int)
            
            for key in unmapped_keys:
                if key.startswith("transformer."):
                    p = "transformer.*"
                elif key.startswith("diffusion_model."):
                    p = "diffusion_model.*"
                elif key.startswith("model."):
                    p = "model.*"
                elif key.startswith("unet."):
                    p = "unet.*"
                elif "." in key:
                    p = key.split(".")[0] + ".*"
                else:
                    p = key[:20] + "*"
                unmapped_prefixes[p] += 1
            
            print("\nUnmapped key prefixes:")
            for prefix, count in sorted(unmapped_prefixes.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {prefix:40s}: {count:4d}")
            
            # Check for specific patterns in unmapped keys
            print("\n" + "=" * 100)
            print("Unmapped Key Details (Sample):")
            print("=" * 100)
            for key in unmapped_keys[:20]:
                # Try to understand why it's unmapped
                k = key
                prefix_info = ""
                if k.startswith("transformer."):
                    k = k[len("transformer."):]
                    prefix_info = "[transformer.]"
                elif k.startswith("diffusion_model."):
                    k = k[len("diffusion_model."):]
                    prefix_info = "[diffusion_model.]"
                elif k.startswith("model."):
                    k = k[len("model."):]
                    prefix_info = "[model.]"
                
                # Check suffix
                has_lora_suffix = _RE_LORA_SUFFIX.search(k) is not None
                has_alpha_suffix = _RE_ALPHA_SUFFIX.search(k) is not None
                
                print(f"  {key:70s}")
                print(f"    {prefix_info} processed: '{k}' | lora_suffix: {has_lora_suffix} | alpha_suffix: {has_alpha_suffix}")
        
        if mapped_keys:
            print("\n" + "=" * 100)
            print("MAPPED KEYS (Sample by base_key):")
            print("=" * 100)
            for base_key in sorted(mapped_details.keys())[:10]:
                details = mapped_details[base_key]
                print(f"\n{base_key}:")
                for orig_key, group, comp, ab in details[:3]:
                    print(f"  {orig_key} -> group={group}, comp={comp}, ab={ab}")
        
        # Show first 30 keys with mapping status
        print("\n" + "=" * 100)
        print("First 30 Keys with Mapping Status:")
        print("=" * 100)
        for i, key in enumerate(keys[:30], 1):
            parsed = _classify_and_map_key(key)
            status = "✓ MAPPED" if parsed else "✗ UNMAPPED"
            group_info = f" -> {parsed[1]}" if parsed else ""
            print(f"{i:3d}. [{status:12s}] {key:60s}{group_info}")
        
except ImportError:
    print("Error: safetensors library not available")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
