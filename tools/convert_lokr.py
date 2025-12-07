import argparse
import os
import re
import torch
from safetensors.torch import load_file, save_file
# from tqdm import tqdm

def decompose_lokr(w1, w2, rank=64):
    """
    Decompose LoKR weights (w1, w2) into standard LoRA weights (A, B) using SVD.
    w2 @ w1 (Kronecker product) approx B @ A
    """
    # Ensure float32 for precision
    w1 = w1.float()
    w2 = w2.float()
    
    # Determine Kronecker product order and shape
    # Try w2 ⊗ w1 (Standard)
    # Shape: (w2.rows * w1.rows, w2.cols * w1.cols)
    
    # Note: w1, w2 shapes in safetensors might be transposed depending on implementation.
    # Standard LoKR: w1=[a, b], w2=[c, d] -> result=[ac, bd]
    
    # Construct full matrix on CPU
    try:
        # Try standard kron(w2, w1)
        full_W = torch.kron(w2, w1)
        
        # If full_W is huge, SVD might be slow, but it's offline conversion so OK.
        
        # SVD
        target_rank = min(min(full_W.shape), rank)
        U, S, Vh = torch.linalg.svd(full_W, full_matrices=False)
        
        U = U[:, :target_rank]
        S = S[:target_rank]
        Vh = Vh[:target_rank, :]
        
        # B @ A
        # B = U * sqrt(S)
        # A = sqrt(S) * Vh
        sqrt_S = torch.sqrt(S)
        B = U @ torch.diag(sqrt_S)
        A = torch.diag(sqrt_S) @ Vh
        
        return A, B
    except Exception as e:
        print(f"Error decomposing LoKR: {e}")
        return None, None

def convert_lokr_file(input_path, output_path, rank=64):
    print(f"Loading {input_path}...")
    state_dict = load_file(input_path)
    new_state_dict = {}
    
    # Group keys by base name
    # keys look like: ...lokr_w1, ...lokr_w2, ...alpha
    groups = {}
    
    for key in state_dict.keys():
        if key.endswith("lokr_w1"):
            base = key[:-7].rstrip(".") # remove lokr_w1 and potential dot
            if base not in groups: groups[base] = {}
            groups[base]["w1"] = state_dict[key]
        elif key.endswith("lokr_w2"):
            base = key[:-7].rstrip(".")
            if base not in groups: groups[base] = {}
            groups[base]["w2"] = state_dict[key]
        elif key.endswith("alpha"):
            # Alpha might belong to a lokr group
            # Try to find matching base
            # Often key is base.alpha
            base = key[:-5].rstrip(".")
            # We'll handle alpha association later or assume standard naming
            if base not in groups: groups[base] = {}
            groups[base]["alpha"] = state_dict[key]
        else:
            # Pass through other keys (standard LoRA or unknown)
            new_state_dict[key] = state_dict[key]
            
    print(f"Found {len(groups)} potential LoKR groups.")
    
    converted_count = 0
    total_groups = len(groups)
    
    for i, (base, items) in enumerate(groups.items()):
        if i % 10 == 0:
            print(f"Processing group {i+1}/{total_groups}...", end="\r")
            
        if "w1" in items and "w2" in items:
            w1 = items["w1"]
            w2 = items["w2"]
            
            # Check if standard LoRA or LoKR
            # If shapes are small enough, might be standard LoRA split? No, key says 'lokr'.
            
            print(f"Converting {base}: w1={w1.shape}, w2={w2.shape} -> rank={rank}")
            A, B = decompose_lokr(w1, w2, rank=rank)
            
            if A is not None and B is not None:
                # Map to standard names
                # lora_down.weight (A), lora_up.weight (B)
                # Key mapping:
                # base.lokr_w1 -> base.lora_down.weight ?
                # Need to verify standard LoRA naming conventions for the target model.
                # Usually: lora_unet_..._proj.lora_down.weight
                
                # We will use "lora_down.weight" and "lora_up.weight" suffix
                # Assuming 'base' is the module name
                
                # If base ends with "img_mod_1", "txt_mod_1", etc.
                # Just append .lora_down.weight
                
                new_key_down = f"{base}.lora_down.weight"
                new_key_up = f"{base}.lora_up.weight"
                
                new_state_dict[new_key_down] = A.to(dtype=w1.dtype)
                new_state_dict[new_key_up] = B.to(dtype=w2.dtype)
                
                if "alpha" in items:
                    new_state_dict[f"{base}.alpha"] = items["alpha"]
                else:
                    # Calculate alpha? Or standard 1.0?
                    # Usually alpha is kept. If missing, maybe default to rank.
                    pass
                
                converted_count += 1
            else:
                print(f"Failed to convert {base}")
        else:
            # Copy unconnected items
            if "w1" in items: new_state_dict[f"{base}.lokr_w1"] = items["w1"]
            if "w2" in items: new_state_dict[f"{base}.lokr_w2"] = items["w2"]
            if "alpha" in items: new_state_dict[f"{base}.alpha"] = items["alpha"]

    print(f"Converted {converted_count} LoKR modules.")
    print(f"Saving to {output_path}...")
    save_file(new_state_dict, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LoKR LoRA to Standard LoRA")
    parser.add_argument("input_file", help="Input .safetensors file")
    parser.add_argument("output_file", help="Output .safetensors file")
    parser.add_argument("--rank", type=int, default=64, help="Target rank for conversion (default: 64)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        exit(1)
        
    convert_lokr_file(args.input_file, args.output_file, args.rank)

