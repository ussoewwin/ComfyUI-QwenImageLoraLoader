# Comprehensive Technical Report: Z-Image (Nunchaku) LoRA Integration

**Version**: 2.0.8 Integration Analysis
**Date**: 2025-12-25
**Scope**: Enabling full LoRA support for CompVis/Nunchaku-based Z-Image models in ComfyUI.

---

## 1. Context & Objective
The objective was to integrate LoRA support for the **Z-Image** model running on the **Nunchaku** inference backend within `ComfyUI-QwenImageLoraLoader`. This was challenging because Nunchaku optimizations fundamentally change how the model is structured and executed compared to standard `diffusers` pipelines.

Unlike Qwen/Flux models which were already supported, Z-Image introduced two critical hurdles:
1.  **Tensor Shape Mismatch**: The Nunchaku engine enforces strict 5D input requirements `(B, F, C, H, W)` that conflicted with ComfyUI's standard latent formats.
2.  **MLP Layer Incompatibility**: The model uses a highly optimized `SwiGLU` activation in its Feed-Forward (MLP) blocks. Standard LoRA injection logic failed to recognize these blocks, resulting in "silent failures" where LoRA weigths were loaded but not applied to the majority of model parameters.

---

## 2. Technical Architecture Analysis

### 2.1 The Nunchaku Z-Image Block
The core processing unit `ZImageTransformerBlock` deviates from standard implementations.
*   **Path**: `nunchaku.models.transformers.transformer_zimage.ZImageTransformerBlock`
*   **Structure**:
    *   **Attention**: Standard Self-Attention (Q, K, V).
    *   **Feed-Forward (MLP)**:
        *   Uses `diffusers.models.activations.SwiGLU`.
        *   Historically, Diffusers models separate `w1` (Gate) and `w3` (Up) linear layers.
        *   Nunchaku's implementation **fuses** `w1` and `w3` into a single projection layer.

**Block Structure Details:**

1. **Attention Block**:
   - Standard multi-head self-attention mechanism.
   - Q, K, V projections are fused into a single `to_qkv` linear layer for efficiency.
   - This fusion is standard in many optimized transformer implementations and doesn't require special LoRA handling.

2. **Feed-Forward Block Structure**:
   - Path in model: `layers[i].feed_forward.net` where `net` is a `nn.Sequential`:
     - `net[0]`: `SwiGLU` container module containing `proj: Linear(dim_in, dim_out * 2)`
     - `net[1]`: Activation function (usually identity or SiLU)
     - `net[2]`: `Linear(dim_out, dim_out * expansion_factor)` (the `w2` layer)
   - The `SwiGLU` at `net[0]` replaces the original separate `w1` and `w3` layers.
   - Original structure would be: `w1: Linear(dim_in, dim_out)`, `w3: Linear(dim_in, dim_out)`, `w2: Linear(dim_out, dim_out * expansion)`.
   - Fused structure: `proj: Linear(dim_in, dim_out * 2)` (inside SwiGLU), `w2: Linear(dim_out, dim_out * expansion)`.

### 2.2 SwiGLU Mechanics
The `SwiGLU` class acts as a customized Container, not a raw `nn.Linear` layer.
```python
class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        self.proj = nn.Linear(dim_in, dim_out * 2)  # Fused w1+w3
        self.activation = nn.SiLU()

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.activation(gate)
```
**Implication**: You cannot apply LoRA to `SwiGLU` directly. You must apply it to `SwiGLU.proj`. Furthermore, because `SwiGLU.proj` outputs `dim_out * 2`, the LoRA weights for `w1` and `w3` must be similarly fused before application.

**Detailed Technical Analysis:**

1. **Why Direct LoRA Application Fails**:
   - Standard LoRA injection mechanisms search for `nn.Linear` or `nn.Conv2d` modules in the model's module tree.
   - `SwiGLU` is a container module (`nn.Module`) that wraps a `nn.Linear` layer, but the injection logic doesn't recursively search inside container modules.
   - When the injection logic encounters `SwiGLU`, it sees an unsupported module type and skips it, logging `[MISS] Module found but unsupported type: <class 'diffusers.models.activations.SwiGLU'>`.

2. **The Fused Projection Layer**:
   - `SwiGLU.proj` is a `nn.Linear(dim_in, dim_out * 2)` layer that outputs twice the dimension of the intended output.
   - In the forward pass, this output is split using `chunk(2, dim=-1)`, creating two tensors of shape `(..., dim_out)`.
   - The first half becomes the "value" (`x`), and the second half becomes the "gate" (`gate`).
   - The gate is passed through `SiLU` activation, then multiplied element-wise with the value: `x * SiLU(gate)`.

3. **LoRA Weight Structure**:
   - Standard Z-Image LoRA files contain separate weights for `w1` (gate projection) and `w3` (value/up projection).
   - Each has its own LoRA matrices: `w1_A`, `w1_B`, `w3_A`, `w3_B` (and optionally `w1_alpha`, `w3_alpha`).
   - The base model has these fused into a single `proj` layer, so LoRA weights must be fused to match this structure.

4. **Mathematical Relationship**:
   - If the original model had separate `w1: Linear(dim_in, dim_out)` and `w3: Linear(dim_in, dim_out)`, the fused version is `proj: Linear(dim_in, dim_out * 2)`.
   - The fused weight matrix is: `W_fused = [W1; W3]` (vertical concatenation along the output dimension), where `W1` and `W3` are the weight matrices of `w1` and `w3` respectively.
   - This means `W_fused.shape == (dim_out * 2, dim_in)`, matching the `proj` layer's weight shape.
   - During forward pass: `proj_output = W_fused @ x`, which produces shape `(dim_out * 2, batch_size)`. The `chunk(2, dim=-1)` splits this into two `(dim_out, batch_size)` tensors, the first being the value and the second being the gate.
   - For LoRA application, the same fusion principle applies: if separate LoRAs modify `w1` and `w3`, their weight deltas must be fused before application to the fused `proj` layer.

---

## 3. Deep Dive: Debugging & Resolution Steps

### Phase 1: The Tensor Shape Crisis (`wrappers/zimageturbo.py`)
**Symptom**: `RuntimeError` regarding batch size mismatches during generation.
**Cause**:
ComfyUI often processes latents as a 4D tensor `(Batch * Frames, C, H, W)`. Nunchaku's `forward` pass explicitly expects 5D input to handle temporal data:
```python
# Nunchaku Requirement
x.shape == (batch_size, num_frames, channels, height, width)
```
**Fix Strategy**:
1.  **Batch Synchronization**: We explicitly pass `batch_size=B` to the inference call, ensuring `x` and `cap_feats` align.
2.  **Dimensional Gymnastics**: The wrapper now intelligently reshapes usage.
    *   Input: `[B, C, H, W]` (Single frame inference contexts)
    *   Transform: `unsqueeze(1)` -> `[B, 1, C, H, W]`
    *   Inference: Nunchaku processes the 5D tensor.
    *   Output: `squeeze(1)` -> `[B, C, H, W]` to return to ComfyUI format.

**Implementation Details (`wrappers/zimageturbo.py`, lines 251-280):**

The `_execute_model` method handles tensor format conversion and device management:

1. **List Format Handling (lines 253-266)**:
   - If `x` is already a list (from model base conversion), filters out `None` elements.
   - Each tensor in the list is moved to the model's device if needed: `img.to(model_device)`.
   - Validates that the filtered list is not empty, raising `ValueError` if empty.

2. **Tensor Format Handling (lines 267-275)**:
   - If `x` is still a tensor, moves it to model device first.
   - Checks for 5D input: `input_is_5d = x.ndim == 5`.
   - If 5D, squeezes dimension 2: `x = x.squeeze(2)` to convert from `(B, F, C, H, W)` format.

3. **Context Device Management (lines 277-280)**:
   - Moves `context` tensor to model device if it's a tensor (not a list).
   - Ensures all tensors are on the same device before model execution.

4. **List Format Conversion for Direct Forward (lines 308-337)**:
   - Z-Image-Turbo's forward method expects `x` as a list of tensors with shape `(C, F, H, W)` where F is the frame dimension.
   - Conversion process:
     - If `x` is already a list, processes each element individually (lines 309-322).
     - Filters out `None` elements and validates `img.numel() > 0` to avoid empty tensors (line 313).
     - Handles batch dimension: if input is `(1, C, H, W)` (batch size 1), squeezes dimension 0 to get `(C, H, W)` (lines 315-316).
     - Ensures frame dimension: if tensor is `(C, H, W)` (3D), unsqueezes dimension 1 to get `(C, 1, H, W)` (lines 318-320).
     - Appends each processed tensor to `x_list` for batch processing.
   - If `x` is a tensor (not a list), splits it by batch dimension (lines 323-333).
   - **Empty list validation** (lines 335-337): Raises `ValueError` if `x_list` is empty, as Z-Image-Turbo cannot process empty batches.
   - **Length mismatch detection** (lines 381-386): Checks if `len(x_list) != len(cap_feats)` and warns, as `zip()` will truncate to the shorter length, potentially causing batch size mismatches.

### Phase 2: The "Weak LoRA" Mystery (MLP Injection Failure)
**Symptom**: LoRA was loaded, but generated images changed very slightly. Users reported "it feels like strength 0.1 even at 1.0".
**Investigation**:
Using detailed `[DEBUG]` logging inspection of the `apply_lora` function in `lora_qwen.py`.

#### Findings
1.  **Success**: Attention layers (`to_q`, `to_k`, `to_v`) were recognized and mapped correctly to `to_qkv`.
2.  **Failure**: MLP layers (`w1`, `w2`, `w3`) showed erratic behavior.
    *   *Attempt 1*: Regex targeted `ff.net`. result: **0 matches**.
    *   *Correction 1*: Variable name is `feed_forward`, not `ff`.
    *   *Attempt 2*: Regex targeted `feed_forward.net`. result: **Match found, Injection Failed**.

**Investigation Process Details:**

The debugging process involved inspecting the `_classify_and_map_key` function (`nunchaku_code/lora_qwen.py`, lines 170-222) and module resolution:

1. **Key Mapping Process**:
   - LoRA keys from the safetensors file are parsed using regex patterns in `KEY_MAPPING` (line 27).
   - The regex `r"^(layers)[._](\d+)[._]feed_forward[._](w1|w3)$"` matches keys like `layers.0.feed_forward.w1` or `layers_0_feed_forward_w3`.
   - The template `r"\1.\2.feed_forward.net.0.proj"` maps to `layers.0.feed_forward.net.0.proj`.
   - The group type `"glu"` triggers GLU fusion logic in `compose_loras_v2`.
   - Before pattern matching, `_classify_and_map_key` (line 170) removes common prefixes like `"transformer."`, `"diffusion_model."`, or `"lora_unet_"` (lines 176-182).
   - The function uses `_RE_LORA_SUFFIX` regex (line 120: `r"\.(?P<tag>lora(?:[._](?:A|B|down|up)))(?:\.[^.]+)*\.weight$"`) to extract the base key and identify A/B/alpha components (lines 188-195).
   - If the suffix matches "lora_A" or ends with ".A" or contains "down", `ab` is set to "A". If it matches "lora_B" or ends with ".B" or contains "up", `ab` is set to "B" (lines 192-195).

2. **Module Resolution**:
   - `_resolve_module_name` (line 254) attempts to find the module using `_get_module_by_name` (line 230).
   - The path `layers.0.feed_forward.net.0.proj` is traversed step by step: `layers` -> `0` (index) -> `feed_forward` -> `net` -> `0` (index) -> `proj`.
   - `_get_module_by_name` uses a two-step resolution strategy (lines 237-251):
     - **First**: Try `hasattr(module, part)` then `getattr(module, part)`. This works for regular attributes and also for numeric keys in `nn.Sequential`/`nn.ModuleDict` (which Python exposes as attributes for keys like "0", "1", "2").
     - **Second**: If `hasattr` fails and `part.isdigit()` is true, check if the module is indexable (`_is_indexable_module` checks for `nn.ModuleList`, `nn.Sequential`, `list`, `tuple`). If so, use integer indexing: `module[int(part)]`.
   - The `hasattr`-first approach is necessary because `nn.Sequential` stores modules as attributes with numeric names, so `hasattr(sequential, "0")` returns `True` even though it's not a traditional attribute. `nn.ModuleList` does not expose numeric keys via `hasattr`, requiring the indexing fallback.
   - Fallback mappings (lines 269-277): If direct resolution fails, `_resolve_module_name` tries alternative paths:
     - `.ff.net.0.proj` -> `.mlp_fc1` (for models using `mlp_fc1` instead of `feed_forward.net.0.proj`)
     - `.ff.net.2` -> `.mlp_fc2`
     - This allows compatibility with different model architectures that use different naming conventions.

3. **Failure Point**:
   - When the regex matched `feed_forward.net.0` (without `.proj`), the resolved module was `SwiGLU` container.
   - `_apply_lora_to_module` (line 873) checks `isinstance(module, nn.Linear)`, which fails for `SwiGLU`.
   - The function also checks for `proj_down`/`proj_up` attributes (Nunchaku LoRA-ready modules), which `SwiGLU` also lacks.
   - Result: `ValueError: Unsupported module type <class 'diffusers.models.activations.SwiGLU'>` or similar error, logged as `[MISS]`.

#### The SwiGLU Barrier
The log showed:
```text
[MISS] Module found but unsupported type: <class 'diffusers.models.activations.SwiGLU'>
```
The standard injection logic looks for `nn.Linear`, `nn.Conv2d`, or specific `lora.Linear` layers. It saw `SwiGLU` and skipped it, having no knowledge of how to inject weights into a container.

### Phase 3: The GLU Fusion Solution (`lora_qwen.py`)

To solve this, we architected a specific path for "GLU Fusion".

#### 1. Regex Retargeting
We modified `KEY_MAPPING` to "pierce" the container.
*   **Old**: `...feed_forward.net.0` (Targets the container)
*   **New**: `...feed_forward.net.0.proj` (Targets the inner Linear layer)

**Regex Pattern Details (`nunchaku_code/lora_qwen.py`, line 33):**

```python
(re.compile(r"^(layers)[._](\d+)[._]feed_forward[._](w1|w3)$"), 
 r"\1.\2.feed_forward.net.0.proj", "glu", lambda m: m.group(3))
```

- **Pattern**: `r"^(layers)[._](\d+)[._]feed_forward[._](w1|w3)$"`
  - `^(layers)`: Matches "layers" at start
  - `[._]`: Matches either dot or underscore separator
  - `(\d+)`: Captures layer number (e.g., "0", "1")
  - `[._]feed_forward[._]`: Matches "feed_forward" with any separator
  - `(w1|w3)`: Captures either "w1" or "w3"
- **Template**: `r"\1.\2.feed_forward.net.0.proj"`
  - `\1`: Layer group name ("layers")
  - `\2`: Layer number
  - Adds `.feed_forward.net.0.proj` to target the inner Linear layer
- **Group Type**: `"glu"` indicates GLU fusion is needed
- **Component Function**: `lambda m: m.group(3)` extracts "w1" or "w3" for fusion logic

#### 2. Weight Fusion Logic (`_fuse_glu_lora`)
The Z-Image LoRA files (standard format) provide `w1` (Gate) and `w3` (Up) as separate weight matrices. The Model has them fused. We must replicate this fusion in the LoRA weights *before* injection.

```python
# Concept
w_fused_A = cat([w1_A, w3_A], dim=0)  # Stack ranks? No, usually features.
w_fused_B = cat([w1_B, w3_B], dim=1)  # Output dimension doubles
```
*Note: The actual implementation in `_fuse_glu_lora` ensures correct dimensional concatenation to match `dim_out * 2`.*

**Implementation Details (`nunchaku_code/lora_qwen.py`, lines 640-695):**

The `_fuse_glu_lora` function (line 640) performs the actual fusion:

1. **Input Validation (lines 653-667)**:
   - Requires both `w1_A` and `w3_A` in `glu_weights`, returns `(None, None, None)` if missing.
   - Validates `A_w1.shape[0] == A_w3.shape[0]` (input features must match), logs warning on mismatch.

2. **A Matrix Fusion (line 674)**:
   - Concatenates along dimension 0: `A_fused = torch.cat([A_w1, A_w3], dim=0)`.
   - Result shape: `(r1 + r3, in_features)` where `r1` and `r3` are the ranks of `w1` and `w3` respectively.

3. **B Matrix Fusion (lines 682-684)**:
   - Creates block-diagonal matrix: `B_fused = torch.zeros(out1 + out3, r1 + r3, ...)`.
   - Assigns `B_fused[:out1, :r1] = B_w1` and `B_fused[out1:, r1:] = B_w3`.
   - Result shape: `(out1 + out3, r1 + r3)` matching the fused `proj` layer's output dimension.

4. **Integration in compose_loras_v2 (line 1078-1079)**:
   - GLU fusion is detected by checking `"w1_A" in lw or "w3_A" in lw` in the grouped LoRA weights.
   - The fused weights are then applied via `_apply_lora_to_module` (line 1152) to the `feed_forward.net.0.proj` module (a standard `nn.Linear` layer).

5. **Weight Aggregation and Scaling (lines 1107-1152)**:
   - Multiple LoRAs can target the same module. Weights are aggregated in `aggregated_weights[module_key].append({...})` during the first loop (line 1108).
   - In the second loop (lines 1114-1152), for each module, all LoRA parts from different LoRAs are combined:
     - **Scaling calculation** (line 1129): `scale = strength * (scale_alpha / max(1.0, float(r_lora)))` where `scale_alpha = alpha.item() if alpha is not None else float(r_lora)`. 
     - If `alpha` is `None`, it defaults to the rank (`r_lora`), making the scale simply `strength`. If `alpha` is provided, it acts as a normalization factor: higher alpha increases the effect per rank unit.
     - **Special handling for AdaNorm layers** (lines 1131-1134): If the module name contains `.norm1.linear` or `.norm1_context.linear`, the B matrix is reordered using `reorder_adanorm_lora_up(B, splits=6)` to match AdaNorm's split structure. Similar handling for single transformer blocks with `.norm.linear` (splits=3).
     - **Device and dtype synchronization** (lines 1136-1141): Determines target dtype and device from the module's weight (or `proj_down` for Nunchaku modules), ensuring all tensors match before concatenation.
     - `all_A.append(A.to(dtype=target_dtype, device=target_device))` (line 1143) - A matrices are moved to target device/dtype without scaling.
     - `all_B_scaled.append((B * scale).to(...))` (line 1144) - B matrices are scaled before concatenation. Scaling is applied here rather than after concatenation to allow per-LoRA strength control.
   - **Final concatenation** (lines 1149-1150): `final_A = torch.cat(all_A, dim=0)` concatenates A matrices along the rank dimension, increasing total rank. `final_B = torch.cat(all_B_scaled, dim=1)` concatenates scaled B matrices along the rank dimension, matching the concatenated A shape.
   - The concatenated weights are applied to the module via `_apply_lora_to_module` (line 1152).

6. **Module Application (`_apply_lora_to_module`, lines 873-934)**:
   - For standard `nn.Linear` modules (which includes `feed_forward.net.0.proj`):
     - Validates tensor dimensions: `A.ndim == 2` and `B.ndim == 2`, `A.shape[1] == module.in_features`, `B.shape[0] == module.out_features` (lines 876-882).
     - Backs up original weight to CPU in `_lora_slots` dictionary: `module.weight.detach().cpu().clone()` (line 920). This backup enables LoRA reset functionality.
     - Calculates weight delta: `delta = B @ A` where `B.shape == (out_features, rank)` and `A.shape == (rank, in_features)` (line 925). The matrix multiplication produces `delta.shape == (out_features, in_features)`, matching the module weight shape.
     - Applies delta in-place: `module.weight.data.add_(delta.to(dtype=module.weight.dtype, device=module.weight.device))` (line 930). The `add_` operation modifies weights directly, avoiding temporary tensor allocation.
   - For Nunchaku LoRA-ready modules (with `proj_down`/`proj_up` attributes):
     - Unpacks existing low-rank weights using `unpack_lowrank_weight` (lines 887-888).
     - Determines base rank and concatenation axis based on weight layout (lines 890-897).
     - Concatenates new LoRA A matrices along the rank dimension and B matrices along the output dimension (lines 893, 899).
     - Repacks using `pack_lowrank_weight` and updates `module.rank` (lines 901-903).
     - Tracks appended rank in `_lora_slots` for reset support (line 908).

#### 3. Output Projection (`w2`)
The `w2` layer (Down projection) is a standard linear layer appearing at `feed_forward.net.2`. It requires no special fusion, just correct mapping.

**Mapping Details (`nunchaku_code/lora_qwen.py`, line 35):**

The `w2` layer uses standard regex mapping without fusion:
```python
(re.compile(r"^(layers)[._](\d+)[._]feed_forward[._]w2$"), 
 r"\1.\2.feed_forward.net.2", "regular", None)
```

- Maps `layers.X.feed_forward.w2` directly to `layers.X.feed_forward.net.2`
- Group type `"regular"` means standard LoRA application (no fusion)
- Component function is `None` (no component extraction needed)
- The resolved module is a standard `nn.Linear`, handled by `_apply_lora_to_module` (lines 911-930)

---

## 4. Final Configuration

### Supported Layer Mappings
| LoRA Key (Source) | Model Module (Target) | Logic |
| :--- | :--- | :--- |
| `layers.*.attention.to_q/k/v` | `attention.to_qkv` | **QKV Fusion**: Standard concatenation. |
| `layers.*.feed_forward.w1/w3` | `feed_forward.net.0.proj` | **GLU Fusion (NEW)**: Fuses Gate+Up, injects into SwiGLU inner linear. |
| `layers.*.feed_forward.w2` | `feed_forward.net.2` | **Direct**: Standard linear injection. |

---

## 5. Verification Results
**Test Condition**: Z-Image Model + `MistyPokemon_ZIT.safetensors` (Strength 1.0).

**Logs (v2.0.8)**:
```text
[APPLY] LoRA applied to: layers.0.attention.to_qkv          (Success)
[APPLY] LoRA applied to: layers.0.feed_forward.net.0.proj   (Success - GLU Fused)
[APPLY] LoRA applied to: layers.0.feed_forward.net.2        (Success - Linear)
...
Applied LoRA compositions to 150 modules.
```
**Outcome**:
Changes in generated images are now distinct and accurate to the requested LoRA style, confirming that the MLP layers (which contain ~60% of model parameters) are finally participating in the inference.

**Verification Process:**

1. **Log Analysis**:
   - The `[APPLY]` log messages confirm successful application to `feed_forward.net.0.proj`.
   - The total count "Applied LoRA compositions to 150 modules" includes all transformer layers (attention + MLP).
   - Each layer has attention (`to_qkv`) and MLP (`net.0.proj`, `net.2`) modules, so 150 modules ≈ 50 layers × 3 modules per layer.

2. **Before Fix Behavior**:
   - Only attention layers were modified, representing ~40% of parameters.
   - MLP layers (60% of parameters) were unchanged, causing weak LoRA effects.
   - Users needed to set strength > 1.0 to see noticeable changes, and even then results were inaccurate.

3. **After Fix Behavior**:
   - Both attention and MLP layers are modified.
   - Strength 1.0 produces accurate LoRA style transfer.
   - Image generation shows distinct style changes matching the LoRA training data.

4. **Performance Impact**:
   - LoRA fusion adds minimal computational overhead during composition (one-time operation).
   - Block-diagonal B matrix structure maintains efficient matrix multiplication: `B @ A` computation remains O(rank × features).
   - Memory overhead: fused LoRA weights have shape `(r1 + r3, in_features)` for A and `(out1 + out3, r1 + r3)` for B, compared to separate `(r1, in)`, `(r3, in)`, `(out1, r1)`, `(out3, r3)`.
   - The memory increase is negligible compared to model weights (LoRA rank is typically 4-128, while model dimensions are 1024-4096).

---

## 6. How to Maintain
*   **New Architectures**: If Nunchaku adds new model types, check if they use `SwiGLU`. If so, ensure the regex targets `.proj`.
*   **Updates to Diffusers**: If `diffusers` changes the `SwiGLU` internal name (unlikely, but possible) from `.proj` to `.linear`, `KEY_MAPPING` must be updated.

**Maintenance Guidelines:**

1. **Adding Support for New Model Types**:
   - Inspect the model architecture using `model.named_modules()` to find the path to SwiGLU modules.
   - Verify the path format (e.g., `layers.X.feed_forward.net.0` contains a `SwiGLU` with `.proj` attribute).
   - Add regex pattern to `KEY_MAPPING` if the path format differs from existing patterns.
   - Test with a known LoRA file to verify successful application (check logs for `[APPLY]` messages).

2. **Debugging Failed LoRA Applications**:
   - Enable debug logging: `logger.setLevel(logging.DEBUG)`.
   - Check `[MISS]` log messages to see which modules were not found or unsupported.
   - Use `_resolve_module_name` manually to verify module path resolution.
   - Inspect `_classify_and_map_key` output to verify regex pattern matching.

3. **Key Mapping Updates**:
   - If model architecture changes, update regex patterns in `KEY_MAPPING` (line 27).
   - Ensure template strings correctly map LoRA keys to model module paths.
   - Component functions (lambda) should extract necessary information for fusion logic.
   - Group types (`"glu"`, `"qkv"`, `"regular"`) determine which fusion function to call.

4. **Fusion Logic Maintenance**:
   - `_fuse_glu_lora` assumes `w1` (gate) and `w3` (up) weights are present. If model uses different naming, update the detection logic (line 1078).
   - Block-diagonal B matrix structure assumes `out1 == out3`. If this changes, fusion logic needs adjustment.
   - Alpha handling assumes same alpha for w1/w3. If models require different alphas, fusion logic must be updated.

