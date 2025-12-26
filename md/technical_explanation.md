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

#### 2. Weight Fusion Logic (`_fuse_glu_lora`)
The Z-Image LoRA files (standard format) provide `w1` (Gate) and `w3` (Up) as separate weight matrices. The Model has them fused. We must replicate this fusion in the LoRA weights *before* injection.

```python
# Concept
w_fused_A = cat([w1_A, w3_A], dim=0)  # Stack ranks? No, usually features.
w_fused_B = cat([w1_B, w3_B], dim=1)  # Output dimension doubles
```
*Note: The actual implementation in `_fuse_glu_lora` ensures correct dimensional concatenation to match `dim_out * 2`.*

#### 3. Output Projection (`w2`)
The `w2` layer (Down projection) is a standard linear layer appearing at `feed_forward.net.2`. It requires no special fusion, just correct mapping.

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

---

## 6. How to Maintain
*   **New Architectures**: If Nunchaku adds new model types, check if they use `SwiGLU`. If so, ensure the regex targets `.proj`.
*   **Updates to Diffusers**: If `diffusers` changes the `SwiGLU` internal name (unlikely, but possible) from `.proj` to `.linear`, `KEY_MAPPING` must be updated.

