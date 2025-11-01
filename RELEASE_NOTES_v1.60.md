# v1.60 — MAJOR UPDATE: Simplified Installation (No Integration Required!)

## Summary

**As of v1.60, ComfyUI-QwenImageLoraLoader is now a fully independent custom node that requires no integration with ComfyUI-nunchaku's `__init__.py`.**

* ✅ **Removed ComfyUI-nunchaku integration requirement** — No manual modification of `__init__.py` needed
* ✅ **Simplified installation** — Just `git clone` and restart ComfyUI
* ✅ **No batch scripts** — No installer/uninstaller scripts required
* ✅ **Automatic node registration** — ComfyUI's built-in mechanism handles everything
* ✅ **Backward compatible** — All existing LoRA files and workflows continue to work

## What Changed

### Installation (Before → After)

**Before v1.60:**
```
1. Clone repository
2. Choose installation script (global or portable Python)
3. Run batch file (modifies ComfyUI-nunchaku/__init__.py)
4. Restart ComfyUI
```

**After v1.60:**
```
1. Clone repository
2. Restart ComfyUI
Done!
```

---

# Why Integration with ComfyUI-nunchaku's `__init__.py` is Unnecessary - Complete Technical Explanation

## Chapter 1: Project Background and Initial Misunderstanding

### 1-1 Extraction from GavChap's Fork

ComfyUI-QwenImageLoraLoader was extracted from the "qwen-lora-suport-standalone" branch that GavChap created by forking ComfyUI-nunchaku. In GavChap's fork version, LoRA loader functionality for Qwen Image was added, which did not exist in the official ComfyUI-nunchaku. Subsequently, this QI-specific LoRA functionality was separated as an independent custom node and modified to be compatible with the official ComfyUI-nunchaku.

At the time of extraction, many developers believed that "external nodes require integration with the main body." This is the origin of this misunderstanding. In GavChap's fork, the QI LoRA node was written directly into ComfyUI-nunchaku's `__init__.py`. That design pattern was referenced, and after independent node creation, the idea that "nunchaku's main body should have integration code added" continued. However, this premise was actually wrong. When investigating the details of ComfyUI's specifications and implementation, it became clear that integration with the main body was unnecessary.

### 1-2 Comparison with FLUX Loader

Looking at the official ComfyUI-nunchaku, the FLUX LoRA loader is built-in. This is implemented at lines 45-50 of nunchaku's `__init__.py`:

```python
try:
    from .nodes.lora.flux import NunchakuFluxLoraLoader, NunchakuFluxLoraStack
    NODE_CLASS_MAPPINGS["NunchakuFluxLoraLoader"] = NunchakuFluxLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuFluxLoraStack"] = NunchakuFluxLoraStack
except ImportError:
    logger.exception("Nodes `NunchakuFluxLoraLoader` and `NunchakuFluxLoraStack` import failed:")
```

FLUX nodes are directly embedded. Meanwhile, QI LoRA is an external feature added by GavChap and does not exist in the official version. Why is FLUX internally embedded while QI is external? The reason is that FLUX support was planned from the initial stages of Nunchaku, whereas QI support is a later expansion.

The question then arises: does the QI LoRA loader, which exists as an external node, require integration with the main body to operate the same way as FLUX? The answer is **NO**. The reasons are explained below.

## Chapter 2: ComfyUI's Node Loading Specification

### 2-1 Automatic Loading Mechanism

ComfyUI's startup flow:

1. ComfyUI scans the `custom_nodes/` directory
2. Automatically executes `__init__.py` in each directory
3. Collects `NODE_CLASS_MAPPINGS` dictionaries if they exist
4. Merges all `NODE_CLASS_MAPPINGS` into one final node dictionary
5. Displays in UI

In other words, if the `ComfyUI-QwenImageLoraLoader/` directory has an `__init__.py` where `NODE_CLASS_MAPPINGS` is defined, ComfyUI automatically recognizes it. Integration code with the main body is completely unnecessary. This automatic discovery mechanism is the foundation of everything that follows.

### 2-2 How Multiple NODE_CLASS_MAPPINGS are Merged

ComfyUI's core implementation merges at startup as follows (conceptually):

1. Loads `NODE_CLASS_MAPPINGS` from nunchaku's internal implementation
2. Loads `NODE_CLASS_MAPPINGS` from QwenImageLoraLoader's internal implementation
3. Loads `NODE_CLASS_MAPPINGS` from all other custom nodes
4. Merges all of these into one large dictionary
5. Displays in UI

In other words, each node pair that has an independent `__init__.py` coexists without integration code.

## Chapter 3: Model Structure and Reference Relationships

### 3-1 ComfyUI's ModelPatcher

ComfyUI's basic model structure is ModelPatcher. This wraps the model and records patches (weight changes) while maintaining the original structure. In Nunchaku's case, this is extended with NunchakuModelPatcher.

### 3-2 NunchakuQwenImage's Structure

NunchakuQwenImage extends the official QwenImage. The important part is that `diffusion_model` is assigned `NunchakuQwenImageTransformer2DModel`. This is a Nunchaku type, but it is the location where ComfyUI's LoRA loader accesses.

### 3-3 LoRA Loader's Access Path

When the LoRA loader executes:

```python
def load_lora(self, model, lora_name: str, lora_strength: float):
    model_wrapper = model.model.diffusion_model
```

This `model` is `NunchakuModelPatcher`. `model.model` is `NunchakuQwenImage`. `model.model.diffusion_model` is `NunchakuQwenImageTransformer2DModel`.

The loader can automatically access the correct Transformer from the passed ModelPatcher. Integration code with the main body does not affect this access path.

### 3-4 Existing Pipeline

When QwenImage DiT Loader (model loading node) executes:

1. Loads model file
2. Calls `load_diffusion_model_state_dict`
3. Generates `NunchakuQwenImage` instance
4. Automatically assigns `NunchakuQwenImageTransformer2DModel` to `diffusion_model` within it
5. Wraps with `NunchakuModelPatcher`
6. Returns

When the LoRA loader receives this ModelPatcher, the correct structure is already established.

## Chapter 4: Complete Logic of LoRA Application

### 4-1 Mathematical Principle of LoRA

LoRA (Low-Rank Adaptation) adds fine-tuning through low-rank matrices without directly changing the model's original weights.

Original weight matrix: `W: (out_dim, in_dim)`

LoRA addition: `ΔW = α × B @ A`

* A: (in_dim, rank)
* B: (out_dim, rank)
* α: Scaling coefficient

In forward propagation: `output = (W + α × B @ A) @ input`

The original parameter W remains unchanged, and an additional low-rank term ΔW is added on top.

### 4-2 Nunchaku's LoRA Implementation

Nunchaku adopts a `_lora_slots` mechanism. Pre-allocated LoRA slots are reserved for specific layers of the model. Each slot has a tensor area for computation, where `α`, `A`, and `B` are stored.

### 4-3 compose_loras_v2 Processing

The LoRA composition function `compose_loras_v2(model, lora_configs)` performs:

1. Traverses the model and enumerates all `_lora_slots`
2. Loads LoRA files and extracts state dictionaries
3. Key mapping process normalizes formats like "transformer_blocks.0.attn.to_qkv.lora_b.weight" to actual model paths
4. For multiple LoRAs, adds with strength consideration
5. Assigns final `α, A, B` to each slot

This processing does not depend on model type. As long as the `_lora_slots` attribute exists, it processes mechanically.

### 4-4 ComfyQwenImageWrapper's Role

ComfyQwenImageWrapper:

1. Receives LoRA list from external sources
2. Detects changes
3. Calls `compose_loras_v2` if changes are detected
4. Executes model forward propagation (LoRA is applied at this point)

### 4-5 Loader's Invocation

The LoRA Loader node's `load_lora` method:

1. Extracts `model.model.diffusion_model` from the passed ModelPatcher
2. Imports `ComfyQwenImageWrapper`
3. Checks Transformer type
4. If not wrapped, adds wrapper and replaces `model.model.diffusion_model`
5. Performs deepcopy
6. Adds LoRA path and strength to `loras` list

The loader only "registers the LoRA"; actual application occurs automatically during inference.

### 4-6 Why Main Body Integration is Unnecessary

The complete flow:

1. **ComfyUI startup**: Loads `ComfyUI-QwenImageLoraLoader/__init__.py`, registers nodes
2. **Model loading**: QwenImage DiT Loader generates `NunchakuQwenImage`, assigns `NunchakuQwenImageTransformer2DModel` to `diffusion_model`
3. **LoRA addition**: QI LoRA Loader applies `ComfyQwenImageWrapper`, adds to `loras` list
4. **Inference**: Wrapper's `forward` calls `compose_loras_v2`, applies LoRA

This pipeline is completely independent. Without main body integration code, everything works through node `__init__.py` registration alone.

## Chapter 5: nunchaku_code's LoRA Implementation

### 5-1 Origin and Role of nunchaku_code/lora_qwen.py

`ComfyUI-QwenImageLoraLoader/nunchaku_code/lora_qwen.py` is extracted from GavChap's fork. It contains core LoRA composition logic.

Functions include:

* `compose_loras_v2(model, lora_configs)`: Composes multiple LoRAs
* `reset_lora_v2(model)`: Resets LoRA state
* `_load_lora_state_dict(lora_path_or_dict)`: Loads LoRA files
* `_classify_and_map_key(key)`: Key mapping
* `_get_module_by_path(model, module_key)`: Gets modules
* `_apply_lora_to_slot(slot, lora_config)`: Applies to slots

### 5-2 Relationship with Main Body

nunchaku_code is independent logic extracted from GavChap. External nunchaku package dependencies are only conversion utilities. There is no dependency on the main body's `__init__.py`. The implementation is completely self-contained.

## Chapter 6: Implementation Independence

### 6-1 File Structure Completeness

Each layer in the file structure is completely self-contained:

* `__init__.py` for node registration
* `nodes/lora/qwenimage.py` for loader body
* `wrappers/qwenimage.py` for LoRA composition wrapper
* `nunchaku_code/lora_qwen.py` for LoRA composition logic
* `js/widgethider.js` for UI dynamic control

### 6-2 Dependency Minimization

External dependencies are only:

* **nunchaku** (types and utilities)
* **torch** (basic library)
* **ComfyUI** (node system)

**Zero dependency on main ComfyUI-nunchaku.**

## Chapter 7: Conclusion and Final Reasoning

**ComfyUI-nunchaku's `__init__.py` integration is unnecessary for these reasons:**

**First:** Automatic node loading exists. ComfyUI's specification of auto-scanning `custom_nodes/` and merging `NODE_CLASS_MAPPINGS` makes main body additions unnecessary.

**Second:** Model structure independence exists. QwenImage DiT Loader already assigns the correct type to `diffusion_model`. The LoRA loader just accesses that path.

**Third:** External type import exists. `NunchakuQwenImageTransformer2DModel` can be directly imported from the nunchaku package. Type detection and processing are realized without main body modification.

**Fourth:** Composition logic generality exists. `compose_loras_v2` is model-independent. It can be applied to any model with `_lora_slots`.

**Fifth:** Wrapper mechanism exists. `ComfyQwenImageWrapper` handles change detection and composition. This realizes loose coupling with the main body.

**In conclusion:** The original assessment that "integration code is mandatory" was wrong. Based on ComfyUI and Nunchaku's design, the LoRA loader as an external node **operates completely independently.** Main body integration is unnecessary, and registration through `ComfyUI-QwenImageLoraLoader/__init__.py` alone is sufficient for full functionality.

---

## Installation Instructions

### Quick Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader.git
```

### Requirements

* Python 3.11+
* ComfyUI (latest version)
* ComfyUI-nunchaku plugin (for Qwen Image support)
* CUDA-capable GPU (optional, recommended for performance)

## Upgrade from v1.57 or Earlier

If you have v1.57 or earlier installed:

1. **If you used the installer scripts before:**
   * The integration code is already in your ComfyUI-nunchaku `__init__.py`
   * It's safe to leave it there (it will simply be ignored)
   * You can optionally remove the integration code

2. **To upgrade to v1.60:**
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-QwenImageLoraLoader
   git pull origin main
   ```
   Restart ComfyUI

3. **Optional: Clean up old integration code**
   * Edit `ComfyUI-nunchaku/__init__.py`
   * Search for "ComfyUI-QwenImageLoraLoader Integration"
   * Delete that entire try/except block
   * Restart ComfyUI

## Backward Compatibility

* ✅ All v1.57 and earlier LoRA files work without modification
* ✅ All existing workflows work without modification
* ✅ Old integration code in ComfyUI-nunchaku `__init__.py` is safely ignored

## Known Issues

* **RES4LYF Sampler**: Not supported due to device mismatch issues (Issue #7, #8)
   * Workaround: Use other samplers
* **LoRA Stack UI**: 10th row always visible (Issue #9)
   * Visual only; does not affect functionality

## Special Thanks

* GavChap for the original LoRA composition implementation
* Nunchaku team for the underlying model and infrastructure
* Community members for testing and feedback
