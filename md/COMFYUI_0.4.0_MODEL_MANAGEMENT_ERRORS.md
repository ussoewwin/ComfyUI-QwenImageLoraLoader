# Issue #25: ComfyUI 0.4.0 Model Management Errors

- **Issue**: After the ComfyUI 0.4.0 update, multiple nodes (including this one in some environments) experienced errors such as `TypeError: 'NoneType' object is not callable` and `AttributeError: 'NoneType' object has no attribute`. In our environment, we resolved these errors by modifying ComfyUI's core `model_management.py`. Note that in our environment, these errors did not occur with this node (ComfyUI-QwenImageLoraLoader). Nunchaku library and ComfyUI-Nunchaku nodes should use the latest versions. If errors persist even after applying the latest version of this node (ComfyUI-QwenImageLoraLoader), modification of ComfyUI's core `model_management.py` may be necessary.
- **Root Cause**: In ComfyUI 0.4.0, ComfyUI's core `model_management.py` lacks sufficient None checks, causing `TypeError` and `AttributeError` when accessing methods or attributes on objects that became `None` after models were unloaded or garbage collected. This problem is not a bug in individual nodes, but rather a structural issue in ComfyUI 0.4.0's model management (`model_management.py`).
- **GC Changes in ComfyUI 0.4.0**: Compared to ComfyUI 0.3.x, automatic model unloading occurs earlier, making the following flow more likely:

```
ModelPatcher → GC → weakref(None)
```

This also explains why the occurrence of the issue varies by user environment.
- **Technical Basis**:
  1. **Multiple Locations with "Missing None Checks"** - This is not a bug in individual nodes, but the core main logic crashes when accessing attributes on `None`. The added fixes like `if model is None: continue` are defensive code that ComfyUI core should have in all paths.
  2. **Post-Weak-Reference GC Behavior Not Considered** - The introduction of `LoadedModel._model = weakref.ref(ModelPatcher)` in ComfyUI 0.4.0 is a breaking change. When the weak reference target is garbage collected, it returns `None`, but this is not handled. Post-processing for the breaking update is incomplete.
  3. **Multiple Nodes Were Affected in a Chain Reaction** - This is not a problem with nodes, but multiple nodes were affected in a chain reaction due to core behavior changes. Model loading/unloading, memory calculation, GPU/CPU offloading, and `ModelPatcher` lifecycle are all controlled by ComfyUI core.
  4. **All Fix Locations Are Core Responsibility Areas** - The locations fixed (`model_memory`, `model_offloaded_memory`, `load_models_gpu`, `free_memory`, `model_unload`, `is_dead` checks, etc.) are all ComfyUI core functions. These are not areas that node developers should touch. The fact that all fix locations are core logic leaves no explanation other than a core defect.
  5. **Result of Applying Fixes** - After applying None check fixes to ComfyUI's core `model_management.py` in our environment, similar errors were resolved. This demonstrates that the problem can be solved by adding defensive code that the core should have.
- **Model Lifecycle and ModelPatcher Initialization Relationship**:
  - **Fact 1: Relationship between LoadedModel and ModelPatcher** - The `LoadedModel` class (lines 502-524 in ComfyUI's `model_management.py`) holds a weak reference to `ModelPatcher`:

```python
def _set_model(self, model):
    self._model = weakref.ref(model)  # Weak reference to ModelPatcher

@property
def model(self):
    return self._model()  # Returns None when garbage collected
```

  - **Fact 2: ModelPatcher Initialization** - In the `__init__` of the `ModelPatcher` class (lines 215-237 in `model_patcher.py`), the `pinned` attribute is initialized:

```python
def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
    # ...
    self.pinned = set()  # Line 237: Initialized
```

  - **Fact 3: Fix Content in ComfyUI Core's model_management.py** - The fix in ComfyUI's core `model_management.py` now skips `LoadedModel` instances where `model` is `None`: In `load_models_gpu()`, skips `LoadedModel` instances where `model` is `None` (lines 712, 727, 743); In `free_memory()`, excludes `LoadedModel` instances where `model` is `None` (line 646).
  - **Fact 4: Problem Before Fix** - `LoadedModel` holds a weak reference to `ModelPatcher`. When garbage collected, `LoadedModel.model` returns `None`. Before the fix, methods were called on `LoadedModel` instances where `model` was `None`, causing errors.
  - **Fact 5: Behavior After Fix** - By skipping `LoadedModel` instances where `model` is `None`, errors do not occur. Because errors do not occur, processing continues normally.
  - **Fact 6: Why copy.deepcopy Fails** - `copy.deepcopy` fails because references to GC'd `ModelPatcher` instances remain in the dictionary being deepcopied. When these references are accessed, they return `None`, causing deepcopy to stop.
  - **Fact 7: Confirmation Items** - After applying the fix, `copy.deepcopy` and `pinned` attribute errors do not occur in our environment. Nunchaku library and ComfyUI-Nunchaku nodes should use the latest versions, but this may still be insufficient. While these errors did not occur with this node (ComfyUI-QwenImageLoraLoader) in our environment, the fix to ComfyUI's core `model_management.py` may have indirectly affected it, making errors less likely to occur.
- **Important Note: Not a Problem with Nunchaku Library** - This problem is not caused by the Nunchaku library's implementation. Nunchaku's `model_config` and `ModelPatcher` itself are normal. The problem is in the upstream = ComfyUI core's `model_management.py` GC processing.
- **Speculation (Items That May Be Environment-Dependent)**: The fix allows `ModelPatcher` initialization to complete normally. As a result, the `pinned` attribute is also properly initialized. Accessing `self.pinned` in `__del__` does not cause errors.
- **Recommendations**:
  1. Update Nunchaku library and ComfyUI-Nunchaku nodes to the latest version (addresses `model_config` issues)
  2. Consider applying None check fixes to ComfyUI's core `model_management.py` (may address the root cause)
- **Note**: This is the first support measure. I have published the technical details of the fixes I applied to ComfyUI's core `model_management.py` in my environment. See [COMFYUI_0.4.0_UPDATE_ERROR_FIXES.md](./COMFYUI_0.4.0_UPDATE_ERROR_FIXES.md) for details. Note that these fixes were applied in my specific environment and may not work universally in all environments. This may also resolve `copy.deepcopy` and `pinned` attribute errors.

