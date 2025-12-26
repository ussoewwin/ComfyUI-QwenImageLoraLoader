# Fix: Multiple NoneType Errors in model_management.py Caused by Weakref-Based Model Lifecycle (10 Critical Locations)

## Summary

This PR fixes multiple NoneType errors occurring across various nodes in ComfyUI 0.4.0 due to weakref-based model lifetime management.

The root cause is that `LoadedModel.model` may legitimately return `None` when the underlying `ModelPatcher` has been garbage collected—but several core functions do not guard against this scenario.

This results in unpredictable crashes such as:

- `TypeError: 'NoneType' object is not callable`
- `AttributeError: 'NoneType' object has no attribute 'weights'`
- `AttributeError: 'NoneType' object has no attribute '__deepcopy__'`

I identified 10 critical sites in `model_management.py` where `lm.model` must be checked for `None` before dereferencing.

Applying these checks eliminated every crash in my environment and also resolved multiple user-reported node failures (e.g., Issue #25).

## Technical Root Cause

### 1. Weakref-based model ownership (0.4.0 change)

ComfyUI 0.4.0 introduced:

```python
self._model = weakref.ref(model)
```

**Meaning:**

`LoadedModel.model → returns None after GC`

However, several core functions assume the model always exists.

### 2. Missing None checks in core memory/model routines

Affected functions include:

- `is_dead()`
- `model_memory()`
- `model_loaded_memory()`
- `model_offloaded_memory()`
- `model_memory_required()`
- `model_unload()`
- `load_models_gpu()` (3 locations)
- `free_memory()`

All of these iterate over `LoadedModel` objects and dereference `lm.model` without confirming it is non-`None`.

When GC collects a `ModelPatcher` during unloading, this leads to immediate crashes inside unrelated nodes.

### 3. Why this causes widespread node failures

When `LoadedModel.model` becomes `None`, the following operations fail across the system:

- memory estimation
- model pinning
- deepcopy
- device transition
- unload/load decisions

This produces dozens of different stack traces depending on timing, making it appear as though individual nodes are broken.

In reality, the issue comes from stale `LoadedModel` objects referencing collected `ModelPatchers`.

## Fix

This PR adds explicit:

```python
if lm.model is None:
    continue
```

or equivalent early returns in all affected locations.

After applying the patch:

- ✔ No remaining NoneType crashes
- ✔ Memory management behaves deterministically
- ✔ Model load/unload cycles no longer poison the cache state
- ✔ Third-party nodes stop failing as a side effect
- ✔ User-reported issue #25 (and similar) no longer reproduce

## Why this patch is necessary

Weakref-based model ownership is correct by design.

However:

**Any weakref dereference must treat `None` as a valid and expected state.**

This invariant is currently violated in several core paths.

This PR ensures:

- Garbage-collected models no longer corrupt the model management pipeline
- All memory and lifecycle operations are robust against `None`
- `LoadedModel` remains safe even when backing objects are destroyed

I can provide:

- A complete diff
- Additional tests if needed
- Validation logs demonstrating stability after patching

Thank you for your work on ComfyUI—this patch should greatly stabilize the 0.4.0 model management system.

