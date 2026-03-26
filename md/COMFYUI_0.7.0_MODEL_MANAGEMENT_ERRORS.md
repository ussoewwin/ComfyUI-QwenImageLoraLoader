**ComfyUI 0.7.0 Model Management Errors - Complete Documentation**

**Overview**

After the ComfyUI 0.7.0 update, weakref-based model lifecycle management was introduced. This causes self.model (a weakref) in the LoadedModel class to potentially become None due to garbage collection, leading to multiple NoneType errors.

This document details all 13 fixes that should be applied to ComfyUI/comfy/model_management.py.

**Fix 1: is_dead() Method**

**Error Cause:**
When self.real_model is None, calling self.real_model() raises TypeError: 'NoneType' object is not callable.

**Before:**
```
def is_dead(self):
    return self.real_model() is not None and self.model is None
```

**After:**
```
def is_dead(self):
    if self.real_model is None:
        return False
    return self.real_model() is not None and self.model is None
```

**Meaning:**
If real_model is None, it is considered not yet dead (either before initialization or already cleaned up).

**Fix 2: model_memory() Method**

**Error Cause:**
When self.model is None, calling self.model.model_size() raises AttributeError: 'NoneType' object has no attribute 'model_size'.

**Before:**
```
def model_memory(self):
    return self.model.model_size()
```

**After:**
```
def model_memory(self):
    if self.model is None:
        return 0
    return self.model.model_size()
```

**Meaning:**
If the model is None, return 0 as the memory usage.

**Fix 3: model_loaded_memory() Method**

**Error Cause:**
When self.model is None, calling self.model.loaded_size() raises AttributeError: 'NoneType' object has no attribute 'loaded_size'.

**Before:**
```
def model_loaded_memory(self):
    return self.model.loaded_size()
```

**After:**
```
def model_loaded_memory(self):
    if self.model is None:
        return 0
    return self.model.loaded_size()
```

**Meaning:**
If the model is None, return 0 as the loaded memory.

**Fix 4: model_offloaded_memory() Method**

**Error Cause:**
When self.model is None, calling self.model.model_size() or self.model.loaded_size() raises AttributeError.

**Before:**
```
def model_offloaded_memory(self):
    return self.model.model_size() - self.model.loaded_size()
```

**After:**
```
def model_offloaded_memory(self):
    if self.model is None:
        return 0
    return self.model.model_size() - self.model.loaded_size()
```

**Meaning:**
If the model is None, return 0 as the offloaded memory.

**Fix 5: model_memory_required() Method**

**Error Cause:**
When self.model is None, calling self.model.current_loaded_device() raises AttributeError.

**Before:**
```
def model_memory_required(self, device):
    if device == self.model.current_loaded_device():
        return self.model_offloaded_memory()
    else:
        return self.model_memory()
```

**After:**
```
def model_memory_required(self, device):
    if self.model is None:
        return 0
    if device == self.model.current_loaded_device():
        return self.model_offloaded_memory()
    else:
        return self.model_memory()
```

**Meaning:**
If the model is None, return 0 as the required memory.

**Fix 6: model_load() Method**

**Error Cause:**
When self.model is None, calling self.model.model_patches_to() raises AttributeError.

**Before:**
```
def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
    self.model.model_patches_to(self.device)
    self.model.model_patches_to(self.model.model_dtype())
    # ... rest of the code
```

**After:**
```
def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
    if self.model is None:
        return None
    self.model.model_patches_to(self.device)
    self.model.model_patches_to(self.model.model_dtype())
    # ... rest of the code
```

**Meaning:**
If the model is None, return None for early return. The caller should check the return value.

**Fix 7: model_unload() Method**

**Error Cause:**
When self.model is None, calling self.model.loaded_size() or self.model.detach() raises AttributeError. Also, when self.model_finalizer is None, calling self.model_finalizer.detach() raises AttributeError.

**Before:**
```
def model_unload(self, memory_to_free=None, unpatch_weights=True):
    if memory_to_free is not None:
        if memory_to_free < self.model.loaded_size():
            freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
            if freed >= memory_to_free:
                return False
    self.model.detach(unpatch_weights)
    self.model_finalizer.detach()
    self.model_finalizer = None
    self.real_model = None
    return True
```

**After:**
```
def model_unload(self, memory_to_free=None, unpatch_weights=True):
    if self.model is None:
        return True
    if memory_to_free is not None:
        if memory_to_free < self.model.loaded_size():
            freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
            if freed >= memory_to_free:
                return False
    self.model.detach(unpatch_weights)
    if self.model_finalizer is not None:
        self.model_finalizer.detach()
    self.model_finalizer = None
    self.real_model = None
    return True
```

**Meaning:**
If the model is already None, consider it already unloaded and return True. If model_finalizer is None, do not call detach().

**Fix 8: model_use_more_vram() Method**

**Error Cause:**
When self.model is None, calling self.model.partially_load() raises AttributeError.

**Before:**
```
def model_use_more_vram(self, extra_memory, force_patch_weights=False):
    return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)
```

**After:**
```
def model_use_more_vram(self, extra_memory, force_patch_weights=False):
    if self.model is None:
        return False
    return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)
```

**Meaning:**
If the model is None, return False to indicate that VRAM usage cannot be increased.

**Fix 9: should_reload_model() Method**

**Error Cause:**
When self.model is None, calling self.model.lowvram_patch_counter() raises AttributeError.

**Before:**
```
def should_reload_model(self, force_patch_weights=False):
    if force_patch_weights and self.model.lowvram_patch_counter() > 0:
        return True
    return False
```

**After:**
```
def should_reload_model(self, force_patch_weights=False):
    if self.model is None:
        return False
    if force_patch_weights and self.model.lowvram_patch_counter() > 0:
        return True
    return False
```

**Meaning:**
If the model is None, consider reloading not necessary.

**Fix 10: __eq__() Method (Important: Implementation that does not change semantics)**

**Note:**
__eq__() is used in set operations such as keep_loaded. Changing its semantics may break the logic in free_memory().

**Fix Policy:**
Keep the original implementation. Adding None checks may cause situations where the same object is not considered equal, potentially causing models that should be in keep_loaded to not match.

**Current Implementation (Maintained):**
```
def __eq__(self, other):
    return self.model is other.model
```

**Meaning:**
By using self.model is other.model, it returns True when both are None. This maintains the original behavior and preserves consistency in set operations.

**Fix 11: load_models_gpu() Function - Multiple Locations**

**Error Cause:**
When loaded_model.model is None, calling is_clone() or detach() raises AttributeError. Also, when model_finalizer is None, calling detach() raises AttributeError.

**Fix Location 1 (Pre-load check):**
```
for loaded_model in models_to_load:
    if loaded_model.model is None:
        continue
    to_unload = []
    for i in range(len(current_loaded_models)):
        if current_loaded_models[i].model is not None and loaded_model.model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload
    for i in to_unload:
        model_to_unload = current_loaded_models.pop(i)
        if model_to_unload.model is not None:
            model_to_unload.model.detach(unpatch_all=False)
        if model_to_unload.model_finalizer is not None:
            model_to_unload.model_finalizer.detach()
```

**Fix Location 2 (Memory calculation):**
```
total_memory_required = {}
for loaded_model in models_to_load:
    if loaded_model.model is not None:
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)
```

**Fix Location 3 (Model loading):**
```
for loaded_model in models_to_load:
    model = loaded_model.model
    if model is None:
        continue
    torch_dev = model.load_device
    # ... rest of the code
```

**Meaning:**
If the model is None, skip that model and continue processing.

**Fix 12: free_memory() Function - Multiple Locations**

**Error Cause:**
When shift_model.model is None, calling model_offloaded_memory() or model_memory() internally raises AttributeError. Also, passing None to sys.getrefcount() may cause issues.

**Fix Location 1 (Unload candidate determination):**
```
if shift_model not in keep_loaded and not shift_model.is_dead():
    if shift_model.model is not None:
        can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
        shift_model.currently_used = False
```

**Fix Location 2 (Log output):**
```
if current_loaded_models[i].model is not None:
    logging.debug(f"Unloading {current_loaded_models[i].model.model.__class__.__name__}")
if current_loaded_models[i].model_unload(memory_to_free):
```

**Meaning:**
If the model is None, exclude it from unload candidates and skip log output.

**Fix 13: cleanup_models_gc() Function - Multiple Locations**

**Error Cause:**
When cur.real_model() returns None, accessing cur.real_model().__class__.__name__ raises AttributeError.

**Before:**
```
if cur.is_dead():
    logging.info("Potential memory leak detected with model {}, doing a full garbage collect, for maximum performance avoid circular references in the model code.".format(cur.real_model().__class__.__name__))
    do_gc = True
    break
```

**After:**
```
if cur.is_dead():
    if cur.real_model is not None and cur.real_model() is not None:
        logging.info("Potential memory leak detected with model {}, doing a full garbage collect, for maximum performance avoid circular references in the model code.".format(cur.real_model().__class__.__name__))
    do_gc = True
    break
```

**Fix Location 2:**
```
if cur.is_dead():
    if cur.real_model is not None and cur.real_model() is not None:
        logging.warning("WARNING, memory leak with model {}. Please make sure it is not being referenced from somewhere.".format(cur.real_model().__class__.__name__))
```

**Meaning:**
If real_model is None, skip log output but still execute garbage collection.

**Summary**

**Applied Fixes:**
1. is_dead() - None check for real_model
2. model_memory() - None check for model
3. model_loaded_memory() - None check for model
4. model_offloaded_memory() - None check for model
5. model_memory_required() - None check for model
6. model_load() - None check for model (returns None)
7. model_unload() - None checks for model and model_finalizer
8. model_use_more_vram() - None check for model
9. should_reload_model() - None check for model
10. __eq__() - Original implementation maintained to avoid changing semantics
11. load_models_gpu() - Multiple None checks for model and model_finalizer
12. free_memory() - None check for model
13. cleanup_models_gc() - None check for real_model

**Important Notes:**
For __eq__(), the original implementation is maintained to avoid changing semantics. Adding None checks may break consistency in set operations.
model_load() may return None, so callers should check the return value.

