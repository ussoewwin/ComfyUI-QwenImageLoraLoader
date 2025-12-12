# ComfyUI 0.4.0 Update: Fixes for Various Errors in model_management.py

## Root Cause of Errors

In ComfyUI's `model_management.py`, after models were unloaded or garbage collected, code accessed methods or attributes on `None` objects, causing `TypeError` and `AttributeError`.

## Modified File

**File**: `ComfyUI/comfy/model_management.py`

## Detailed Fixes

### Fix 1: `is_dead()` Method (Lines 597-600)

**Error Cause:**
```python
# Before
def is_dead(self):
    return self.real_model() is not None and self.model is None
```
`model_unload()` sets `self.real_model = None`, but `is_dead()` calls `self.real_model()`, causing `TypeError: 'NoneType' object is not callable`.

**After:**
```python
def is_dead(self):
    if self.real_model is None:
        return False
    return self.real_model() is not None and self.model is None
```

**Meaning:**
If `self.real_model` is `None`, return `False`; otherwise call it as a `weakref.ref`.

---

### Fix 2: `model_memory()` Method (Lines 526-529)

**Error Cause:**
```python
# Before
def model_memory(self):
    return self.model.model_size()
```
Calling `model_size()` when `self.model` is `None` (garbage collected) causes `AttributeError: 'NoneType' object has no attribute 'model_size'`.

**After:**
```python
def model_memory(self):
    if self.model is None:
        return 0
    return self.model.model_size()
```

**Meaning:**
If `self.model` is `None`, return `0` (already freed, so memory usage is 0).

---

### Fix 3: `model_loaded_memory()` Method (Lines 531-534)

**Error Cause:**
```python
# Before
def model_loaded_memory(self):
    return self.model.loaded_size()
```
Calling `loaded_size()` when `self.model` is `None` causes an error.

**After:**
```python
def model_loaded_memory(self):
    if self.model is None:
        return 0
    return self.model.loaded_size()
```

**Meaning:**
If `self.model` is `None`, return `0`.

---

### Fix 4: `model_offloaded_memory()` Method (Lines 536-539)

**Error Cause:**
```python
# Before
def model_offloaded_memory(self):
    return self.model.model_size() - self.model.loaded_size()
```
Calling `model_size()` when `self.model` is `None` causes an error.

**After:**
```python
def model_offloaded_memory(self):
    if self.model is None:
        return 0
    return self.model.model_size() - self.model.loaded_size()
```

**Meaning:**
If `self.model` is `None`, return `0`.

---

### Fix 5: `model_memory_required()` Method (Lines 541-547)

**Error Cause:**
```python
# Before
def model_memory_required(self, device):
    if device == self.model.current_loaded_device():
        return self.model_offloaded_memory()
    else:
        return self.model_memory()
```
Calling `current_loaded_device()` when `self.model` is `None` causes an error.

**After:**
```python
def model_memory_required(self, device):
    if self.model is None:
        return 0
    if device == self.model.current_loaded_device():
        return self.model_offloaded_memory()
    else:
        return self.model_memory()
```

**Meaning:**
If `self.model` is `None`, return `0`.

---

### Fix 6: `model_unload()` Method (Lines 574-583)

**Error Cause:**
```python
# Before
def model_unload(self, memory_to_free=None, unpatch_weights=True):
    # ...
    self.model.detach(unpatch_weights)
    self.model_finalizer.detach()  # ← Error here
    self.model_finalizer = None
    self.real_model = None
    return True
```
Calling `detach()` when `self.model_finalizer` is `None` causes `AttributeError: 'NoneType' object has no attribute 'detach'`.

**After:**
```python
def model_unload(self, memory_to_free=None, unpatch_weights=True):
    # ...
    self.model.detach(unpatch_weights)
    if self.model_finalizer is not None:
        self.model_finalizer.detach()
    self.model_finalizer = None
    self.real_model = None
    return True
```

**Meaning:**
Call `detach()` only if `self.model_finalizer` is not `None`.

---

### Fix 7: `load_models_gpu()` Method - Clone Detection (Lines 710-722)

**Error Cause:**
```python
# Before
for loaded_model in models_to_load:
    to_unload = []
    for i in range(len(current_loaded_models)):
        if loaded_model.model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload
    for i in to_unload:
        model_to_unload = current_loaded_models.pop(i)
        model_to_unload.model.detach(unpatch_all=False)
        model_to_unload.model_finalizer.detach()  # ← Error
```
Calling methods when `loaded_model.model` or `model_to_unload.model_finalizer` is `None` causes errors.

**After:**
```python
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

**Meaning:**
- Skip if `loaded_model.model` is `None`
- Don't call `is_clone()` if `current_loaded_models[i].model` is `None`
- Don't call `detach()` if `model_to_unload.model` is `None`
- Don't call `detach()` if `model_to_unload.model_finalizer` is `None`

---

### Fix 8: `load_models_gpu()` Method - Memory Calculation (Lines 724-727)

**Error Cause:**
```python
# Before
total_memory_required = {}
for loaded_model in models_to_load:
    total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)
```
If `loaded_model.model` is `None`, `model_memory_required()` may error.

**After:**
```python
total_memory_required = {}
for loaded_model in models_to_load:
    if loaded_model.model is not None:
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)
```

**Meaning:**
Exclude models where `loaded_model.model` is `None` from memory calculation (already garbage collected).

---

### Fix 9: `load_models_gpu()` Method - Model Loading (Lines 740-744)

**Error Cause:**
```python
# Before
for loaded_model in models_to_load:
    model = loaded_model.model
    torch_dev = model.load_device  # ← Error
```
Accessing `load_device` when `loaded_model.model` is `None` causes an error.

**After:**
```python
for loaded_model in models_to_load:
    model = loaded_model.model
    if model is None:
        continue
    torch_dev = model.load_device
```

**Meaning:**
Skip if `model` is `None`.

---

### Fix 10: `free_memory()` Function (Lines 641-647)

**Error Cause:**
```python
# Before
for i in range(len(current_loaded_models) -1, -1, -1):
    shift_model = current_loaded_models[i]
    if shift_model.device == device:
        if shift_model not in keep_loaded and not shift_model.is_dead():
            can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
```
Calling `sys.getrefcount(shift_model.model)` when `shift_model.model` is `None` causes an error. Adding `model`-`None` models to `can_unload` also makes memory calculation inaccurate.

**After:**
```python
for i in range(len(current_loaded_models) -1, -1, -1):
    shift_model = current_loaded_models[i]
    if shift_model.device == device:
        if shift_model not in keep_loaded and not shift_model.is_dead():
            if shift_model.model is not None:
                can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
                shift_model.currently_used = False
```

**Meaning:**
Exclude models where `shift_model.model` is `None` from `can_unload` (already garbage collected, so exclude from memory calculation).

---

## Effects of Fixes

1. **Error Prevention**: `None` checks prevent `TypeError` and `AttributeError`
2. **More Accurate Memory Calculation**: Excluding `model`-`None` models improves accuracy
3. **Better VRAM Management**: Skipping unnecessary models optimizes VRAM usage
4. **Faster Processing**: Fewer fallbacks to tiled encoding, enabling normal processing

These fixes stabilize ComfyUI's model management and reduce VRAM-related errors.

