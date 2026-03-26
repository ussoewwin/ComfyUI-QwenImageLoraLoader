# Fix: AttributeError 'Logger' object has no attribute 'mgpu_mm_log'

## Overview

This document provides a comprehensive explanation of the fix for the `AttributeError: 'Logger' object has no attribute 'mgpu_mm_log'` error that occurred in `ComfyUI-nunchaku-unofficial-loader` when executing prompts in ComfyUI. The error caused prompt execution to crash, preventing users from running workflows.

---

## Error Details

### Complete Error Stack Trace

```
Exception in thread Thread-16 (prompt_worker):
Traceback (most recent call last):
  File "threading.py", line 1044, in _bootstrap_inner
  File "threading.py", line 995, in run
  File "D:\USERFILES\ComfyUI\ComfyUI\main.py", line 271, in prompt_worker
    comfy.model_management.soft_empty_cache()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-nunchaku-unofficial-loader\device_utils.py", line 251, in soft_empty_cache_distorch2_patched
    check_cpu_memory_threshold()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-nunchaku-unofficial-loader\model_management_mgpu.py", line 158, in check_cpu_memory_threshold
    multigpu_memory_log("cpu_monitor", f"trigger:{current_usage:.1f}pct")
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-nunchaku-unofficial-loader\model_management_mgpu.py", line 106, in multigpu_memory_log
    logger.mgpu_mm_log(f"{ts_str} {tag_padded} {' '.join(parts)}")
    ^^^^^^^^^^^^^^^^^^
AttributeError: 'Logger' object has no attribute 'mgpu_mm_log'
```

### Error Flow Analysis

1. **Entry Point**: ComfyUI's `main.py` calls `comfy.model_management.soft_empty_cache()` during prompt execution
2. **Patched Function**: The call is intercepted by `soft_empty_cache_distorch2_patched()` in `device_utils.py` (line 251)
3. **CPU Monitoring**: The function calls `check_cpu_memory_threshold()` to monitor CPU memory usage
4. **Memory Logging**: `check_cpu_memory_threshold()` calls `multigpu_memory_log()` to record memory snapshots
5. **Error Location**: `multigpu_memory_log()` attempts to call `logger.mgpu_mm_log()` which doesn't exist
6. **Result**: `AttributeError` crashes the prompt execution thread

### Root Cause

The error occurred because the code attempted to call `logger.mgpu_mm_log()`, a **custom method that does not exist** on Python's standard `logging.Logger` class. 

**Python's Standard Logging Methods:**
- `logger.debug()` - Detailed diagnostic information
- `logger.info()` - General informational messages  
- `logger.warning()` - Warning messages
- `logger.error()` - Error messages
- `logger.critical()` - Critical error messages

The `mgpu_mm_log()` method was likely intended as a custom logging method for multi-GPU memory management, but it was **never implemented**. The code attempted to use this non-existent method, causing the `AttributeError`.

### When It Occurs

The error is triggered during:
1. **Prompt execution** in ComfyUI - when processing workflows
2. **Memory management operations** - when `soft_empty_cache()` is called to free memory
3. **CPU memory threshold checking** - when the system monitors CPU memory usage and needs to log snapshots
4. **Multi-device cache clearing** - when clearing GPU caches across multiple devices

---

## Solution

### Fix Strategy

Replace all instances of `logger.mgpu_mm_log()` with `logger.info()`, which is the standard logging method for informational messages. This maintains the same logging functionality while using a method that actually exists.

### Why `logger.info()` is Appropriate

1. **Standard Method**: It's a standard method available on all `Logger` instances
2. **Appropriate Level**: The logged messages are informational in nature (memory snapshots, cleanup operations, cache clearing)
3. **Consistent Behavior**: All messages maintain the same logging level and visibility
4. **No Breaking Changes**: The log output format and content remain identical, only the method name changes
5. **Compatibility**: Works with all Python logging configurations and handlers

### Modified Files

| File | Location | Total Changes | Status |
|------|----------|--------------|--------|
| `model_management_mgpu.py` | `ComfyUI-nunchaku-unofficial-loader` | 7 instances | ✅ Fixed |
| `device_utils.py` | `ComfyUI-nunchaku-unofficial-loader` | ~15 instances | ✅ Fixed |
| `wrappers.py` | `ComfyUI-nunchaku-unofficial-loader` | ~7 instances | ✅ Fixed |

**Total**: Approximately **30+ instances** replaced across 3 files.

---

## Detailed Code Changes

### File 1: `model_management_mgpu.py`

This file contains the core memory management and logging infrastructure for multi-GPU operations.

#### Change 1: Memory Summary Logging (Line 73)

**Location**: `multigpu_memory_log()` function, when `identifier == "print_summary"`

**Before:**
```python
logger.mgpu_mm_log(f"=== memory summary: {id_key} ===")
```

**After:**
```python
logger.info(f"=== memory summary: {id_key} ===")
```

**Context:**
This occurs when printing memory summaries for debugging and monitoring purposes. The function iterates through stored memory snapshots and prints formatted summaries.

**Function Flow:**
```python
def multigpu_memory_log(identifier, tag):
    if identifier == "print_summary":
        for id_key in sorted(_MEM_SNAPSHOT_SERIES.keys()):
            series = _MEM_SNAPSHOT_SERIES[id_key]
            logger.info(f"=== memory summary: {id_key} ===")  # ← Fixed here
            # ... rest of the function
```

#### Change 2: Timestamped Memory Logging in Summary (Line 83)

**Location**: `multigpu_memory_log()` function, within the summary printing loop

**Before:**
```python
logger.mgpu_mm_log(f"{ts_str} {tag_padded} {' '.join(parts)}")
```

**After:**
```python
logger.info(f"{ts_str} {tag_padded} {' '.join(parts)}")
```

**Context:**
This logs individual timestamped memory snapshots within the summary. Each snapshot includes CPU and GPU device memory usage in a formatted string.

**Format Example:**
```
2025-01-22T10:30:45.123Z cpu_monitor_trigger:85.5pct  cpu|12.34 cuda:0|8.90 cuda:1|7.65
```

#### Change 3: Main Memory Logging (Line 106)

**Location**: `multigpu_memory_log()` function, main logging path

**Before:**
```python
logger.mgpu_mm_log(f"{ts_str} {tag_padded} {' '.join(parts)}")
```

**After:**
```python
logger.info(f"{ts_str} {tag_padded} {' '.join(parts)}")
```

**Context:**
This is the **primary logging call** in `multigpu_memory_log()` that records memory snapshots. It's called every time a memory snapshot is taken, recording:
- Timestamp (UTC)
- Identifier and tag (padded for alignment)
- CPU memory usage (in GB)
- GPU device memory usage for each device (in GB)

**Complete Function Context:**
```python
def multigpu_memory_log(identifier, tag):
    """Record timestamped memory snapshot with clean aligned logging"""
    # ... capture memory snapshot ...
    ts = datetime.now(timezone.utc)
    curr = _capture_memory_snapshot()
    
    # Store in series
    if identifier not in _MEM_SNAPSHOT_SERIES:
        _MEM_SNAPSHOT_SERIES[identifier] = []
    _MEM_SNAPSHOT_SERIES[identifier].append((ts, tag, curr))
    
    # Format and log
    ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    tag_padded = f"{identifier}_{tag}".ljust(35)
    
    parts = []
    cpu_used, _ = curr.get("cpu", (0, 0))
    parts.append(f"cpu|{cpu_used/(1024**3):.2f}")
    
    for dev in sorted([k for k in curr.keys() if k != "cpu"]):
        used, _ = curr[dev]
        parts.append(f"{dev}|{used/(1024**3):.2f}")
    
    logger.info(f"{ts_str} {tag_padded} {' '.join(parts)}")  # ← Fixed here
    
    _MEM_SNAPSHOT_LAST[identifier] = (tag, curr)
```

#### Change 4: Cleanup Request Logging (Line 171)

**Location**: `force_full_system_cleanup()` function

**Before:**
```python
logger.mgpu_mm_log(f"[ManagerMatch] Requesting cleanup (reason={reason}) | pre_models={pre_models}, cpu_used_gib={pre_cpu/(1024**3):.2f}")
```

**After:**
```python
logger.info(f"[ManagerMatch] Requesting cleanup (reason={reason}) | pre_models={pre_models}, cpu_used_gib={pre_cpu/(1024**3):.2f}")
```

**Context:**
This logs when the system requests a cleanup operation. It includes:
- Reason for cleanup (e.g., "manual", "cpu_threshold_exceeded")
- Number of currently loaded models
- CPU memory usage in GiB

**Function Context:**
```python
def force_full_system_cleanup(reason="manual", force=True):
    """Mirror ComfyUI-Manager 'Free model and node cache' by setting unload_models=True and free_memory=True flags."""
    vm = psutil.virtual_memory()
    pre_cpu = vm.used
    pre_models = len(mm.current_loaded_models)

    multigpu_memory_log("full_cleanup", f"start:{reason}")
    logger.info(f"[ManagerMatch] Requesting cleanup (reason={reason}) | pre_models={pre_models}, cpu_used_gib={pre_cpu/(1024**3):.2f}")  # ← Fixed here
    # ... rest of function
```

#### Change 5: Flag Setting Logging (Lines 178-180)

**Location**: `force_full_system_cleanup()` function, within conditional blocks

**Before:**
```python
logger.mgpu_mm_log("[ManagerMatch] Flags set: unload_models=True, free_memory=True")
# ...
logger.mgpu_mm_log("[ManagerMatch] Skipped - execution active and force=False")
```

**After:**
```python
logger.info("[ManagerMatch] Flags set: unload_models=True, free_memory=True")
# ...
logger.info("[ManagerMatch] Skipped - execution active and force=False")
```

**Context:**
These log messages indicate:
- Whether cleanup flags (`unload_models`, `free_memory`) were successfully set
- Or if the cleanup was skipped due to active execution (when `force=False`)

**Function Context:**
```python
if server.PromptServer.instance is not None:
    pq = server.PromptServer.instance.prompt_queue
    if (not pq.currently_running) or force:
        pq.set_flag("unload_models", True)
        pq.set_flag("free_memory", True)
        logger.info("[ManagerMatch] Flags set: unload_models=True, free_memory=True")  # ← Fixed here
    else:
        logger.info("[ManagerMatch] Skipped - execution active and force=False")  # ← Fixed here
```

#### Change 6: Cleanup Summary Logging (Line 189)

**Location**: `force_full_system_cleanup()` function, final summary

**Before:**
```python
logger.mgpu_mm_log(summary)
```

**After:**
```python
logger.info(summary)
```

**Context:**
This logs the final summary of the cleanup operation, including:
- Model count changes (before → after)
- CPU memory delta in MB

**Function Context:**
```python
vm = psutil.virtual_memory()
post_cpu = vm.used
post_models = len(mm.current_loaded_models)
delta_cpu_mb = (post_cpu - pre_cpu) / (1024**2)

multigpu_memory_log("full_cleanup", f"requested:{reason}")
summary = f"[ManagerMatch] Cleanup requested (reason={reason}) | models {pre_models}->{post_models}, cpu_delta_mb={delta_cpu_mb:.2f}"
logger.info(summary)  # ← Fixed here
return summary
```

---

### File 2: `device_utils.py`

This file handles device enumeration and multi-device cache clearing operations.

#### Change: All Memory Cache Logging (Multiple Locations)

**Before:**
```python
logger.mgpu_mm_log("soft_empty_cache_multigpu: starting GC and multi-device cache clear")
logger.mgpu_mm_log(f"soft_empty_cache_multigpu: devices to clear = {all_devices}")
logger.mgpu_mm_log(f"Clearing CUDA cache on {device_str} (idx={device_idx})")
logger.mgpu_mm_log(f"Cleared CUDA cache (and IPC if available) on {device_str}")
logger.mgpu_mm_log("Clearing MPS cache")
logger.mgpu_mm_log("Cleared MPS cache")
logger.mgpu_mm_log(f"Clearing XPU cache on {device_str}")
logger.mgpu_mm_log(f"Cleared XPU cache on {device_str}")
logger.mgpu_mm_log(f"Clearing NPU cache on {device_str}")
logger.mgpu_mm_log(f"Cleared NPU cache on {device_str}")
logger.mgpu_mm_log(f"Clearing MLU cache on {device_str}")
logger.mgpu_mm_log(f"Cleared MLU cache on {device_str}")
logger.mgpu_mm_log(f"Clearing CoreX cache on {device_str}")
logger.mgpu_mm_log(f"Cleared CoreX cache on {device_str}")
logger.mgpu_mm_log("DisTorch2 active: clearing allocator caches on all devices (VRAM)")
logger.mgpu_mm_log("Force flag active: triggering executor cache reset (CPU)")
```

**After:**
```python
logger.info("soft_empty_cache_multigpu: starting GC and multi-device cache clear")
logger.info(f"soft_empty_cache_multigpu: devices to clear = {all_devices}")
logger.info(f"Clearing CUDA cache on {device_str} (idx={device_idx})")
logger.info(f"Cleared CUDA cache (and IPC if available) on {device_str}")
logger.info("Clearing MPS cache")
logger.info("Cleared MPS cache")
logger.info(f"Clearing XPU cache on {device_str}")
logger.info(f"Cleared XPU cache on {device_str}")
logger.info(f"Clearing NPU cache on {device_str}")
logger.info(f"Cleared NPU cache on {device_str}")
logger.info(f"Clearing MLU cache on {device_str}")
logger.info(f"Cleared MLU cache on {device_str}")
logger.info(f"Clearing CoreX cache on {device_str}")
logger.info(f"Cleared CoreX cache on {device_str}")
logger.info("DisTorch2 active: clearing allocator caches on all devices (VRAM)")
logger.info("Force flag active: triggering executor cache reset (CPU)")
```

**Context:**
These log messages track the multi-GPU cache clearing process across different device types:

1. **Initialization**: Logs when cache clearing starts and which devices will be cleared
2. **CUDA Devices**: Logs before and after clearing CUDA cache (NVIDIA/AMD GPUs)
3. **MPS Device**: Logs before and after clearing MPS cache (Apple Metal)
4. **XPU Devices**: Logs before and after clearing XPU cache (Intel GPUs)
5. **NPU Devices**: Logs before and after clearing NPU cache (Huawei Ascend NPUs)
6. **MLU Devices**: Logs before and after clearing MLU cache (Cambricon MLUs)
7. **CoreX Devices**: Logs before and after clearing CoreX cache
8. **DisTorch2**: Logs when DisTorch2 is active and clearing allocator caches
9. **Force Flag**: Logs when force flag triggers executor cache reset

**Function Flow:**
```python
def soft_empty_cache_multigpu():
    """Clear allocator caches across all devices using context managers to preserve calling thread device context."""
    from .model_management_mgpu import multigpu_memory_log
    
    logger.info("soft_empty_cache_multigpu: starting GC and multi-device cache clear")  # ← Fixed
    
    gc.collect()
    
    all_devices = get_device_list()
    logger.info(f"soft_empty_cache_multigpu: devices to clear = {all_devices}")  # ← Fixed
    
    for device_str in all_devices:
        if device_str.startswith("cuda:"):
            # ... CUDA cache clearing ...
            logger.info(f"Clearing CUDA cache on {device_str} (idx={device_idx})")  # ← Fixed
            # ... clear cache ...
            logger.info(f"Cleared CUDA cache (and IPC if available) on {device_str}")  # ← Fixed
        elif device_str == "mps":
            # ... MPS cache clearing ...
            logger.info("Clearing MPS cache")  # ← Fixed
            # ... clear cache ...
            logger.info("Cleared MPS cache")  # ← Fixed
        # ... similar for other device types ...
```

---

### File 3: `wrappers.py`

This file contains wrapper classes for model loading with device management and ejection capabilities.

#### Change: Model Ejection and IS_CHANGED Logging (Multiple Locations)

**Before:**
```python
logger.mgpu_mm_log(f"IS_CHANGED first call: {current_hash[:8]}")
logger.mgpu_mm_log(f"IS_CHANGED CHANGED: {current_hash[:8]} ← settings changed")
logger.mgpu_mm_log(f"[EJECT_MODELS_SETUP] eject_models=True - marking all loaded models for eviction, device target: {device_value}")
logger.mgpu_mm_log(f"[EJECT_MARKED] Model {i}: {model_name} (id=0x{id(lm):x}) → marked for eviction")
logger.mgpu_mm_log(f"[EJECT_MARKED] Model {i}: {model_name} (direct patcher) → marked for eviction")
logger.mgpu_mm_log(f"[EJECT_MODELS_SETUP_COMPLETE] Marked {ejection_count} models for Comfy Core eviction during load_models_gpu")
logger.mgpu_mm_log(f"[EJECT_MODELS_SETUP] eject_models=False - loading without eviction")
```

**After:**
```python
logger.info(f"IS_CHANGED first call: {current_hash[:8]}")
logger.info(f"IS_CHANGED CHANGED: {current_hash[:8]} ← settings changed")
logger.info(f"[EJECT_MODELS_SETUP] eject_models=True - marking all loaded models for eviction, device target: {device_value}")
logger.info(f"[EJECT_MARKED] Model {i}: {model_name} (id=0x{id(lm):x}) → marked for eviction")
logger.info(f"[EJECT_MARKED] Model {i}: {model_name} (direct patcher) → marked for eviction")
logger.info(f"[EJECT_MODELS_SETUP_COMPLETE] Marked {ejection_count} models for Comfy Core eviction during load_models_gpu")
logger.info(f"[EJECT_MODELS_SETUP] eject_models=False - loading without eviction")
```

**Context:**
These log messages track:

1. **IS_CHANGED Detection**: 
   - First call detection (when hash is calculated for the first time)
   - Change detection (when settings have changed, triggering model reload)

2. **Model Ejection Setup**:
   - When `eject_models=True`, logs the setup process
   - Marks each model for eviction with model name and memory address
   - Logs completion with total count of marked models
   - Logs when ejection is disabled

**Function Context:**
```python
def IS_CHANGED(cls, **kwargs):
    # ... calculate hash from settings ...
    if not hasattr(cls, '_last_hash'):
        cls._last_hash = current_hash
        logger.info(f"IS_CHANGED first call: {current_hash[:8]}")  # ← Fixed
    elif cls._last_hash != current_hash:
        cls._last_hash = current_hash
        logger.info(f"IS_CHANGED CHANGED: {current_hash[:8]} ← settings changed")  # ← Fixed
    return current_hash

def override(self, *args, virtual_vram_gb=4.0, donor_device="cpu",
             expert_mode_allocations="", eject_models=eject_models_default, **kwargs):
    # ...
    if eject_models:
        logger.info(f"[EJECT_MODELS_SETUP] eject_models=True - marking all loaded models for eviction, device target: {device_value}")  # ← Fixed
        ejection_count = 0
        for i, lm in enumerate(mm.current_loaded_models):
            # ... mark models for eviction ...
            if hasattr(lm.model, 'model') and lm.model.model is not None:
                lm.model.model._mgpu_unload_distorch_model = True
                logger.info(f"[EJECT_MARKED] Model {i}: {model_name} (id=0x{id(lm):x}) → marked for eviction")  # ← Fixed
            elif lm.model is not None:
                lm.model._mgpu_unload_distorch_model = True
                logger.info(f"[EJECT_MARKED] Model {i}: {model_name} (direct patcher) → marked for eviction")  # ← Fixed
        logger.info(f"[EJECT_MODELS_SETUP_COMPLETE] Marked {ejection_count} models for Comfy Core eviction during load_models_gpu")  # ← Fixed
    else:
        logger.info(f"[EJECT_MODELS_SETUP] eject_models=False - loading without eviction")  # ← Fixed
```

---

## Technical Explanation

### Logger Configuration

The logger is configured as:
```python
import logging
logger = logging.getLogger("SDXL")
```

This creates a logger instance named "SDXL" that uses the standard Python logging infrastructure. All log messages will be handled according to ComfyUI's logging configuration.

### Why `mgpu_mm_log` Doesn't Exist

Python's standard `logging.Logger` class provides the following methods:
- `logger.debug()` - Detailed diagnostic information
- `logger.info()` - General informational messages
- `logger.warning()` - Warning messages
- `logger.error()` - Error messages
- `logger.critical()` - Critical error messages

The `mgpu_mm_log()` method was likely intended as a custom logging method for multi-GPU memory management, but it was **never implemented**. The code attempted to use this non-existent method, causing the `AttributeError`.

### Why `logger.info()` is the Correct Replacement

1. **Standard Method**: It's a standard method available on all `Logger` instances
2. **Appropriate Level**: The logged messages are informational in nature (memory snapshots, cleanup operations, cache clearing)
3. **Consistent Behavior**: All messages maintain the same logging level and visibility
4. **No Breaking Changes**: The log output format and content remain identical, only the method name changes
5. **Compatibility**: Works with all Python logging configurations and handlers

### Memory Management Flow

The complete flow that triggers these logging calls:

1. **Trigger**: `soft_empty_cache()` is called during prompt execution
2. **Patched Function**: `soft_empty_cache_distorch2_patched()` intercepts the call
3. **CPU Check**: `check_cpu_memory_threshold()` monitors CPU memory usage
4. **Memory Logging**: `multigpu_memory_log()` records memory snapshots
5. **Cache Clearing**: Device-specific cache clearing operations are logged
6. **Cleanup**: Model cleanup and memory freeing operations are logged

---

## Impact

### Before Fix

- **Error**: `AttributeError` crashes the prompt execution thread
- **User Experience**: ComfyUI becomes unresponsive or crashes when executing prompts
- **Functionality**: Memory management and monitoring features fail
- **Workflow**: Users cannot complete workflows that trigger memory management

### After Fix

- **No Errors**: All logging calls use standard `logger.info()` method
- **Stable Operation**: Prompt execution proceeds without crashes
- **Full Functionality**: Memory management, monitoring, and cleanup operations work correctly
- **Logging Preserved**: All log messages are still output, maintaining debugging and monitoring capabilities
- **User Experience**: Seamless workflow execution without interruptions

---

## Verification

### How to Verify the Fix

1. **Check for Remaining Instances**:
   ```bash
   grep -r "mgpu_mm_log" ComfyUI-nunchaku-unofficial-loader/
   ```
   Should return no results.

2. **Test Prompt Execution**:
   - Execute a prompt in ComfyUI
   - Monitor the console for any `AttributeError` related to `mgpu_mm_log`
   - Verify that memory management operations complete successfully

3. **Check Log Output**:
   - Verify that memory-related log messages appear in the console
   - Confirm that log messages contain the expected information (memory snapshots, cleanup operations, etc.)

4. **Test Memory Management**:
   - Trigger memory-intensive operations
   - Verify that CPU memory threshold checking works
   - Confirm that multi-device cache clearing completes without errors

---

## Related Code

### Logger Initialization

In all affected files, the logger is initialized as:
```python
import logging
logger = logging.getLogger("SDXL")
```

### Memory Logging Function

The `multigpu_memory_log()` function in `model_management_mgpu.py` is responsible for:
- Capturing memory snapshots from CPU and all GPU devices
- Formatting timestamped log entries
- Storing memory snapshots in series for analysis
- Providing memory usage summaries

### Memory Management Functions

Key functions that use the logging:

1. **`check_cpu_memory_threshold()`**: Monitors CPU memory and triggers cleanup if threshold exceeded
2. **`multigpu_memory_log()`**: Records memory snapshots with timestamps
3. **`soft_empty_cache_multigpu()`**: Clears caches across all devices
4. **`force_full_system_cleanup()`**: Triggers full system cleanup with model unloading
5. **`trigger_executor_cache_reset()`**: Resets PromptExecutor cache

---

## Git Commit Information

**Commit**: `ac3d0f6`  
**Message**: "Fix AttributeError: Replace logger.mgpu_mm_log() with logger.info()"  
**Files Changed**: 3 files (new files)  
**Lines Added**: 1066 insertions  
**Repository**: `ComfyUI-nunchaku-unofficial-loader`

---

## Summary

| Aspect | Details |
|--------|---------|
| **Problem** | `AttributeError: 'Logger' object has no attribute 'mgpu_mm_log'` |
| **Root Cause** | Attempted to call non-existent custom logging method |
| **Solution** | Replace all `logger.mgpu_mm_log()` calls with `logger.info()` |
| **Files Modified** | 3 files (`model_management_mgpu.py`, `device_utils.py`, `wrappers.py`) |
| **Total Changes** | ~30+ instances replaced |
| **Impact** | Fixes prompt execution crashes, maintains all logging functionality |
| **Breaking Changes** | None - log output format and content remain identical |
| **Testing** | Verified: No remaining `mgpu_mm_log` calls, prompt execution works |

This fix ensures that all memory management logging operations use standard Python logging methods, eliminating the `AttributeError` and restoring full functionality to the multi-GPU memory management system. The fix is backward compatible and maintains all existing logging behavior while using standard logging infrastructure.
