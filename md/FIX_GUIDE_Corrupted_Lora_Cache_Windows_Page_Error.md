# Repair Guide: Windows Fatal "Page Error" Crash When Loading a Damaged Precompiled LoRA Cache

**Project:** ComfyUI-QwenImageLoraLoader
**File modified:** `nunchaku_code/lora_cache.py`
**Commit:** `cd3ff69` — `fix(cache): read precompiled cache as bytes, not mmap (Windows page error)`
**Date:** 2026-08-26

---

## 1. What the Error Was

While running a QwenImage generation workflow, ComfyUI crashed hard — the whole
process died, not just the node. The console ended with:

```
[INFO] Requested to load WanVAE
[MultiGPU Issue21] Using non-recursive ModelPatcherDynamic._load_list guard
[INFO] Composing 1 LoRAs (Direct Fix V6)...
[INFO] ⚠️  AWQ Modulation Layer LoRA Injection ENABLED (via override).
Windows fatal exception: page error

Stack (most recent call first):
  File "D:\USERFILES\ComfyUI\python_embeded\Lib\site-packages\safetensors\torch.py", line 359 in load_file
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-QwenImageLoraLoader\nunchaku_code\lora_cache.py", line 272 in load_precompiled
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-QwenImageLoraLoader\nunchaku_code\lora_qwen.py", line 1411 in compose_loras_v2
```

The final line is the key signature:

```
Windows fatal exception: page error
```

This is **not a Python exception** — it is a native Windows fatal error
(`STATUS_IN_PAGE_ERROR`, process exit code `-1073741818` / `0xC0000006`).
It appeared inside `safetensors.torch.load_file` while `load_precompiled()`
was reading a precompiled LoRA cache. Because it is a native crash, no
`try/except` anywhere in Python could catch it, and the entire ComfyUI process
terminated instantly.

---

## 2. What Caused It

### 2.1 Diagnostic evidence

Two of the precompiled cache files under
`ComfyUI/models/SVDQLora/` were damaged on disk:

| Cache file | Read test result |
|---|---|
| `Qwen-Image-Lightning-8steps-V2.0_d25379dc_precompiled.safetensors` (2.11 GB) | ❌ Crash / `OSError: [Errno 22] Invalid argument` |
| `Qwen-NSFW-Beta5_d25379dc_precompiled.safetensors` (0.68 GB) | ❌ Crash |
| All other 4 cache files (Asian, InStyle, Qwen-Edit, face_swap) | ✅ Loaded fine |
| Both source LoRA files in `D:\USERFILES\StableDiffusion\models\Lora\qi\quality\` | ✅ Loaded fine (2160 / 2520 keys) |

Key observations:

- The **source LoRA files were perfectly healthy**. Only the *generated
  precompiled caches* were corrupt.
- The damaged files could still report a **valid, readable safetensors header**
  (metadata), but their **tensor data region was physically unreadable**
  (disk-level damage — bad sector or similar).
- At the OS level, even a plain sequential read of the data region fails with
  `Errno 22`.

### 2.2 Why it crashed instead of falling back

The cache pipeline is designed to recover: `load_precompiled()` already had a
`try/except` and was documented to "return an empty dict on any error so the
caller can fall back to a full re-fuse gracefully." **The problem is the read
mechanism:**

```python
# BEFORE (the broken path)
from safetensors.torch import load_file as st_load_file
flat = st_load_file(str(cache_path), device="cpu")
```

`safetensors.torch.load_file()` is **mmap-based**. It maps the file into the
process address space and only touches the pages as tensors are accessed.
For a damaged file:

1. The header pages are intact → metadata parse succeeds → the loader
   believes the cache is valid.
2. When a tensor's data region is paged in, the OS cannot read it from disk →
   the memory manager raises `STATUS_IN_PAGE_ERROR` → **Windows kills the
   process immediately**.

A `STATUS_IN_PAGE_ERROR` is delivered as a native fault, not as a Python
exception, so the `except Exception` block never ran. Result: hard crash,
every single time, as long as the damaged cache file was detected as "valid".

### 2.3 Why the damaged cache was treated as valid

`is_cache_valid()` only checked **file existence + recorded source mtime**:

```python
if not cache_path.is_file():
    return False
# ... reads meta.json, compares source_mtime ...
return True
```

It performed **no integrity check of the cache file contents**, so a file with
a readable header but an unreadable data region was considered "up to date and
usable."

---

## 3. The Modified Code (Full Text)

### 3.1 The changed block — before vs. after

**Before** (`nunchaku_code/lora_cache.py`, inside `load_precompiled()`):

```python
    try:
        from safetensors.torch import load_file as st_load_file
    except ImportError as exc:
        logger.error(f"[CACHE] Cannot load precompiled cache: safetensors not available. {exc}")
        return {}

    try:
        flat: Dict[str, torch.Tensor] = st_load_file(str(cache_path), device="cpu")
    except Exception as exc:
        logger.error(f"[CACHE] Failed to load {cache_path}: {exc}")
        return {}
```

**After** (the fix, commit `cd3ff69`):

```python
    try:
        from safetensors.torch import load as st_load_bytes
    except ImportError as exc:
        logger.error(f"[CACHE] Cannot load precompiled cache: safetensors not available. {exc}")
        return {}

    try:
        # Read as plain bytes instead of ``load_file`` (mmap). A damaged cache
        # file can still expose a readable header but fail during the mmap'd
        # tensor read, which crashes the whole process with a Windows fatal
        # "page error" that Python cannot catch. A plain read surfaces the same
        # damage as an ordinary OSError, so the existing except below falls
        # back to a full re-fuse gracefully.
        with open(cache_path, "rb") as fh:
            data = fh.read()
        flat: Dict[str, torch.Tensor] = st_load_bytes(data)
    except Exception as exc:
        logger.error(f"[CACHE] Failed to load {cache_path}: {exc}")
        return {}
```

### 3.2 The full `load_precompiled()` function after the fix

```python
def load_precompiled(
    cache_path: Path,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """
    Load a precompiled cache file and reconstruct the ``processed_groups`` dict.

    Returns:
        ``{ module_key: (A, B, alpha_or_None) }`` ? identical structure to
        what ``compose_loras_v2`` produces after the classify+fuse stage.
        Returns an empty dict on any error so the caller can fall back to
        a full re-fuse gracefully.
    """
    try:
        from safetensors.torch import load as st_load_bytes
    except ImportError as exc:
        logger.error(f"[CACHE] Cannot load precompiled cache: safetensors not available. {exc}")
        return {}

    try:
        # Read as plain bytes instead of ``load_file`` (mmap). A damaged cache
        # file can still expose a readable header but fail during the mmap'd
        # tensor read, which crashes the whole process with a Windows fatal
        # "page error" that Python cannot catch. A plain read surfaces the same
        # damage as an ordinary OSError, so the existing except below falls
        # back to a full re-fuse gracefully.
        with open(cache_path, "rb") as fh:
            data = fh.read()
        flat: Dict[str, torch.Tensor] = st_load_bytes(data)
    except Exception as exc:
        logger.error(f"[CACHE] Failed to load {cache_path}: {exc}")
        return {}

    # Reconstruct { module_key: { "A": T, "B": T, "alpha": T|None } }
    raw: Dict[str, Dict[str, torch.Tensor]] = {}
    for flat_key, tensor in flat.items():
        # flat_key format:  "<module_key>__<role>"
        # module_key itself may contain dots but never "__"
        sep_idx = flat_key.rfind(_SEP)
        if sep_idx == -1:
            logger.warning(f"[CACHE] Unrecognised flat key format: '{flat_key}' ? skipping.")
            continue
        module_key = flat_key[:sep_idx]
        role = flat_key[sep_idx + len(_SEP):]
        raw.setdefault(module_key, {})[role] = tensor

    processed_groups: Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = {}
    for module_key, parts in raw.items():
        A = parts.get(_ROLE_A)
        B = parts.get(_ROLE_B)
        alpha_tensor = parts.get(_ROLE_ALPHA)  # 1-D float32 scalar tensor or None

        if A is None or B is None:
            logger.warning(f"[CACHE] Incomplete entry for '{module_key}' (missing A or B) ? skipping.")
            continue

        # Convert alpha back to the same form compose_loras_v2 expects:
        # a torch.Tensor scalar (not a Python float).
        alpha: Optional[torch.Tensor] = None
        if alpha_tensor is not None:
            alpha = alpha_tensor  # keep as tensor; compose_loras_v2 calls .item() on it

        processed_groups[module_key] = (A, B, alpha)

    logger.info(f"[CACHE LOAD] Loaded {len(processed_groups)} module entries ← {cache_path.name}")
    return processed_groups
```

---

## 4. What the Code Does / Why It Fixes the Crash

### 4.1 `from safetensors.torch import load as st_load_bytes`

We switch from `load_file` (mmap-backed) to `load` (bytes-backed):

- `load_file(path)` memory-maps the file and lazily faults in pages during
  tensor access → a damaged data region triggers an *uncatchable native page
  error*.
- `load(bytes)` parses a Python `bytes` object already held in memory; there is
  no mmap involved at all.

### 4.2 `with open(cache_path, "rb") as fh: data = fh.read()`

The whole cache file is read with a **plain sequential read** instead of mmap.
This converts the failure mode from "native page fault" into an **ordinary
Python `OSError`**:

- On a healthy file: reads everything, no difference in behavior.
- On a damaged file: the OS read hits the dead region and raises
  `OSError: [Errno 22] Invalid argument` — a *normal Python exception* that the
  surrounding `try/except Exception` can catch.

Memory note: a 2 GB cache transiently occupies ~2 GB of host RAM as `bytes`
before parsing, which is acceptable on a typical workstation and far cheaper
than a hard process crash.

### 4.3 The existing `except Exception` now actually works

```python
    except Exception as exc:
        logger.error(f"[CACHE] Failed to load {cache_path}: {exc}")
        return {}
```

Because the failure now arrives as a Python exception, this block runs for the
first time on damaged caches. It logs the error and returns an **empty dict** —
exactly the documented contract of `load_precompiled()`. The caller
(`compose_loras_v2` in `lora_qwen.py`) then falls back to a **full re-fuse** of
the source LoRA. The freshly fused result is re-saved through
`save_precompiled()`, overwriting the damaged cache with a healthy one
(`os.replace` of a `.tmp` file). The system self-heals: no manual deletion
needed, no repeated crash.

### 4.4 Why `is_cache_valid()` was left unchanged

`is_cache_valid()` (file existence + mtime sidecar) is a cheap fast-path check.
Adding a full data-integrity scan there would make every cache validation read
the entire file, defeating the purpose of the cache. With the fix in
`load_precompiled()`, a corrupt file is caught *at load time* — the only moment
its contents are actually needed — so the fast path can stay cheap.

---

## 5. Verification (measured)

Run with the project's bundled Python:

| Test case | Before fix | After fix |
|---|---|---|
| Damaged cache (`...Lightning-8steps...safetensors`) | **Process crash** (`exit=-1073741818`) | ✅ Returns `{}` (empty dict, no crash) |
| Healthy cache (`InStyle-0.5...safetensors`) | 600 entries loaded | ✅ 600 entries loaded (unchanged) |
| Source LoRA files (2160 / 2520 keys) | fine | ✅ fine (untouched) |

Additional evidence collected during diagnosis:

```
BROKEN -> result: dict len=0 (NO CRASH)
OK cache -> len=600 entries loaded
```

---

## 6. Operational Notes

- The two damaged caches on the affected machine were renamed to
  `*.safetensors.broken` (and matching `.meta.json.broken`) so the loader
  treats them as invalid and regenerates them from the healthy source LoRAs on
  next startup.
- No source LoRA file was modified or re-downloaded — the corruption was
  limited to the *generated* cache artifacts.
- If a crash of this kind is ever seen again (any safetensors mmap load), the
  same read-as-bytes pattern applies.