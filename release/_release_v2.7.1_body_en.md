<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.7.1.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

This document covers one change: no code that runs during ComfyUI's prestartup phase
imports `torch` any more, so `main.py` no longer logs

```
WARNING: Potential Error in code: Torch already imported, torch should never be imported before this point.
```

Two node packages were involved: this repository (`ComfyUI-QwenImageLoraLoader`) and
`ComfyUI-HSWQ-Loader-and-Tools`. Both are covered below.

## 1. What the problem was

### 1.1 The symptom

Every ComfyUI start printed the warning above a few seconds after the "Prestartup times for
custom nodes" block. Nothing failed: the server started, models loaded, images rendered. The
line is a diagnostic from ComfyUI itself, not an error from a node.

The cost was diagnostic noise on every start, and - worse - it made the startup log useless as
a signal: a real early-import problem would have looked exactly the same.

### 1.2 What had been done about it before this change

Commit `9e30661` (2026-07-21, "fix: suppress cosmetic Torch already imported warning in
prestartup") added a `logging.Filter` to the root logger from `prestartup_script.py`:

- the filter matched the substring `Torch already imported`;
- the first matching record was dropped (`return False`) and every later record passed;
- an environment variable `QWENIMAGE_SUPPRESS_TORCH_WARNING` could disable the filter.

Measured effect: with the filter installed, the warning disappeared from the log while `torch`
was still imported during prestartup. The mechanism hid the symptom and left the cause intact.
It also hid the warning for *other* nodes, because the filter sat on the root logger.

### 1.3 What was actually observed in this installation

| State | `Torch already imported` in the startup log |
| --- | --- |
| v2.7.0 (filter installed, both nodes importing torch) | 0 lines - suppressed |
| filter removed, this package fixed only | 1 line |
| filter removed, this package and HSWQ fixed | 0 lines |

The middle row is the important one: after removing the filter and fixing only this package the
warning was still printed, because a second node imported `torch` earlier in the prestartup
order. Fixing one node was not the fix.

## 2. The essential cause

### 2.1 Where ComfyUI checks

`main.py`, in the order it executes:

| Line | Statement |
| --- | --- |
| 254 | `execute_prestartup_script()` |
| 262 | `if 'torch' in sys.modules:` |
| 263 | `logging.warning("WARNING: Potential Error in code: Torch already imported, ...")` |
| 266 | `import comfy.utils` |
| 272 | `import nodes` (custom nodes load) |
| 296 | `comfy_aimdo.control.init_devices(...)` |

The check is a single membership test on `sys.modules`. Anything that imports `torch` before
line 262 - including a `comfy.*` module, because `comfy.memory_management` imports `torch` on
its first line - trips it. The check runs after every `prestartup_script.py`, so one node is
enough.

### 2.2 Trigger one: this package's rotary compat shim

`patches/nunchaku_patch.py` had `import torch` at module level (line 10), and the module is
loaded by `prestartup_script.py`. Loading the module alone put `torch` into `sys.modules`:

```
before= False  after= True
```

The rotary compat function additionally did `import comfy.ldm.qwen_image.model` at prestartup
time to decide whether to alias `apply_rope1`. That import pulls `torch` as well. Measured
after calling it: `torch=True comfy=True`.

The shim exists for a real reason: ComfyUI removed `comfy.ldm.qwen_image.model.apply_rotary_emb`
in `6e1ef2311b` and ComfyUI-nunchaku still imports it, so the symbol must exist before nunchaku's
module is imported. That is why the work was in prestartup in the first place.

### 2.3 Trigger two: HSWQ fast-disk support

`ComfyUI-HSWQ-Loader-and-Tools/prestartup_script.py` called `_install_fastdisk_support()` at
module level. That function loads `patches/comfy_aimdo_fastdisk_guard.py` and applies it, and
the application imports `comfy.memory_management` (`patches/comfy_aimdo_fastdisk_guard.py`,
`_patch_mem_slice`) plus the `comfy_aimdo` modules. `comfy/memory_management.py` line 4 is
`import torch`.

Stack at the first `torch` import (measured):

```
torch
  <frozen importlib._bootstrap>: _load_unlocked / exec_module
  comfy/memory_management.py:4
  ComfyUI-HSWQ-Loader-and-Tools/prestartup_script.py: (the __import__ hook line)
```

Other prestartup scripts were measured and do not import `torch`:

| Node | After its prestartup |
| --- | --- |
| `rgthree-comfy` | `torch=False comfy=True` |
| `comfyui-easy-use` | `torch=False comfy=True` |
| `ComfyUI-Legacy-Tools` | `torch=False comfy=False` |
| `ComfyUI-Manager` | `torch=False comfy=True` |

### 2.4 Why the calls could not simply be deleted

Both prestartup jobs exist for ordering reasons, and the order they protect is not the order
ComfyUI's warning cares about:

- **Rotary alias**: nunchaku reads `apply_rotary_emb` when *its* module is imported. The alias
  must already be in place, so it cannot be deferred past the custom-node load. It only has to
  be applied *before that read*, not before ComfyUI's check.
- **Fast-disk guard** (`patches/comfy_aimdo_fastdisk_guard.py` docstring): comfy-aimdo is
  single-threaded by design - `g_devctx` is thread-local, the native reader has one completion
  slot that the previous read must retire, and stream resolution uses the calling thread's
  current stream. ComfyUI v0.38 async offload / parallel load / prefetch drive those calls from
  several threads, so the guard runs every native call inline on the caller thread, serialises
  with a process-wide `RLock`, syncs before page-mapping and after device-destination copies
  (never during CUDA graph capture), and converts a failed read into a recoverable `False` so
  ComfyUI's mmap fallback materialises the data instead of aborting. Therefore the guard must be
  installed before any aimdo call and before `init_devices`.

Deleting either call would remove the protection. The fix had to move the *arming point* into
the window between `main.py:262` (check) and `main.py:296` (`init_devices`), without importing
`torch` before the check.

### 2.5 The mechanism used for both

An import hook (`sys.meta_path` finder, or the existing `builtins.__import__` hook) that does no
import itself and applies the work at the moment the target module is loaded:

- this package: intercept `comfy.ldm.qwen_image.model`, wrap its loader, install the alias in
  `exec_module` right after the module executes;
- HSWQ: keep the `builtins.__import__` hook, and install the fast-disk guard on the first import
  that happens after `comfy.memory_management` is fully imported (detected by the presence of
  `read_tensor_file_slice_into`).

Both hooks fire inside `import comfy.utils` at `main.py:266` - after the check, before
`init_devices`.

## 3. Files created and modified

| Repository | File | Change | Size |
| --- | --- | --- | --- |
| `ComfyUI-QwenImageLoraLoader` | `patches/nunchaku_patch.py` | module-level `import torch` removed; `from __future__ import annotations` added; local `import torch` in two functions; rotary compat reimplemented as an import hook; warning filter and `suppress_torch_preimport_warning()` deleted | 25,346 B, 625 lines |
| `ComfyUI-QwenImageLoraLoader` | `prestartup_script.py` | 13-line suppression block deleted | 3,464 B, 83 lines |
| `ComfyUI-HSWQ-Loader-and-Tools` | `prestartup_script.py` | eager `_install_fastdisk_support()` replaced by `_maybe_install_fastdisk()` driven from the existing `__import__` hook; installer now returns the applied result | 10,741 B, 288 lines |
| `ComfyUI-QwenImageLoraLoader` | `md/TORCH_PREIMPORT_ELIMINATION.md` | this document | new |

Commit for this repository: `96c9a20` "fix: keep torch out of prestartup and remove the warning
filter" (2 files, +108 / -105). The HSWQ change is applied to the live tree; no commit was made.

## 4. Full code

### 4.1 `patches/nunchaku_patch.py` - module header (before / after)

Before:

```python
import inspect
import logging
import os
import re
import sys
import types
from typing import Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)
```

After:

```python
from __future__ import annotations

import inspect
import logging
import os
import re
import sys
import types
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)
```

### 4.2 `patches/nunchaku_patch.py` - removed suppression block

Deleted in full (it was added by `9e30661`):

```python
_torch_preimport_warning_suppressed: bool = False
_TORCH_PREIMPORT_WARNING_MARKER = "Torch already imported"


def _torch_warning_suppression_disabled_by_env() -> bool:
    value = os.environ.get("QWENIMAGE_SUPPRESS_TORCH_WARNING", "").strip().lower()
    return value in ("0", "false", "no", "off", "disable", "disabled")


class _TorchPreimportWarningFilter(logging.Filter):
    """Drop ComfyUI's cosmetic 'Torch already imported' warning."""

    def __init__(self) -> None:
        super().__init__()
        self._suppressed = False

    def filter(self, record: logging.LogRecord) -> bool:
        if self._suppressed:
            return True
        try:
            message = record.getMessage()
        except Exception:
            return True
        if _TORCH_PREIMPORT_WARNING_MARKER in message:
            self._suppressed = True
            return False
        return True


def suppress_torch_preimport_warning() -> bool:
    global _torch_preimport_warning_suppressed
    if _torch_preimport_warning_suppressed:
        return True
    if _torch_warning_suppression_disabled_by_env():
        logger.info(
            "Torch pre-import warning suppression skipped "
            "(QWENIMAGE_SUPPRESS_TORCH_WARNING is disabled)"
        )
        return False
    try:
        logging.getLogger().addFilter(_TorchPreimportWarningFilter())
        _torch_preimport_warning_suppressed = True
        return True
    except Exception as exc:
        logger.debug("Failed to install torch pre-import warning filter: %s", exc)
        return False
```

### 4.3 `patches/nunchaku_patch.py` - rotary compat as an import hook (new code, complete)

```python
_ROTARY_TARGET_MODULE = "comfy.ldm.qwen_image.model"
_ROTARY_PARENT_MODULE = "comfy.ldm.qwen_image"
_rotary_hook_installed: bool = False


def _install_rotary_alias(qwen_image_model) -> bool:
    """Alias apply_rope1 onto an already imported comfy.ldm.qwen_image.model module."""
    global _qwen_apply_rotary_emb_compat_applied
    if qwen_image_model is None:
        return False
    if _rotary_shim_installed_on(qwen_image_model):
        _qwen_apply_rotary_emb_compat_applied = True
        return True
    if _comfyui_has_native_apply_rotary_emb(qwen_image_model):
        logger.info(
            "apply_rotary_emb compat skipped: ComfyUI already exports apply_rotary_emb"
        )
        return False

    nunchaku_needs = _nunchaku_qwenimage_still_imports_apply_rotary_emb()
    if nunchaku_needs is False:
        logger.info(
            "apply_rotary_emb compat skipped: ComfyUI-nunchaku no longer imports apply_rotary_emb"
        )
        return False

    from comfy.ldm.flux.math import apply_rope1

    qwen_image_model.apply_rotary_emb = _mark_rotary_shim(apply_rope1)
    _qwen_apply_rotary_emb_compat_applied = True
    logger.info(
        "Patched comfy.ldm.qwen_image.model.apply_rotary_emb -> apply_rope1 "
        "(ComfyUI-nunchaku Qwen Image compat)"
    )
    return True


class _RotaryAliasLoader:
    """Loader wrapper that applies the alias right after the target module executes."""

    def __init__(self, loader) -> None:
        self._loader = loader

    def __getattr__(self, item):
        return getattr(self._loader, item)

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        try:
            _install_rotary_alias(module)
        except Exception as exc:
            logger.error("Failed to apply apply_rotary_emb compat patch: %s", exc)


class _RotaryCompatFinder:
    """Resolve comfy.ldm.qwen_image.model through its parent package without importing it.

    Importing comfy here would pull torch during prestartup, which is exactly the state
    ComfyUI warns about. The finder only wraps the loader, so comfy and torch enter
    sys.modules at the moment the model module is actually imported, as usual.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _ROTARY_TARGET_MODULE or fullname in sys.modules:
            return None
        parent = sys.modules.get(_ROTARY_PARENT_MODULE)
        parent_path = list(getattr(parent, "__path__", None) or [])
        if not parent_path:
            return None
        import importlib.machinery

        spec = importlib.machinery.PathFinder.find_spec(fullname, parent_path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _RotaryAliasLoader(spec.loader)
        return spec


def apply_qwen_image_apply_rotary_emb_compat() -> bool:
    """Arm the apply_rotary_emb compat shim without importing comfy or torch.

    ComfyUI >= 0.24 removed comfy.ldm.qwen_image.model.apply_rotary_emb and
    ComfyUI-nunchaku still imports it. The alias is applied by an import hook: at once
    if the module is already imported, otherwise as soon as it is imported - which is
    still before ComfyUI-nunchaku reads the symbol.
    """
    global _qwen_apply_rotary_emb_compat_applied, _rotary_hook_installed
    if _qwen_apply_rotary_emb_compat_applied:
        return True
    if _rotary_compat_disabled_by_env():
        logger.info(
            "apply_rotary_emb compat skipped (QWENIMAGE_ROTARY_COMPAT is disabled)"
        )
        return False

    loaded = sys.modules.get(_ROTARY_TARGET_MODULE)
    if loaded is not None:
        try:
            return _install_rotary_alias(loaded)
        except Exception as e:
            logger.error("Failed to apply apply_rotary_emb compat patch: %s", e)
            return False

    if _rotary_hook_installed:
        return True
    try:
        sys.meta_path.insert(0, _RotaryCompatFinder())
        _rotary_hook_installed = True
        logger.info(
            "apply_rotary_emb compat armed: alias applies when comfy.ldm.qwen_image.model "
            "is imported (torch is not imported during prestartup)"
        )
        return True
    except Exception as e:
        logger.error("Failed to arm apply_rotary_emb compat hook: %s", e)
        return False
```

### 4.4 `patches/nunchaku_patch.py` - local `import torch` (the two functions that run after load)

```python
def _torch_device_fallback() -> torch.device:
    import torch

    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
def _safe_copy_params_into(src: torch.nn.Module, dst: torch.nn.Module, non_blocking: bool = True):
    """
    Safely copy parameters and buffers from src to dst by name matching (with positional fallback),
    and correctly synchronize wtscale attribute without attribute deletion.
    """
    import torch

    with torch.no_grad():
        ...
```

All other `torch.` occurrences in the module (13 in total before the change) are type
annotations, which `from __future__ import annotations` turns into strings, or attribute access
on objects passed in at call time. No other module-level `torch` reference remains.

### 4.5 `prestartup_script.py` (this repository) - full file

```python
"""
Inject apply_rotary_emb on comfy.ldm.qwen_image.model before any custom node __init__.

ComfyUI-nunchaku loads before ComfyUI-QwenImageLoraLoader (Windows listdir order), so
__init__.py alone is too late. prestartup_script.py runs from main.execute_prestartup_script().
"""
import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

_PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches")
_NUNCHAKU_PATCH_PATH = os.path.join(_PATCH_DIR, "nunchaku_patch.py")
_DOCSTRING_PATCH_PATH = os.path.join(_PATCH_DIR, "transformers_qwen_vl_docstring_patch.py")
_IP_KWARGS_PATCH_PATH = os.path.join(_PATCH_DIR, "transformers_image_processor_kwargs_patch.py")


def _load_patch_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load patch module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the nunchaku patch module once; reuse it for the apply_rotary_emb compat shim.
_patch_module = None
try:
    _patch_module = _load_patch_module(
        "comfyui_qwenimageloraloader_nunchaku_patch_prestartup",
        _NUNCHAKU_PATCH_PATH,
    )
except Exception:
    logger.exception("ComfyUI-QwenImageLoraLoader prestartup: failed to load nunchaku patch module")

try:
    _docstring_patch_module = _load_patch_module(
        "comfyui_qwenimageloraloader_docstring_patch_prestartup",
        _DOCSTRING_PATCH_PATH,
    )
    if _docstring_patch_module.apply_transformers_causal_lm_docstring_patch():
        logger.info("ComfyUI-QwenImageLoraLoader prestartup: CausalLM ModelOutput docstring patch applied")
    else:
        logger.debug(
            "ComfyUI-QwenImageLoraLoader prestartup: CausalLM ModelOutput docstring patch not applied"
        )
except Exception:
    logger.exception("ComfyUI-QwenImageLoraLoader prestartup: CausalLM ModelOutput docstring patch failed")

# ImageProcessorKwargs docstring patch (deepseek_vl_hybrid / kimi_k25 / paddleocr_vl
# [ERROR] "but not documented" noise): wraps get_args_doc_from_source at prestartup,
# before any custom node imports those transformers modules.
try:
    _ip_kwargs_patch_module = _load_patch_module(
        "comfyui_qwenimageloraloader_image_processor_kwargs_patch_prestartup",
        _IP_KWARGS_PATCH_PATH,
    )
    if _ip_kwargs_patch_module.apply_transformers_image_processor_kwargs_patch():
        logger.info(
            "ComfyUI-QwenImageLoraLoader prestartup: ImageProcessorKwargs docstring patch applied"
        )
    else:
        logger.debug(
            "ComfyUI-QwenImageLoraLoader prestartup: ImageProcessorKwargs docstring patch not applied"
        )
except Exception:
    logger.exception(
        "ComfyUI-QwenImageLoraLoader prestartup: ImageProcessorKwargs docstring patch failed"
    )

if _patch_module is not None:
    try:
        if _patch_module.apply_qwen_image_apply_rotary_emb_compat():
            logger.info("ComfyUI-QwenImageLoraLoader prestartup: apply_rotary_emb compat applied")
        else:
            logger.debug(
                "ComfyUI-QwenImageLoraLoader prestartup: apply_rotary_emb compat not needed or already present"
            )
    except Exception:
        logger.exception("ComfyUI-QwenImageLoraLoader prestartup: apply_rotary_emb compat failed")
```

### 4.6 `ComfyUI-HSWQ-Loader-and-Tools/prestartup_script.py` - new blocks

```python
_FASTDISK_INSTALLED = False
_FASTDISK_INSTALLING = False


def _maybe_install_fastdisk() -> None:
    """Arm fast-disk support on the first comfy.memory_management import.

    Importing comfy here would pull torch during prestartup, and ComfyUI then logs
    "Torch already imported". comfy.memory_management is first imported by
    comfy.utils inside main.py, i.e. immediately after that check and long before
    init_devices, so the guard is still armed before aimdo initialises.
    """
    global _FASTDISK_INSTALLED, _FASTDISK_INSTALLING
    if _FASTDISK_INSTALLED or _FASTDISK_INSTALLING:
        return
    installer = globals().get("_install_fastdisk_support")
    if installer is None:
        return
    _FASTDISK_INSTALLING = True
    try:
        _FASTDISK_INSTALLED = bool(installer())
    except Exception as e:  # noqa: BLE001
        try:
            print(f"[HSWQ FastDisk] support install failed: {e}", flush=True)
        except Exception:
            pass
    finally:
        _FASTDISK_INSTALLING = False
```

The existing import hook gained one block at the end (everything above it is unchanged):

```python
def _import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _ORIG_IMPORT(name, globals, locals, fromlist, level)
    if not _DIT_LOCK_PATCHED and not _DIT_LOCK_PATCHING and (
        str(name) == "comfy.ops"
        or (str(name) == "comfy" and any(str(x) == "ops" for x in (fromlist or ())))
    ):
        _patch_dit_quant_lock()
    if not _CTRL_PATCHED and not _CTRL_PATCHING and (
        str(name) == "comfy.controlnet"
        or (str(name) == "comfy" and any(str(x) == "controlnet" for x in (fromlist or ())))
    ):
        _patch_controlnet_int8()
    if not _PATCHED and not _PATCHING and "comfy_quant_nvfp4" in str(name):
        _try_patch()
    elif not _PATCHED and not _PATCHING and fromlist:
        if any("comfy_quant_nvfp4" in str(x) for x in fromlist):
            _try_patch()
    if not _FASTDISK_INSTALLED and not _FASTDISK_INSTALLING:
        _mem = sys.modules.get("comfy.memory_management")
        if _mem is not None and hasattr(_mem, "read_tensor_file_slice_into"):
            _maybe_install_fastdisk()
    return mod
```

The installer now reports whether the guard armed (previously it returned nothing):

```python
def _install_fastdisk_support() -> bool:
    try:
        import importlib.util as _ilu2
        import os as _os2

        path = _os2.path.join(_ROOT, "patches", "comfy_aimdo_fastdisk_guard.py")
        spec = _ilu2.spec_from_file_location("_hswq_fastdisk_guard", path)
        mod = _ilu2.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return bool(mod.apply_comfy_aimdo_fastdisk_guard())
    except Exception as e:  # noqa: BLE001
        try:
            print(f"[HSWQ FastDisk] support install failed: {e}", flush=True)
        except Exception:
            pass
        return False
```

### 4.7 `ComfyUI-HSWQ-Loader-and-Tools/prestartup_script.py` - removed line

```python
_install_fastdisk_support()   # removed: it ran during prestartup and imported comfy -> torch
```

The `# Safety net: if comfy.ops somehow got imported before this hook could fire` block below it
is unchanged.

## 5. What it means

### 5.1 The check is now false for this installation

Measured after both prestartup scripts, in the real order (HSWQ first, this package second):

```
torch=False comfy=False transformers=True
```

`transformers` is imported by the docstring patches, and it does not import `torch` on this
stack (measured `torch=False` immediately after each patch call). With `torch` absent from
`sys.modules`, `main.py:262` cannot warn.

Real startup log after the change: `Torch already imported` 0 lines, `Potential Error in code`
0 lines, no `[ERROR]`, no traceback.

### 5.2 Both protections are still in place, and in the right order

The arming point moved into the window between `main.py:262` and `main.py:296`:

| Requirement | Measured |
| --- | --- |
| Hook fires after the check | arming happens inside `import comfy.utils` (`main.py:266`) |
| Guard armed before `init_devices` | `comfy_aimdo.control.init_devices` is only called at `main.py:296` |
| No CUDA context yet | `torch.cuda.is_initialized()` = `False` at the arming point |
| No aimdo worker threads yet | `threading.enumerate()` had no non-main threads at the arming point |
| Fast-disk sub-patches all applied | `control=True host_buffer=True model_vbar=True vram_buffer=True mem_slice=True` |
| Wrapped entry points before init | `control.init_devices` / `init_device` / `get_devctx` wrapped; `host_buffer` 10 markers; `model_vbar` 18; `vram_buffer` 4; `memory_management.read_tensor_file_slice_into._hswq_fastdisk_recoverable = True` |
| DiT-lock and INT8 ControlNet still armed | `[HSWQ DiT-lock] ... armed`, `[HSWQ INT8 CN] Retrofit armed` in the startup log |

Startup log order, verbatim:

```
[HSWQ FastDisk] v0.38 fast-disk support armed (control=True host_buffer=True model_vbar=True vram_buffer=True mem_slice=True)
aimdo: aimdo_setup_hooks: installing 6 hooks
comfy-aimdo inited for GPU: ... (VRAM: 16283 MB)
DynamicVRAM support detected and enabled
```

The guard is installed before aimdo's own hooks and before its device initialisation, which is
the ordering the guard's docstring requires.

### 5.3 The rotary alias behaves exactly as before

| Situation | Behaviour |
| --- | --- |
| ComfyUI exports `apply_rotary_emb` (0.46.0) | hook fires on module load, second gate wins, alias not installed - measured log: `apply_rotary_emb compat skipped: ComfyUI already exports apply_rotary_emb` |
| ComfyUI removed the symbol, nunchaku still imports it | hook installs `comfy.ldm.flux.math.apply_rope1` in the module's `exec_module`, i.e. before nunchaku's `from ... import apply_rotary_emb` returns |
| module already imported when prestartup runs | immediate path: alias applied without the hook |
| `QWENIMAGE_ROTARY_COMPAT` disabled | alias skipped, same as before |
| nunchaku no longer imports the symbol | alias skipped, same as before |

The three gates and their order are unchanged; only *when* the decision runs moved.

### 5.4 What is no longer hidden

The `logging` filter is gone. If any node - this one, HSWQ, or a third party - starts importing
`torch` during prestartup again, the warning will be printed, naming the state, not hidden. That
is the intended signal: the check exists to catch exactly that, and the only correct response is
to remove the early import.

The `comfy.memory_management` requirement also means the HSWQ work is retried if the first
attempt does not arm (`_FASTDISK_INSTALLED` now stores the real result instead of `True`
regardless), so a partially initialised `comfy.memory_management` cannot silently leave
fast-disk unprotected.

### 5.5 Cost

| Node | Before | After |
| --- | --- | --- |
| HSWQ prestartup | 11.8 s | 0.0 s (work moved to `comfy.utils` import) |
| This package prestartup | 49.1 s | 40.2 s (no `torch` import) |

### 5.6 Residual risk

The startup path was verified end to end by measurement, and every patch was confirmed applied
by its marker. A full workflow run (model load through the fast-disk path with an INT8/ConvRot
checkpoint) was not executed as part of this change; that is the remaining check.
