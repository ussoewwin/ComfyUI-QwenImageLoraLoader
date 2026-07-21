<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.5.2.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

Related issue: [#53](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/53)
Fix commit: `9e30661`

## 1. Reported issue

At ComfyUI startup, the following warning is printed while loading this custom node:

```
WARNING: Potential Error in code: Torch already imported, torch should never be imported before this point.
```

Issue #53 reports that `patches/nunchaku_patch.py` does `import torch` at module level.
That file is loaded during ComfyUI's prestartup phase (via `prestartup_script.py`), which
causes the guard in `main.py` to fire. The issue suggests moving `import torch` inside the
functions that need it and using `from __future__ import annotations`.

The warning is purely cosmetic (no functional error), but users find it alarming.

## 2. Root cause

`ComfyUI/main.py` runs prestartup scripts and then checks whether `torch` is already present
in `sys.modules`:

```python
# ComfyUI/main.py (around these lines)
execute_prestartup_script()          # line ~217
...
if 'torch' in sys.modules:           # line ~225
    logging.warning("WARNING: Potential Error in code: Torch already imported, "
                    "torch should never be imported before this point.")
```

This node **must** install the `apply_rotary_emb` compatibility shim during prestartup,
because `ComfyUI-nunchaku` loads before `ComfyUI-QwenImageLoraLoader` on Windows (listdir
order), so patching from `__init__.py` is too late. See
`QWEN_IMAGE_APPLY_ROTARY_EMB_COMPAT_FIX.md`.

Installing that shim requires importing `comfy.ldm.qwen_image.model`, whose import chain
reaches `comfy.ldm.flux.math`, and **that ComfyUI module imports `torch` at module level**:

```python
# ComfyUI/comfy/ldm/flux/math.py
import torch
from torch import Tensor
```

Therefore, even if `nunchaku_patch.py` stopped importing `torch` directly (as issue #53
suggests), the required `comfy.ldm` import during prestartup would still put `torch` into
`sys.modules` before `main.py`'s check. The `import torch` line in `nunchaku_patch.py` is
**not** the only cause; the early `torch` presence is unavoidable given the required shim.

Importantly, `cuda_malloc` and all CUDA-related environment setup in `main.py` run **before**
`execute_prestartup_script()`, so the early `torch` import does not corrupt any CUDA
configuration. The warning is cosmetic only.

## 3. Fix overview

Instead of trying to defer the `torch` import (which cannot remove the warning because of the
`comfy.ldm.flux.math` chain), we **suppress only that single warning message** with a
one-shot logging filter installed on the root logger during prestartup:

- The filter is installed at the very start of `prestartup_script.py`, before any step that
  imports `comfy.ldm`, and before `main.py` logs the warning.
- The filter drops exactly one log record whose message contains `"Torch already imported"`,
  then disables itself so every subsequent log record passes through untouched.
- An environment variable `QWENIMAGE_SUPPRESS_TORCH_WARNING=0` (also `false`/`no`/`off`/
  `disable`/`disabled`) turns suppression off, restoring the original warning for anyone who
  wants to see it.

This approach keeps the mandatory prestartup shim intact and touches nothing else in the log
stream.

## 4. Files changed

- `patches/nunchaku_patch.py` — added the warning filter class and the installer function.
- `prestartup_script.py` — load the patch module once and call the installer first.

## 5. Full changed code

### 5.1 `patches/nunchaku_patch.py` (added)

```python
_torch_preimport_warning_suppressed: bool = False
_TORCH_PREIMPORT_WARNING_MARKER = "Torch already imported"


def _torch_warning_suppression_disabled_by_env() -> bool:
    value = os.environ.get("QWENIMAGE_SUPPRESS_TORCH_WARNING", "").strip().lower()
    return value in ("0", "false", "no", "off", "disable", "disabled")


class _TorchPreimportWarningFilter(logging.Filter):
    """Drop ComfyUI's cosmetic 'Torch already imported' warning.

    The apply_rotary_emb compat shim must run in prestartup (before ComfyUI-nunchaku
    loads), and installing it requires importing comfy.ldm modules, which import torch.
    ComfyUI main.py then warns that torch entered sys.modules early. cuda_malloc and all
    CUDA env setup already ran before prestartup, so the early import is harmless; this
    filter hides only that single message and then lets every record through.
    """

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
    """Install a one-shot root-logger filter hiding the cosmetic torch pre-import warning.

    Must be called during prestartup (before ComfyUI main.py logs the warning).
    Set QWENIMAGE_SUPPRESS_TORCH_WARNING=0 to keep the warning visible.
    """
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

### 5.2 `prestartup_script.py` (changed)

```python
# Load the nunchaku patch module once; reuse it for warning suppression and the
# apply_rotary_emb compat shim.
_patch_module = None
try:
    _patch_module = _load_patch_module(
        "comfyui_qwenimageloraloader_nunchaku_patch_prestartup",
        _NUNCHAKU_PATCH_PATH,
    )
except Exception:
    logger.exception("ComfyUI-QwenImageLoraLoader prestartup: failed to load nunchaku patch module")

# Install the cosmetic 'Torch already imported' warning filter first, before any
# prestartup step imports comfy.ldm (which imports torch) and before main.py logs it.
if _patch_module is not None:
    try:
        if _patch_module.suppress_torch_preimport_warning():
            logger.debug(
                "ComfyUI-QwenImageLoraLoader prestartup: torch pre-import warning suppressed"
            )
    except Exception:
        logger.exception(
            "ComfyUI-QwenImageLoraLoader prestartup: torch warning suppression failed"
        )

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

## 6. What the code means

### `_TORCH_PREIMPORT_WARNING_MARKER` / `_torch_preimport_warning_suppressed`
The substring used to identify the target message, and a module-level flag ensuring the
filter is installed at most once (idempotent across repeated prestartup evaluations).

### `_torch_warning_suppression_disabled_by_env()`
Reads `QWENIMAGE_SUPPRESS_TORCH_WARNING`. If the user sets it to a falsey value, suppression
is skipped and the original warning stays visible. This preserves an explicit opt-out.

### `_TorchPreimportWarningFilter`
A `logging.Filter` subclass. `logging` calls `filter(record)` for every record that reaches a
logger/handler that owns the filter.
- Returning `False` drops the record; returning `True` lets it through.
- It matches only records whose rendered message contains `"Torch already imported"`.
- After dropping that one record, `self._suppressed = True` makes all later records pass
  through unconditionally, so no other logs are ever affected.
- `record.getMessage()` is wrapped in `try/except` so a malformed record can never break
  logging.

### `suppress_torch_preimport_warning()`
Installs the filter on the **root** logger via `logging.getLogger().addFilter(...)`. Because
`main.py` emits the warning with `logging.warning(...)` (root logger), a root-level filter is
guaranteed to see and drop it. Returns quickly if already installed or disabled by env.

### `prestartup_script.py` changes
The nunchaku patch module is now loaded **once** and reused. The suppression filter is
installed **first**, before the docstring patch and the `apply_rotary_emb` compat shim run
(those import `comfy.ldm`, which pulls in `torch`). Installing the filter before `main.py`
reaches its `if 'torch' in sys.modules:` check is what guarantees the warning is dropped.
Every branch is wrapped in `try/except` so a failure in one prestartup step cannot abort the
others.

### Why not "just move `import torch` inside functions" (as issue #53 suggested)
That change alone would **not** remove the warning: the mandatory `apply_rotary_emb` shim must
import `comfy.ldm.qwen_image.model`, whose chain imports `comfy.ldm.flux.math`, which imports
`torch` at module level. `torch` therefore enters `sys.modules` during prestartup regardless.
Suppressing the single cosmetic message is the correct, minimal fix; the compat shim itself
(still required upstream) is left unchanged.

