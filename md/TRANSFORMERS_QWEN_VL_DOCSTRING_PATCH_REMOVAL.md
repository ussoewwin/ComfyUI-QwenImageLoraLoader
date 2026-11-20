# Removing the Qwen VL CausalLM docstring patch

| item | value |
| :--- | :--- |
| release | v2.6.9 |
| commit | `8ec3c00` |
| date | 2026-11-20 |
| removed file | `patches/transformers_qwen_vl_docstring_patch.py` (248 lines, blob `7751b492f53c`) |
| modified file | `prestartup_script.py` (4,107 B to 3,383 B, -15 lines) |
| measured on | ComfyUI 0.46.0 (`a277b5bf`), Python 3.13.17, torch 2.15.0+cu134, transformers 5.23.0, nunchaku 1.3.0.dev20261030 |

This document records one change: the removal of the preventive patch that silenced the
Qwen3-VL / Qwen2.5-VL `[ERROR] ... loss/logits ... but not documented` lines printed by
`transformers` at import time. The measurement that justifies the removal, the exact bytes
that were removed, and the procedure to restore them are all below.

## 1. What the problem was

### 1.1 The line itself

`transformers` validates docstrings against signatures while a model module is imported.
The validator collects the parameters it cannot match and prints them at the end of the pass:

```python
# transformers/utils/auto_docstring.py
# L3718-3719 (function signature parameters)
undocumented_parameters.append(
    f"[ERROR] `{param_name}` is part of {func.__qualname__}'s signature, but not documented. "
    f"Make sure to add it to the docstring of the function in {_source_file}."
)
# L4091-4092 (TypedDict / kwargs parameters)
undocumented_parameters.append(
    f"[ERROR] `{param_name}` is part of {kwarg_param.annotation.__args__[0].__qualname__}, "
    f"but not documented. Make sure to add it to the docstring of the function in "
    f"{func.__code__.co_filename}."
)
# L4214-4215 (emission)
if len(undocumented_parameters) > 0:
    print("\n".join(undocumented_parameters))
```

The emission is a plain `print()` to **stdout**, not a logger and not stderr. That detail
matters for section 2 and for anyone writing a probe against this behaviour.

### 1.2 Why Qwen VL modules produced it

The removed patch recorded the mechanism as follows: the Qwen VL modelings declared their own causal-LM output class that inherited
`CausalLMOutputWithPast` and merged the parent documentation through the shared
`ModelOutputArgs` argument block. Because `ModelOutputArgs` did not carry `loss` and `logits`,
the merge left two annotations undocumented and the validator printed two `[ERROR]` lines
on every import of:

- `transformers.models.qwen3_vl.modeling_qwen3_vl`
- `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl`

Nothing failed. Inference was unaffected. The damage was to the log: the lines appear at
import time on every start that loads a Qwen VL text encoder, and a log that contains
expected errors is a log where a real error can hide.

### 1.3 What the patch did (v2.4.7, now removed)

`patches/transformers_qwen_vl_docstring_patch.py` wrapped
`transformers.utils.auto_docstring.get_args_doc_from_source`, and for
`ModelOutputArgs` merged back `loss` and `logits` entries parsed from
`CausalLMOutputWithPast.__doc__`. Two gates decided whether the wrapper was installed:

1. native gate - does `ModelOutputArgs` already document `loss` and `logits`;
2. import gate - a subprocess import of the two modelings, with stdout captured, counting
   lines that contain `[ERROR]` and one of `loss` / `logits`.

If gate 1 said yes, or gate 2 reported a clean import, the wrapper was **not** installed.
If gate 2 could not run (`None`: timeout, non-zero exit), the wrapper **was** installed.
The patch was therefore inert on any stack where the import was already clean - which is
exactly the state measured in section 2.

## 2. Why the patch was removed

### 2.1 The upstream defect is gone at its root, not papered over

Measured on transformers 5.23.0:

| probe | result |
| :--- | :--- |
| `ModelOutputArgs` documents `loss` / `logits` | **False** (the shared block still omits them) |
| `qwen3_vl` modeling references `ModelOutputArgs` | **0 occurrences** |
| custom output classes in `qwen3_vl` modeling | `BaseModelOutputWithDeepstackFeatures(BaseModelOutputWithPooling)` only |
| causal-LM return | plain `CausalLMOutputWithPast` (annotation L1298, construction L1363) |
| `[ERROR]` lines at import of `qwen3_vl` + `qwen2_5_vl`, stdout | **0** |
| `[ERROR]` lines at import of `qwen3_vl` + `qwen2_5_vl`, stderr | **0** |
| `[ERROR]` emission code still present in `auto_docstring` | **yes** (L3719, L4092, L4215) |

So the silence does not come from the validator being disabled, and not from the shared
argument block being completed. It comes from the modelings no longer merging parent docs
through `ModelOutputArgs` at all. The line has nothing left to print.

Both channels were captured separately for the second measurement, because a probe that
reads only stdout would report a clean import even if the line had moved to stderr.

### 2.2 The guard cost every start, and could mask a regression

Measured cost of the removed probe on this machine: **29.82 s** for one call
(`_import_probe_reports_clean()`, which spawns a Python subprocess that imports both
modelings). That cost was paid at every ComfyUI start, where the patch's own gate then
decided to do nothing. A guard that never fires but always costs is dead weight.

The second reason is stronger. The wrapper *suppressed* the diagnostic: if a future
transformers release reintroduced the merge, the wrapper would have merged `loss`/`logits`
back and the line would not appear - the regression would be invisible by construction.
Removing the wrapper makes any recurrence print again, on the channel that already exists
(stdout), with no probe involved.

### 2.3 Scope of the removal

Only the Qwen VL patch was removed. `patches/transformers_image_processor_kwargs_patch.py`
(deepseek_vl_hybrid / kimi_k25 / paddleocr_vl docstring fields) is a different set of
classes, a different upstream defect and a different fix; it remains, and it still runs its
own probe at start (measured 28.14 s, same construction).

Untouched: ComfyUI core, `transformers` site-packages, the nunchaku wheel, and every other
patch in this package (`apply_rotary_emb` compat, `SVDQW4A4Linear.from_linear`,
`fuse_to_svdquant_linear`, `ZImageModelPatcher` `fast_disk`, `copy_params_into`, planar
injection).

## 3. Changed files

| file | change | size |
| :--- | :--- | :--- |
| `patches/transformers_qwen_vl_docstring_patch.py` | deleted | 248 lines, blob `7751b492f53c` |
| `prestartup_script.py` | 15 lines removed (1 constant + 14-line block) | 4,107 B to 3,383 B |
| `RELEASE_NOTES/changelog.md` | v2.6.9 entry (`**Removed**`) | +8 / -1 lines |
| `zhmd/changelogzh.md` | v2.6.9 entry (removal bullet, Chinese side) | +8 / -1 lines |

No other file changed. The removal is one file plus one wiring block.

## 4. The code

### 4.1 The deleted file, in full

```python
"""
Patch transformers auto_docstring for Qwen VL CausalLM ModelOutput classes.

When Hugging Face transformers merges CausalLMOutputWithPast docs into Qwen VL
ModelOutput subclasses, validation falls back to ModelOutputArgs — which lacks
loss/logits — and prints [ERROR] at import time.

Fully automatic at ComfyUI prestartup (no user env vars):
  - Probe upstream ModelOutputArgs before installing any wrapper.
  - If upstream already documents loss/logits, skip (no wrapper).
  - If a subprocess import probe shows clean stdout, skip.
  - Otherwise install the wrapper (default when upstream is still broken).

Fix: wrap get_args_doc_from_source to merge loss/logits into the returned dict
while upstream ModelOutputArgs still omits those fields.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_PATCH_TAG = "_qwen_lora_loader_causal_lm_docstring_patch"
_CAUSAL_LM_OUTPUT_FIELDS = ("loss", "logits")

_QWEN_VL_MODELING_MODULES = (
    "transformers.models.qwen3_vl.modeling_qwen3_vl",
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
)

_patch_applied: bool = False
_original_get_args_doc_from_source: Optional[Callable[..., dict]] = None


def _qwen_vl_modeling_already_imported() -> bool:
    return any(name in sys.modules for name in _QWEN_VL_MODELING_MODULES)


def _upstream_model_output_args_has_causal_lm_fields(auto_docstring_module) -> bool:
    """Probe upstream native state (never via patched get_args_doc_from_source)."""
    model_output_args = auto_docstring_module.ModelOutputArgs
    for field in _CAUSAL_LM_OUTPUT_FIELDS:
        entry = getattr(model_output_args, field, None)
        if not isinstance(entry, dict) or not entry.get("description"):
            return False
    return True


def _build_causal_lm_extra_args(auto_docstring_module) -> Dict[str, dict]:
    """Extract loss/logits entries from CausalLMOutputWithPast.__doc__."""
    try:
        modeling_outputs = importlib.import_module("transformers.modeling_outputs")
    except ImportError:
        return {}

    parent_doc = getattr(modeling_outputs.CausalLMOutputWithPast, "__doc__", None)
    if not parent_doc:
        return {}

    normalized = auto_docstring_module.set_min_indent(parent_doc.strip(), 0)
    documented_params, _remainder = auto_docstring_module.parse_docstring(
        normalized,
        max_indent_level=4,
    )

    extra: Dict[str, dict] = {}
    for field in _CAUSAL_LM_OUTPUT_FIELDS:
        param = documented_params.get(field)
        if not param:
            continue
        extra[field] = {
            "description": param.get("description", ""),
            "shape": param.get("shape"),
            "optional": param.get("optional", False),
            "additional_info": param.get("additional_info", ""),
            "type": param.get("type", ""),
        }
        if param.get("default") is not None:
            extra[field]["default"] = param["default"]
    return extra


def _model_output_args_requested(args_classes: Any, model_output_args_type: type) -> bool:
    if args_classes is model_output_args_type:
        return True
    if isinstance(args_classes, (list, tuple)):
        return model_output_args_type in args_classes
    return False


def _make_patched_get_args_doc_from_source(
    auto_docstring_module,
    original: Callable[..., dict],
) -> Callable[..., dict]:
    model_output_args_type = auto_docstring_module.ModelOutputArgs
    cached_extra: Dict[str, dict] = {}

    def patched_get_args_doc_from_source(args_classes: Any) -> dict:
        result = original(args_classes)

        if not _model_output_args_requested(args_classes, model_output_args_type):
            return result

        if all(field in result for field in _CAUSAL_LM_OUTPUT_FIELDS):
            return result

        nonlocal cached_extra
        if not cached_extra:
            cached_extra = _build_causal_lm_extra_args(auto_docstring_module)

        if not cached_extra:
            return result

        merged = dict(result)
        for field in _CAUSAL_LM_OUTPUT_FIELDS:
            if field in cached_extra:
                merged.setdefault(field, cached_extra[field])
        return merged

    setattr(patched_get_args_doc_from_source, _PATCH_TAG, True)
    return patched_get_args_doc_from_source


def _import_probe_reports_clean() -> Optional[bool]:
    """
    True: Qwen VL imports emit no [ERROR] loss/logits on stdout.
    False: errors still present (patch may be needed).
    None: probe could not run.
    """
    python_exe = sys.executable
    code = (
        "import importlib, io, contextlib\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        "    importlib.import_module('transformers.models.qwen3_vl.modeling_qwen3_vl')\n"
        "    importlib.import_module('transformers.models.qwen2_5_vl.modeling_qwen2_5_vl')\n"
        "lines = buf.getvalue().splitlines()\n"
        "errs = [l for l in lines if '[ERROR]' in l and ('loss' in l or 'logits' in l)]\n"
        "print('CLEAN' if not errs else 'ERRORS')\n"
    )
    try:
        proc = subprocess.run(
            [python_exe, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("CausalLM docstring import probe failed: %s", exc)
        return None

    if proc.returncode != 0:
        logger.debug(
            "CausalLM docstring import probe exit %s stderr=%s",
            proc.returncode,
            proc.stderr[:500] if proc.stderr else "",
        )
        return None

    last_line = (proc.stdout or "").strip().splitlines()
    if not last_line:
        return None
    status = last_line[-1].strip()
    if status == "CLEAN":
        return True
    if status == "ERRORS":
        return False
    return None


def apply_transformers_causal_lm_docstring_patch() -> bool:
    """
    Install get_args_doc_from_source wrapper unless upstream already fixed the issue.

    Returns True if wrapper is active (or was already applied), False if skipped.
    """
    global _patch_applied, _original_get_args_doc_from_source

    if _patch_applied:
        return True

    if _qwen_vl_modeling_already_imported():
        logger.warning(
            "CausalLM ModelOutput docstring patch skipped: Qwen VL modeling modules "
            "already imported before prestartup — restart ComfyUI"
        )
        return False

    try:
        auto_docstring_module = importlib.import_module("transformers.utils.auto_docstring")
    except ImportError:
        logger.debug("transformers.utils.auto_docstring not available; patch skipped")
        return False

    get_args = getattr(auto_docstring_module, "get_args_doc_from_source", None)
    if get_args is None:
        return False

    if getattr(get_args, _PATCH_TAG, False):
        _patch_applied = True
        return True

    if _upstream_model_output_args_has_causal_lm_fields(auto_docstring_module):
        logger.info(
            "CausalLM ModelOutput docstring patch skipped: transformers ModelOutputArgs "
            "already documents loss and logits (upstream fixed — patch not installed)"
        )
        return False

    import_probe = _import_probe_reports_clean()
    if import_probe is True:
        logger.info(
            "CausalLM ModelOutput docstring patch skipped: Qwen VL ModelOutput docstrings "
            "resolve loss/logits without docstring errors (upstream fixed — patch not installed)"
        )
        return False

    _original_get_args_doc_from_source = get_args
    auto_docstring_module.get_args_doc_from_source = _make_patched_get_args_doc_from_source(
        auto_docstring_module,
        _original_get_args_doc_from_source,
    )
    _patch_applied = True

    logger.info(
        "Patched transformers.utils.auto_docstring.get_args_doc_from_source for Qwen VL "
        "CausalLM ModelOutput docstrings (loss/logits); removes when upstream adds them"
    )
    return True


def is_patch_applied() -> bool:
    return _patch_applied


def is_patch_wrapped() -> bool:
    try:
        auto_docstring_module = importlib.import_module("transformers.utils.auto_docstring")
    except ImportError:
        return False
    get_args = getattr(auto_docstring_module, "get_args_doc_from_source", None)
    return get_args is not None and getattr(get_args, _PATCH_TAG, False)
```

### 4.2 The removed lines in `prestartup_script.py`

```diff
@@ -12,7 +12,6 @@ logger = logging.getLogger(__name__)
 
 _PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches")
 _NUNCHAKU_PATCH_PATH = os.path.join(_PATCH_DIR, "nunchaku_patch.py")
-_DOCSTRING_PATCH_PATH = os.path.join(_PATCH_DIR, "transformers_qwen_vl_docstring_patch.py")
 _IP_KWARGS_PATCH_PATH = os.path.join(_PATCH_DIR, "transformers_image_processor_kwargs_patch.py")
 
 
@@ -49,20 +48,6 @@ if _patch_module is not None:
             "ComfyUI-QwenImageLoraLoader prestartup: torch warning suppression failed"
         )
 
-try:
-    _docstring_patch_module = _load_patch_module(
-        "comfyui_qwenimageloraloader_docstring_patch_prestartup",
-        _DOCSTRING_PATCH_PATH,
-    )
-    if _docstring_patch_module.apply_transformers_causal_lm_docstring_patch():
-        logger.info("ComfyUI-QwenImageLoraLoader prestartup: CausalLM ModelOutput docstring patch applied")
-    else:
-        logger.debug(
-            "ComfyUI-QwenImageLoraLoader prestartup: CausalLM ModelOutput docstring patch not applied"
-        )
-except Exception:
-    logger.exception("ComfyUI-QwenImageLoraLoader prestartup: CausalLM ModelOutput docstring patch failed")
-
 # ImageProcessorKwargs docstring patch (deepseek_vl_hybrid / kimi_k25 / paddleocr_vl
 # [ERROR] "but not documented" noise): wraps get_args_doc_from_source at prestartup,
 # before any custom node imports those transformers modules.
```

### 4.3 `prestartup_script.py` after the removal, in full

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
_IP_KWARGS_PATCH_PATH = os.path.join(_PATCH_DIR, "transformers_image_processor_kwargs_patch.py")


def _load_patch_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load patch module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

## 5. What it means

### 5.1 Startup order, before and after

`prestartup_script.py` runs from ComfyUI's prestartup phase, before any custom node
`__init__` and before the pack whose modules it touches is imported. The order inside it is
deliberate: the torch pre-import warning filter is installed first, then the patches that
have to precede other packs.

| position | before | after |
| :--- | :--- | :--- |
| 1 | torch pre-import warning filter | torch pre-import warning filter |
| 2 | Qwen VL CausalLM docstring patch (gates + possible wrapper) | - |
| 3 | ImageProcessorKwargs docstring patch | ImageProcessorKwargs docstring patch |
| 4 | `apply_rotary_emb` compat | `apply_rotary_emb` compat |

At runtime nothing else changes: on this stack the removed patch always took the skip
branch, so **no wrapper was ever installed** while it existed. Behaviour is identical;
what changed is the cost and the visibility.

### 5.2 What the package still guarantees, and what it no longer does

Still guaranteed by other code: the Qwen Image wrapper's weight-supply contract for
ControlNet-borrowed modules, the nunchaku lazy-Linear fixes, the `fast_disk` acceptance on
`ZImageModelPatcher`, the wtscale-safe `copy_params_into`, the planar LoRA injection, and
the `apply_rotary_emb` compat that now self-retires because ComfyUI 0.46.0 exports the
symbol natively.

No longer done: nothing suppresses the Qwen VL docstring line. If it returns, it prints.

### 5.3 How to check the claim, without trusting a log

A start log that contains no `[ERROR]` proves nothing by itself: the modelings are imported
when a Qwen VL text encoder is loaded, not at every start. The measurement that decides is
the import itself, on both channels:

```python
import contextlib, io, importlib
out, err = io.StringIO(), io.StringIO()
mods = ("transformers.models.qwen3_vl.modeling_qwen3_vl",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
    for m in mods:
        importlib.import_module(m)
print("stdout:", len([l for l in out.getvalue().splitlines() if "[ERROR]" in l]))
print("stderr:", len([l for l in err.getvalue().splitlines() if "[ERROR]" in l]))
```

Both counts must be 0. Measured on 2026-11-20: `stdout: 0`, `stderr: 0`.

### 5.4 Limit of this change

The removal is correct for transformers 5.23.0 and later, where the modelings do not use
`ModelOutputArgs`. On an older transformers (the 5.15.x line that motivated the patch) the
line can still appear. In that environment the patch is restorable, and section 6 gives the
exact steps. This package supports Python 3.10-3.12 per `pyproject.toml`; the removal does
not narrow or widen that range.

## 6. How to restore the patch

### 6.1 Restore with git (preferred)

```bash
# from the repository root
git checkout 8ec3c00^ -- patches/transformers_qwen_vl_docstring_patch.py prestartup_script.py
git status --short
```

`8ec3c00^` is the commit immediately before the removal, so this restores both files to
their exact pre-removal bytes. Alternatively, revert the commit and keep it as a commit:

```bash
git revert --no-edit 8ec3c00
```

### 6.2 Restore by hand

1. Create `patches/transformers_qwen_vl_docstring_patch.py` with the exact content in
   section 4.1 (the authoritative bytes are blob `7751b492f53c13998ffedceaa925681a7760635c`; if the file is
   pasted by hand, compare its diff against that blob before committing).
2. Add the constant next to the other patch paths:

```python
_DOCSTRING_PATCH_PATH = os.path.join(_PATCH_DIR, "transformers_qwen_vl_docstring_patch.py")
```

3. Add the block back after the warning-filter block (the removed hunk in section 4.2):

```python
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
```

### 6.3 Verify the restore

```bash
python -m py_compile prestartup_script.py
python -c "import logging, runpy; logging.basicConfig(level=logging.INFO); runpy.run_path('prestartup_script.py')"
```

On transformers 5.23.0 the expected line is the skip branch, because gate 2 reports a clean
import and no wrapper is installed:

```text
INFO ... CausalLM ModelOutput docstring patch skipped: Qwen VL ModelOutput docstrings
resolve loss/logits without docstring errors (upstream fixed - patch not installed)
```

To confirm that no wrapper is active, check the function object:

```python
import importlib
m = importlib.import_module("transformers.utils.auto_docstring")
print(bool(getattr(m.get_args_doc_from_source, "_qwen_lora_loader_causal_lm_docstring_patch", False)))
```

`False` means the live function is the native one.

### 6.4 Cost of restoring

The restored patch adds one subprocess import per start (measured 29.82 s on this machine)
and, on transformers 5.23.0, always ends in the skip branch. It also re-masks the
diagnostic: if the upstream defect ever returns while the wrapper is installed, the line
will not be printed. Restore it only on stacks where the line is actually observed.
