# -*- coding: utf-8 -*-
"""
Patch transformers auto_docstring for ImageProcessorKwargs TypedDict fields.

transformers 5.15.x auto_docstring validation parses each *ImageProcessorKwargs
TypedDict's own __doc__ at import time and prints [ERROR] for every annotation
key that the parser cannot find in that docstring. Known upstream breakages
(transformers 5.15.1):

  - DeepseekVLHybridImageProcessorKwargs: the `high_res_size` docstring entry
    has a stray leading space, so the docstring parser never matches it.
  - Kimi_K25ImageProcessorKwargs: the docstring documents `merge_kernel_size`,
    but the TypedDict annotation is `merge_size`.
  - PaddleOCRVLImageProcessorKwargs: `min_pixels` / `max_pixels` annotations
    are not documented at all.

Fix: wrap transformers.utils.auto_docstring.get_args_doc_from_source so the
returned source dict also carries those four fields. _get_parameter_info()
falls back to source_args_dict when a TypedDict key is missing from its own
docstring, which stops the [ERROR] output without touching site-packages.

Fully automatic at ComfyUI prestartup (no user env vars):
  - Probe upstream state before installing any wrapper.
  - If the source dict already documents the fields, skip (no wrapper).
  - If a subprocess import probe shows clean stdout, skip (no wrapper).
  - Otherwise install the wrapper (default when upstream is still broken).

The wrapper is inert once upstream fixes the docstrings: the fields are only
consulted for TypedDict annotations that declare them, and only when they are
missing from that TypedDict's own docstring.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_PATCH_TAG = "_qwen_lora_loader_image_processor_kwargs_patch"

# Fields missing from the affected TypedDict docstrings, added to the
# ImageProcessorArgs/VideoProcessorArgs source dict as a fallback.
_EXTRA_KWARGS: Dict[str, Dict[str, Any]] = {
    "high_res_size": {
        "type": "dict",
        "shape": None,
        "description": """
    Size of the high resolution output image after resizing. Can be overridden by the `high_res_size` parameter in the `preprocess` method.
    """,
    },
    "merge_size": {
        "type": "int",
        "shape": None,
        "description": """
    The merge size of the vision encoder to llm encoder.
    """,
    },
    "min_pixels": {
        "type": "int",
        "shape": None,
        "description": """
    The minimum allowed number of pixels for the resized image. Ensures the total pixel count does not fall below this value after resizing.
    """,
    },
    "max_pixels": {
        "type": "int",
        "shape": None,
        "description": """
    The maximum allowed number of pixels for the resized image. Ensures the total pixel count does not exceed this value after resizing.
    """,
    },
}

_AFFECTED_MODULES = (
    "transformers.models.deepseek_vl_hybrid.image_processing_deepseek_vl_hybrid",
    "transformers.models.deepseek_vl_hybrid.image_processing_pil_deepseek_vl_hybrid",
    "transformers.models.kimi_k25.image_processing_kimi_k25",
    "transformers.models.paddleocr_vl.image_processing_paddleocr_vl",
    "transformers.models.paddleocr_vl.image_processing_pil_paddleocr_vl",
)

_patch_applied: bool = False
_original_get_args_doc_from_source: Optional[Callable[..., dict]] = None


def _affected_modules_already_imported() -> bool:
    return any(name in sys.modules for name in _AFFECTED_MODULES)


def _upstream_documents_extra_kwargs(auto_docstring_module) -> bool:
    """Probe upstream native state (never via the patched get_args_doc_from_source)."""
    args_classes = []
    for name in ("ImageProcessorArgs", "VideoProcessorArgs"):
        cls = getattr(auto_docstring_module, name, None)
        if cls is not None:
            args_classes.append(cls)
    if not args_classes:
        return False
    try:
        source = auto_docstring_module.get_args_doc_from_source(args_classes)
    except Exception:
        return False
    return all(field in source for field in _EXTRA_KWARGS)


def _image_processor_args_requested(args_classes: Any, image_processor_args_type: type) -> bool:
    if args_classes is image_processor_args_type:
        return True
    if isinstance(args_classes, (list, tuple)):
        return image_processor_args_type in args_classes
    return False


def _make_patched_get_args_doc_from_source(
    auto_docstring_module,
    original: Callable[..., dict],
) -> Callable[..., dict]:
    image_processor_args_type = auto_docstring_module.ImageProcessorArgs

    def patched_get_args_doc_from_source(args_classes: Any) -> dict:
        result = original(args_classes)

        if not _image_processor_args_requested(args_classes, image_processor_args_type):
            return result

        if all(field in result for field in _EXTRA_KWARGS):
            return result

        merged = dict(result)
        for field, entry in _EXTRA_KWARGS.items():
            merged.setdefault(field, dict(entry))
        return merged

    setattr(patched_get_args_doc_from_source, _PATCH_TAG, True)
    return patched_get_args_doc_from_source


def _import_probe_reports_clean() -> Optional[bool]:
    """
    True: affected transformers modules import with no [ERROR] ... not documented lines.
    False: errors still present (patch may be needed).
    None: probe could not run.
    """
    python_exe = sys.executable
    module_list = ", ".join(repr(m) for m in _AFFECTED_MODULES)
    code = (
        "import importlib, io, contextlib\n"
        "buf = io.StringIO()\n"
        "with contextlib.redirect_stdout(buf):\n"
        f"    for m in ({module_list}):\n"
        "        importlib.import_module(m)\n"
        "lines = buf.getvalue().splitlines()\n"
        "errs = [l for l in lines if '[ERROR]' in l and 'but not documented' in l]\n"
        "print('CLEAN' if not errs else 'ERRORS')\n"
    )
    try:
        proc = subprocess.run(
            [python_exe, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("ImageProcessorKwargs docstring import probe failed: %s", exc)
        return None

    if proc.returncode != 0:
        logger.debug(
            "ImageProcessorKwargs docstring import probe exit %s stderr=%s",
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


def apply_transformers_image_processor_kwargs_patch() -> bool:
    """
    Install get_args_doc_from_source wrapper unless upstream already fixed the issue.

    Returns True if the wrapper is active (or was already applied), False if skipped.
    """
    global _patch_applied, _original_get_args_doc_from_source

    if _patch_applied:
        return True

    if _affected_modules_already_imported():
        logger.warning(
            "ImageProcessorKwargs docstring patch skipped: affected transformers modules "
            "already imported before prestartup; restart ComfyUI"
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

    if _upstream_documents_extra_kwargs(auto_docstring_module):
        logger.info(
            "ImageProcessorKwargs docstring patch skipped: transformers source dict "
            "already documents the fields (upstream fixed; patch not installed)"
        )
        return False

    import_probe = _import_probe_reports_clean()
    if import_probe is True:
        logger.info(
            "ImageProcessorKwargs docstring patch skipped: affected transformers modules "
            "import without docstring errors (upstream fixed; patch not installed)"
        )
        return False

    _original_get_args_doc_from_source = get_args
    auto_docstring_module.get_args_doc_from_source = _make_patched_get_args_doc_from_source(
        auto_docstring_module,
        _original_get_args_doc_from_source,
    )
    _patch_applied = True

    logger.info(
        "Patched transformers.utils.auto_docstring.get_args_doc_from_source for "
        "ImageProcessorKwargs fields (high_res_size/merge_size/min_pixels/max_pixels); "
        "removes when upstream fixes the docstrings"
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
