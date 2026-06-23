"""
Patch transformers auto_docstring for Qwen VL CausalLM ModelOutput classes.

When Hugging Face transformers merges CausalLMOutputWithPast docs into Qwen VL
ModelOutput subclasses, validation falls back to ModelOutputArgs — which lacks
loss/logits — and prints [ERROR] at import time.

Upstream self-disable (same pattern as v2.4.6 apply_rotary_emb compat):
  - Probe upstream ModelOutputArgs before installing any wrapper.
  - If upstream already documents loss/logits, skip (no wrapper, harmless).
  - If import probe shows clean stdout, skip.
  - Disable via TRANSFORMERS_CAUSAL_LM_DOCSTRING_PATCH=0|false|off|no|disable|disabled

Fix: wrap get_args_doc_from_source to merge loss/logits into the returned dict
only while upstream ModelOutputArgs still omits those fields.
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


def _patch_disabled_by_env() -> bool:
    value = os.environ.get("TRANSFORMERS_CAUSAL_LM_DOCSTRING_PATCH", "").strip().lower()
    return value in ("0", "false", "no", "off", "disable", "disabled")


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
            env={**os.environ, "TRANSFORMERS_CAUSAL_LM_DOCSTRING_PATCH": "0"},
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

    if _patch_disabled_by_env():
        logger.info(
            "CausalLM ModelOutput docstring patch skipped: "
            "TRANSFORMERS_CAUSAL_LM_DOCSTRING_PATCH is disabled"
        )
        return False

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
