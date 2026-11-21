"""
Inject apply_rotary_emb on comfy.ldm.qwen_image.model before any custom node __init__.

ComfyUI-nunchaku loads before ComfyUI-QwenImageLoraLoader (Windows listdir order), so
__init__.py alone is too late. prestartup_script.py runs from main.execute_prestartup_script().
"""
import importlib
import importlib.machinery
import importlib.util
import locale
import logging
import os
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source-encoding rescue (upstream support).
#
# Python decodes a .py file as UTF-8 unless it carries a coding cookie. When an
# editor or an installer pipeline re-saves a file in a legacy code page (ANSI,
# cp936/GBK, cp932, cp1252), the file is no longer valid UTF-8 and the import
# dies with "(unicode error) 'utf-8' codec can't decode byte 0xNN". The pack
# that owns the file then fails to load, whichever pack it is -- not only this
# one.
#
# The stock loader still runs first, byte for byte, .pyc caching included. This
# hook only steps in when that decode fails: it retries the same bytes with the
# local code page and the common legacy ones, and logs every rescued file with
# the code page it used. Nothing is silenced: the log names the file, the byte
# offset, and the fact that its text may differ from the author's original.
# ---------------------------------------------------------------------------
_RESCUED_SOURCES = []
_RESCUE_CANDIDATES = ("cp936", "cp932", "cp1252", "big5", "shift_jis", "latin-1")


def _decode_failure_offset(message):
    match = re.search(r"position (\d+)", message or "")
    return int(match.group(1)) if match else -1


def _is_decode_failure(exc):
    text = str(exc)
    return "codec can't decode" in text or "unicode error" in text


def _rescue_compile(data, path, exc):
    """Compile bytes the stock UTF-8 decode rejected, using a legacy code page."""
    raw = data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8", "surrogatepass")
    candidates = []
    preferred = locale.getpreferredencoding(False)
    if preferred:
        candidates.append(preferred)
    candidates.extend(_RESCUE_CANDIDATES)
    tried = set()
    for encoding in candidates:
        key = encoding.lower()
        if key in tried:
            continue
        tried.add(key)
        try:
            source = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        try:
            code = compile(source, path, "exec", dont_inherit=True)
        except SyntaxError:
            continue
        _RESCUED_SOURCES.append((path, encoding))
        logger.warning(
            "ComfyUI-QwenImageLoraLoader prestartup: source rescue: %s is not valid UTF-8 "
            "(first invalid byte at %d); compiled as %s. This file was saved outside UTF-8 "
            "(ANSI / cp936 / cp932 / cp1252). Re-download it or re-save it as UTF-8; text from "
            "this file may differ from the author's original.",
            path, _decode_failure_offset(str(exc)), encoding,
        )
        return code
    raise exc


def _install_source_encoding_rescue():
    """Cover every source import in this process, before ComfyUI loads the packs."""
    disabled = os.environ.get("HSWQ_NO_ENCODING_RESCUE", "").strip()
    if disabled not in ("", "0"):
        logger.warning(
            "ComfyUI-QwenImageLoraLoader prestartup: source rescue disabled by "
            "HSWQ_NO_ENCODING_RESCUE=%s",
            disabled,
        )
        return False

    loader_cls = importlib.machinery.SourceFileLoader
    if not getattr(loader_cls.get_code, "__hswq_encoding_rescue__", False):
        original_get_code = loader_cls.get_code

        def get_code(self, fullname):
            try:
                return original_get_code(self, fullname)
            except SyntaxError as exc:
                if not _is_decode_failure(exc):
                    raise
                path = self.get_filename(fullname)
                return _rescue_compile(self.get_data(path), path, exc)

        get_code.__hswq_encoding_rescue__ = True
        loader_cls.get_code = get_code

    # Subclasses that override get_code (ComfyUI's governance loader does) reach
    # the source through source_to_code instead; wrap the class that actually
    # defines it. importlib.abc.SourceLoader is a different class, so resolve it
    # from the real loader's MRO rather than assuming a module.
    source_loader = None
    for klass in loader_cls.__mro__:
        if "source_to_code" in klass.__dict__:
            source_loader = klass
            break
    if source_loader is None:
        return True
    if not getattr(source_loader.source_to_code, "__hswq_encoding_rescue__", False):
        original_source_to_code = source_loader.source_to_code

        def source_to_code(self, data, path, *, _optimize=-1):
            try:
                return original_source_to_code(self, data, path, _optimize=_optimize)
            except SyntaxError as exc:
                if not _is_decode_failure(exc):
                    raise
                return _rescue_compile(data, path, exc)

        source_to_code.__hswq_encoding_rescue__ = True
        source_loader.source_to_code = source_to_code
    return True


_ENCODING_RESCUE_INSTALLED = False
try:
    _ENCODING_RESCUE_INSTALLED = _install_source_encoding_rescue()
except Exception:
    logger.exception(
        "ComfyUI-QwenImageLoraLoader prestartup: source rescue could not be installed"
    )


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
