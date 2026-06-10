"""
Inject apply_rotary_emb on comfy.ldm.qwen_image.model before any custom node __init__.

ComfyUI-nunchaku loads before ComfyUI-QwenImageLoraLoader (Windows listdir order), so
__init__.py alone is too late. prestartup_script.py runs from main.execute_prestartup_script().
"""
import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

_PATCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches", "nunchaku_patch.py")


def _load_patch_module():
    spec = importlib.util.spec_from_file_location(
        "comfyui_qwenimageloraloader_nunchaku_patch_prestartup",
        _PATCH_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load patch module spec from {_PATCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _patch_module = _load_patch_module()
    if _patch_module.apply_qwen_image_apply_rotary_emb_compat():
        logger.info("ComfyUI-QwenImageLoraLoader prestartup: apply_rotary_emb compat applied")
    else:
        logger.debug(
            "ComfyUI-QwenImageLoraLoader prestartup: apply_rotary_emb compat not needed or already present"
        )
except Exception:
    logger.exception("ComfyUI-QwenImageLoraLoader prestartup: apply_rotary_emb compat failed")
