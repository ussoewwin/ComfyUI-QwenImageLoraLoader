import logging
import os

# Version information - must be at module level for ComfyUI Manager
__version__ = "2.6.0"

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("=" * 40 + " ComfyUI-QwenImageLoraLoader Initialization " + "=" * 40)

NODE_CLASS_MAPPINGS = {}
QWEN_V1_NODES = {}
QWEN_V1_NAMES = {}
QWEN_V2_NODES = {}
QWEN_V2_NAMES = {}
QWEN_V3_NODES = {}
QWEN_V3_NAMES = {}
ZIMAGETURBO_V1_NODES = {}
ZIMAGETURBO_V1_NAMES = {}
ZIMAGETURBO_V4_NODES = {}
NODE_CLASS_MAPPINGS = {}
QWEN_V1_NODES = {}
QWEN_V1_NAMES = {}
QWEN_V2_NODES = {}
QWEN_V2_NAMES = {}
QWEN_V3_NODES = {}
QWEN_V3_NAMES = {}
ZIMAGETURBO_V1_NODES = {}
ZIMAGETURBO_V1_NAMES = {}
ZIMAGETURBO_V4_NODES = {}
ZIMAGETURBO_V4_NAMES = {}

# --- Nunchaku availability probe (AMD/ROCm or missing install) ---
# nunchaku requires NVIDIA CUDA; on AMD/ROCm systems it is unavailable, so we
# disable every nunchaku-dependent node and keep only nunchaku-independent ones
# (Krea2 ControlNet LoRA loader, ControlNet node) registered.
try:
    import nunchaku  # noqa: F401
    _NUNCHAKU_AVAILABLE = True
except ImportError:
    _NUNCHAKU_AVAILABLE = False
    logger.warning(
        "[ROCm/AMD] nunchaku is not available on this system. "
        "Nunchaku QwenImage / Z-ImageTurbo LoRA nodes are disabled; "
        "Krea2 ControlNet LoRA loader remains available."
    )
# -----------------------------------------------------

# --- Nunchaku Monkey Patch Application ---
try:
    from .patches.nunchaku_patch import apply_nunchaku_patch
    if apply_nunchaku_patch():
        logger.info(
            "Successfully applied Nunchaku monkey patches (Qwen LoRA planar injection; "
            "Z-Image/SVDQ lazy Linear compat when ComfyUI defers Linear weights)."
        )
    else:
        logger.warning(
            "No Nunchaku patches applied (nunchaku / NunchakuQwenImageTransformerBlock not found)."
        )
except Exception as e:
    logger.error(f"Error importing/applying Nunchaku monkey patch: {e}")
# -----------------------------------------

# Nunchaku-dependent LoRA nodes are registered only when nunchaku is available
# (NVIDIA CUDA). On AMD/ROCm systems they are skipped; the nunchaku-independent
# Krea2 ControlNet LoRA loader is registered right after, unconditionally.
if _NUNCHAKU_AVAILABLE:
    try:
        from .nodes.lora.qwenimage import NunchakuQwenImageLoraLoader
        from .nodes.lora.qwenimage_v2 import GENERATED_NODES as QWEN_V2_NODES, GENERATED_DISPLAY_NAMES as QWEN_V2_NAMES
        from .nodes.lora.qwenimage_v3 import GENERATED_NODES as QWEN_V3_NODES, GENERATED_DISPLAY_NAMES as QWEN_V3_NAMES
        from .nodes.lora.qwenimage_v1 import GENERATED_NODES as QWEN_V1_NODES, GENERATED_DISPLAY_NAMES as QWEN_V1_NAMES
        from .nodes.lora.zimageturbo_v1 import GENERATED_NODES as ZIMAGETURBO_V1_NODES, GENERATED_DISPLAY_NAMES as ZIMAGETURBO_V1_NAMES
        # Z-Image-Turbo V3 is deprecated - removed from registration
        # from .nodes.lora.zimageturbo_v3 import GENERATED_NODES as ZIMAGETURBO_V3_NODES, GENERATED_DISPLAY_NAMES as ZIMAGETURBO_V3_NAMES
        from .nodes.lora.zimageturbo_v4 import GENERATED_NODES as ZIMAGETURBO_V4_NODES, GENERATED_DISPLAY_NAMES as ZIMAGETURBO_V4_NAMES

        # Add version to classes before creating NODE_CLASS_MAPPINGS
        NunchakuQwenImageLoraLoader.__version__ = __version__
        for node_class in QWEN_V2_NODES.values():
            node_class.__version__ = __version__
        for node_class in QWEN_V3_NODES.values():
            node_class.__version__ = __version__
        for node_class in QWEN_V1_NODES.values():
            node_class.__version__ = __version__
        for node_class in ZIMAGETURBO_V1_NODES.values():
            node_class.__version__ = __version__
        # Z-Image-Turbo V3 registration removed
        # for node_class in ZIMAGETURBO_V3_NODES.values():
        #     node_class.__version__ = __version__
        for node_class in ZIMAGETURBO_V4_NODES.values():
            node_class.__version__ = __version__

        NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraLoader"] = NunchakuQwenImageLoraLoader
        NODE_CLASS_MAPPINGS.update(QWEN_V2_NODES)
        NODE_CLASS_MAPPINGS.update(QWEN_V3_NODES)
        NODE_CLASS_MAPPINGS.update(QWEN_V1_NODES)
        NODE_CLASS_MAPPINGS.update(ZIMAGETURBO_V1_NODES)
        # Z-Image-Turbo V3 registration removed
        # NODE_CLASS_MAPPINGS.update(ZIMAGETURBO_V3_NODES)
        NODE_CLASS_MAPPINGS.update(ZIMAGETURBO_V4_NODES)
    except ImportError:
        logger.exception("LoRA nodes import failed:")

# Krea2 ControlNet LoRA loader is nunchaku-independent - always register it.
try:
    from .nodes.lora.krea2_controlnet_lora import Krea2ControlNetLoraLoader
    Krea2ControlNetLoraLoader.__version__ = __version__
    NODE_CLASS_MAPPINGS["Krea2ControlNetLoraLoader"] = Krea2ControlNetLoraLoader
except ImportError:
    logger.exception("Krea2 ControlNet LoRA node import failed:")

# Try to import ControlNet node separately - it may fail if comfy.ldm.lumina.controlnet is not available
try:
    from .nodes.controlnet import NunchakuQwenImageDiffsynthControlnet
    NunchakuQwenImageDiffsynthControlnet.__version__ = __version__
    NODE_CLASS_MAPPINGS["NunchakuQwenImageDiffsynthControlnet"] = NunchakuQwenImageDiffsynthControlnet
    logger.info("ControlNet node loaded successfully")
except ImportError:
    logger.warning("ControlNet node not available (comfy.ldm.lumina.controlnet not found). LoRA nodes will still work.")
except Exception as e:
    logger.warning(f"ControlNet node failed to load: {e}. LoRA nodes will still work.")

NODE_DISPLAY_NAME_MAPPINGS = {
    **({} if not _NUNCHAKU_AVAILABLE else {
        "NunchakuQwenImageLoraLoader": "Nunchaku Qwen Image LoRA Loader",
    }),
    **QWEN_V2_NAMES,
    **QWEN_V3_NAMES,
    **QWEN_V1_NAMES,
    **ZIMAGETURBO_V1_NAMES,
    # Z-Image-Turbo V3 registration removed
    # **ZIMAGETURBO_V3_NAMES,
    **ZIMAGETURBO_V4_NAMES
}
NODE_DISPLAY_NAME_MAPPINGS["Krea2ControlNetLoraLoader"] = "Krea2 controlnet lora loader"

# Add ControlNet display name only if the node was successfully loaded
if "NunchakuQwenImageDiffsynthControlnet" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["NunchakuQwenImageDiffsynthControlnet"] = "Nunchaku ZI Diffsynth Controlnet&Krea2 LoRA ControlNet"

# Register JavaScript extensions
WEB_DIRECTORY = "js"

# Make version available at module level for ComfyUI Manager
VERSION = __version__

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__version__", "VERSION"]
logger.info("=" * (80 + len(" ComfyUI-QwenImageLoraLoader Initialization ")))
