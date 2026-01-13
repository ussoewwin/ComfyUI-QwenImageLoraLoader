import logging
import os

# Version information - must be at module level for ComfyUI Manager
__version__ = "2.2.5"

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("=" * 40 + " ComfyUI-QwenImageLoraLoader Initialization " + "=" * 40)

NODE_CLASS_MAPPINGS = {}
QWEN_V2_NODES = {}
QWEN_V2_NAMES = {}
QWEN_V3_NODES = {}
QWEN_V3_NAMES = {}
ZIMAGETURBO_V4_NODES = {}
ZIMAGETURBO_V4_NAMES = {}

try:
    from .nodes.lora.qwenimage import NunchakuQwenImageLoraLoader, NunchakuQwenImageLoraStack
    from .nodes.lora.qwenimage_v2 import GENERATED_NODES as QWEN_V2_NODES, GENERATED_DISPLAY_NAMES as QWEN_V2_NAMES
    from .nodes.lora.qwenimage_v3 import GENERATED_NODES as QWEN_V3_NODES, GENERATED_DISPLAY_NAMES as QWEN_V3_NAMES
    # Z-Image-Turbo V2 is deprecated (unofficial loader only) - removed from registration to avoid confusion
    # from .nodes.lora.zimageturbo_v2 import GENERATED_NODES as ZIMAGETURBO_V2_NODES, GENERATED_DISPLAY_NAMES as ZIMAGETURBO_V2_NAMES
    # Z-Image-Turbo V3 is deprecated - removed from registration
    # from .nodes.lora.zimageturbo_v3 import GENERATED_NODES as ZIMAGETURBO_V3_NODES, GENERATED_DISPLAY_NAMES as ZIMAGETURBO_V3_NAMES
    from .nodes.lora.zimageturbo_v4 import GENERATED_NODES as ZIMAGETURBO_V4_NODES, GENERATED_DISPLAY_NAMES as ZIMAGETURBO_V4_NAMES

    # Add version to classes before creating NODE_CLASS_MAPPINGS
    NunchakuQwenImageLoraLoader.__version__ = __version__
    NunchakuQwenImageLoraStack.__version__ = __version__
    for node_class in QWEN_V2_NODES.values():
        node_class.__version__ = __version__
    for node_class in QWEN_V3_NODES.values():
        node_class.__version__ = __version__
    # Z-Image-Turbo V2 registration removed (unofficial loader only)
    # for node_class in ZIMAGETURBO_V2_NODES.values():
    #     node_class.__version__ = __version__
    # Z-Image-Turbo V3 registration removed
    # for node_class in ZIMAGETURBO_V3_NODES.values():
    #     node_class.__version__ = __version__
    for node_class in ZIMAGETURBO_V4_NODES.values():
        node_class.__version__ = __version__

    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraLoader"] = NunchakuQwenImageLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraStack"] = NunchakuQwenImageLoraStack
    NODE_CLASS_MAPPINGS.update(QWEN_V2_NODES)
    NODE_CLASS_MAPPINGS.update(QWEN_V3_NODES)
    # Z-Image-Turbo V2 registration removed (unofficial loader only)
    # NODE_CLASS_MAPPINGS.update(ZIMAGETURBO_V2_NODES)
    # Z-Image-Turbo V3 registration removed
    # NODE_CLASS_MAPPINGS.update(ZIMAGETURBO_V3_NODES)
    NODE_CLASS_MAPPINGS.update(ZIMAGETURBO_V4_NODES)
except ImportError:
    logger.exception("LoRA nodes import failed:")

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuQwenImageLoraLoader": "Nunchaku Qwen Image LoRA Loader",
    "NunchakuQwenImageLoraStack": "Nunchaku Qwen Image LoRA Stack (Legacy)",
    **QWEN_V2_NAMES,
    **QWEN_V3_NAMES,
    # Z-Image-Turbo V2 registration removed (unofficial loader only)
    # **ZIMAGETURBO_V2_NAMES,
    # Z-Image-Turbo V3 registration removed
    # **ZIMAGETURBO_V3_NAMES,
    **ZIMAGETURBO_V4_NAMES
}

# Register JavaScript extensions
WEB_DIRECTORY = "js"

# Make version available at module level for ComfyUI Manager
VERSION = __version__

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__version__", "VERSION"]
logger.info("=" * (80 + len(" ComfyUI-QwenImageLoraLoader Initialization ")))
