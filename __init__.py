import logging
import os

# Version information - must be at module level for ComfyUI Manager
__version__ = "2.1.1"

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("=" * 40 + " ComfyUI-QwenImageLoraLoader Initialization " + "=" * 40)

NODE_CLASS_MAPPINGS = {}
QWEN_V2_NODES = {}
QWEN_V2_NAMES = {}

try:
    from .nodes.lora.qwenimage import NunchakuQwenImageLoraLoader, NunchakuQwenImageLoraStack
    from .nodes.lora.qwenimage_v2 import GENERATED_NODES as QWEN_V2_NODES, GENERATED_DISPLAY_NAMES as QWEN_V2_NAMES
    from .nodes.lora.qwenimage_v3 import GENERATED_NODES as QWEN_V3_NODES, GENERATED_DISPLAY_NAMES as QWEN_V3_NAMES
    # Z-Image-Turbo V2 is deprecated (unofficial loader only) - removed from registration to avoid confusion
    # from .nodes.lora.zimageturbo_v2 import GENERATED_NODES as ZIMAGETURBO_V2_NODES, GENERATED_DISPLAY_NAMES as ZIMAGETURBO_V2_NAMES
    from .nodes.lora.zimageturbo_v3 import GENERATED_NODES as ZIMAGETURBO_V3_NODES, GENERATED_DISPLAY_NAMES as ZIMAGETURBO_V3_NAMES

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
    for node_class in ZIMAGETURBO_V3_NODES.values():
        node_class.__version__ = __version__

    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraLoader"] = NunchakuQwenImageLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraStack"] = NunchakuQwenImageLoraStack
    NODE_CLASS_MAPPINGS.update(QWEN_V2_NODES)
    NODE_CLASS_MAPPINGS.update(QWEN_V3_NODES)
    # Z-Image-Turbo V2 registration removed (unofficial loader only)
    # NODE_CLASS_MAPPINGS.update(ZIMAGETURBO_V2_NODES)
    NODE_CLASS_MAPPINGS.update(ZIMAGETURBO_V3_NODES)
except ImportError:
    logger.exception("LoRA nodes import failed:")

# Try to import ControlNet node separately - it may fail if comfy.ldm.lumina.controlnet is not available
try:
    from .nodes.controlnet import NunchakuQwenImageDiffsynthControlnet
    NunchakuQwenImageDiffsynthControlnet.__version__ = __version__
    NODE_CLASS_MAPPINGS["NunchakuQwenImageDiffsynthControlnet"] = NunchakuQwenImageDiffsynthControlnet
    logger.info("✅ ControlNet node loaded successfully")
except ImportError:
    logger.warning("⚠️ ControlNet node not available (comfy.ldm.lumina.controlnet not found). LoRA nodes will still work.")
except Exception as e:
    logger.warning(f"⚠️ ControlNet node failed to load: {e}. LoRA nodes will still work.")

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuQwenImageLoraLoader": "Nunchaku Qwen Image LoRA Loader",
    "NunchakuQwenImageLoraStack": "Nunchaku Qwen Image LoRA Stack (Legacy)",
    **QWEN_V2_NAMES,
    **QWEN_V3_NAMES,
    # Z-Image-Turbo V2 registration removed (unofficial loader only)
    # **ZIMAGETURBO_V2_NAMES,
    **ZIMAGETURBO_V3_NAMES
}

# Add ControlNet display name only if the node was successfully loaded
if "NunchakuQwenImageDiffsynthControlnet" in NODE_CLASS_MAPPINGS:
    NODE_DISPLAY_NAME_MAPPINGS["NunchakuQwenImageDiffsynthControlnet"] = "NunchakuQI&ZITDiffsynthControlnet"

# Register JavaScript extensions
WEB_DIRECTORY = "js"

# Make version available at module level for ComfyUI Manager
VERSION = __version__

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__version__", "VERSION"]
logger.info("=" * (80 + len(" ComfyUI-QwenImageLoraLoader Initialization ")))
