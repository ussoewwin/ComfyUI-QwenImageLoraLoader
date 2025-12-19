import logging
import os

# Version information - must be at module level for ComfyUI Manager
__version__ = "1.72"

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
    from .nodes.controlnet import NunchakuQwenImageDiffsynthControlnet
    from .nodes.ksampler import NunchakuKSampler

    # Add version to classes before creating NODE_CLASS_MAPPINGS
    NunchakuQwenImageLoraLoader.__version__ = __version__
    NunchakuQwenImageLoraStack.__version__ = __version__
    NunchakuQwenImageDiffsynthControlnet.__version__ = __version__
    NunchakuKSampler.__version__ = __version__
    for node_class in QWEN_V2_NODES.values():
        node_class.__version__ = __version__

    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraLoader"] = NunchakuQwenImageLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraStack"] = NunchakuQwenImageLoraStack
    NODE_CLASS_MAPPINGS["NunchakuQwenImageDiffsynthControlnet"] = NunchakuQwenImageDiffsynthControlnet
    NODE_CLASS_MAPPINGS["NunchakuKSampler"] = NunchakuKSampler
    NODE_CLASS_MAPPINGS.update(QWEN_V2_NODES)
except ImportError:
    logger.exception("Nodes import failed:")

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuQwenImageLoraLoader": "Nunchaku Qwen Image LoRA Loader",
    "NunchakuQwenImageLoraStack": "Nunchaku Qwen Image LoRA Stack (Legacy)",
    "NunchakuQwenImageDiffsynthControlnet": "Nunchaku Qwen Image Diffsynth Controlnet",
    "NunchakuKSampler": "Nunchaku KSampler",
    **QWEN_V2_NAMES
}

# Register JavaScript extensions
WEB_DIRECTORY = "js"

# Make version available at module level for ComfyUI Manager
VERSION = __version__

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__version__", "VERSION"]
logger.info("=" * (80 + len(" ComfyUI-QwenImageLoraLoader Initialization ")))
