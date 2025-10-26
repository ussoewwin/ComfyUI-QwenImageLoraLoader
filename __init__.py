import logging
import os

# Version information
__version__ = "1.0.0"

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("=" * 40 + " ComfyUI-QwenImageLoraLoader Initialization " + "=" * 40)

NODE_CLASS_MAPPINGS = {}

try:
    from .nodes.lora.qwenimage import NunchakuQwenImageLoraLoader, NunchakuQwenImageLoraStack

    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraLoader"] = NunchakuQwenImageLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraStack"] = NunchakuQwenImageLoraStack
except ImportError:
    logger.exception("Nodes `NunchakuQwenImageLoraLoader` and `NunchakuQwenImageLoraStack` import failed:")

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}

# Register JavaScript extensions
WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__version__"]
logger.info("=" * (80 + len(" ComfyUI-QwenImageLoraLoader Initialization ")))