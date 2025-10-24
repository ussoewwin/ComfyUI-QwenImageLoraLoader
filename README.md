# ComfyUI-QwenImageLoraLoader

A ComfyUI custom node for loading and applying LoRA (Low-Rank Adaptation) to Nunchaku Qwen Image models.

**This project is based on the fork version of ComfyUI-nunchaku-qwen-lora-suport-standalone.**

## Source

This LoRA loader was extracted and modified from the fork version:
- **Original Fork**: [ComfyUI-nunchaku-qwen-lora-suport-standalone](https://github.com/ussoewwin/ComfyUI-nunchaku-qwen-lora-suport-standalone)
- **Extraction**: LoRA functionality was extracted from the full fork to create an independent custom node
- **Integration**: Modified to work with the official ComfyUI-nunchaku plugin

## Features

- **NunchakuQwenImageLoraLoader**: Load and apply single LoRA to Qwen Image models
- **NunchakuQwenImageLoraStack**: Apply multiple LoRAs with dynamic UI control
- **Dynamic VRAM Management**: Automatic CPU offloading based on available VRAM
- **LoRA Composition**: Efficient LoRA stacking and composition
- **ComfyUI Integration**: Seamless integration with ComfyUI workflows

## Installation

1. Clone this repository to your ComfyUI custom_nodes directory:
```bash
git clone https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader.git
```

2. Install dependencies:
```bash
pip install nunchaku
```

## Integration with ComfyUI-nunchaku

**IMPORTANT**: This node requires modification of the official ComfyUI-nunchaku plugin to function properly.

### Required Modification

You must modify the `__init__.py` file in your ComfyUI-nunchaku plugin to import this LoRA loader:

```python
try:
    # Import from the independent ComfyUI-QwenImageLoraLoader
    import sys
    import os
    qwen_lora_path = os.path.join(os.path.dirname(__file__), "..", "ComfyUI-QwenImageLoraLoader")
    if qwen_lora_path not in sys.path:
        sys.path.insert(0, qwen_lora_path)
    
    # Import directly from the file path
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwenimage", os.path.join(qwen_lora_path, "nodes", "lora", "qwenimage.py"))
    qwenimage_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwenimage_module)
    
    NunchakuQwenImageLoraLoader = qwenimage_module.NunchakuQwenImageLoraLoader
    NunchakuQwenImageLoraStack = qwenimage_module.NunchakuQwenImageLoraStack

    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraLoader"] = NunchakuQwenImageLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoraStack"] = NunchakuQwenImageLoraStack
    logger.info("Successfully imported Qwen Image LoRA loaders from ComfyUI-QwenImageLoraLoader")
except ImportError:
    logger.exception("Nodes `NunchakuQwenImageLoraLoader` and `NunchakuQwenImageLoraStack` import failed:")
```

### Why This Modification is Required

- The independent LoRA loader exists as a separate custom node
- The official ComfyUI-nunchaku plugin needs to reference the independent LoRA loader
- Standard `from ... import ...` fails due to different package structures
- `importlib.util` allows direct module loading from file paths

## Usage

### Single LoRA Loading
Use `NunchakuQwenImageLoraLoader` to load and apply a single LoRA to your Qwen Image model.

### Multiple LoRA Stacking
Use `NunchakuQwenImageLoraStack` to apply multiple LoRAs with dynamic UI control. The number of LoRA slots adjusts automatically based on the `lora_count` parameter.

## Requirements

- ComfyUI
- Nunchaku
- PyTorch
- Python 3.8+
- ComfyUI-nunchaku plugin (with required modification)

## Compatibility

This node is designed to work with:
- ComfyUI-nunchaku plugin (modified)
- Nunchaku Qwen Image models
- Standard ComfyUI workflows

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.