#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to append ComfyUI-QwenImageLoraLoader integration code to ComfyUI-nunchaku __init__.py
"""

import sys
import os

def append_integration_code(init_py_path):
    """Append the integration code to __init__.py file"""
    
    integration_code = '''

# ComfyUI-QwenImageLoraLoader Integration
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
'''

    try:
        with open(init_py_path, 'a', encoding='utf-8') as f:
            f.write(integration_code)
        print(f"Successfully appended integration code to: {init_py_path}")
        return True
    except Exception as e:
        print(f"Error appending integration code: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python append_integration.py <path_to_init.py>")
        sys.exit(1)
    
    init_py_path = sys.argv[1]
    if not os.path.exists(init_py_path):
        print(f"Error: File not found: {init_py_path}")
        sys.exit(1)
    
    success = append_integration_code(init_py_path)
    sys.exit(0 if success else 1)
