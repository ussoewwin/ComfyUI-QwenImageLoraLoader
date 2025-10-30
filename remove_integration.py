# -*- coding: utf-8 -*-
"""Remove the ComfyUI-QwenImageLoraLoader integration block from ComfyUI-nunchaku."""

import io
import os
import re
import sys
from pathlib import Path


BEGIN_MARKER = "# BEGIN ComfyUI-QwenImageLoraLoader Integration"
END_MARKER = "# END ComfyUI-QwenImageLoraLoader Integration"


def _find_comfyui_root(script_dir: Path, explicit_root: str | None) -> Path | None:
    candidates = []
    if explicit_root:
        candidates.append(Path(explicit_root))

    env_root = os.environ.get("COMFYUI_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    resolved = script_dir.resolve()
    candidates.append(resolved)
    candidates.extend(resolved.parents)

    seen: set[Path] = set()
    for root in candidates:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        init_path = root / "custom_nodes" / "ComfyUI-nunchaku" / "__init__.py"
        if init_path.is_file():
            return root
    return None


def _remove_legacy_block(text: str) -> tuple[str, int]:
    """Remove old-style integration block without markers."""
    old_comment = "# ComfyUI-QwenImageLoraLoader Integration"
    removed = 0
    
    start = text.find(old_comment)
    if start == -1:
        return text, 0
    
    # Find the end of the try-except block by looking for the last logger.exception
    # Simple pattern: find from the comment to the end of the except block
    pattern = r'# ComfyUI-QwenImageLoraLoader Integration.*?except ImportError:.*?logger\.exception\(.*?\)'
    
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Remove the entire match plus any trailing whitespace
        start_pos = match.start()
        end_pos = match.end()
        
        # Trim leading whitespace from the block
        while start_pos > 0 and text[start_pos - 1] in "\r\n\t ":
            start_pos -= 1
        
        # Remove the block
        text = text[:start_pos].rstrip() + "\n\n" + text[end_pos:].lstrip()
        removed = 1
    
    return text, removed


def _remove_all_blocks(text: str) -> tuple[str, int]:
    # First, remove new-style blocks with markers
    removed = 0
    start = text.find(BEGIN_MARKER)
    while start != -1:
        stop = text.find(END_MARKER, start)
        if stop == -1:
            break
        stop += len(END_MARKER)
        while stop < len(text) and text[stop] in "\r\n":
            stop += 1
        text = (text[:start].rstrip() + "\n\n" + text[stop:].lstrip())
        removed += 1
        start = text.find(BEGIN_MARKER)
    
    # Then, remove old-style blocks without markers
    text, legacy_removed = _remove_legacy_block(text)
    removed += legacy_removed
    
    return text, removed


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    explicit_root = sys.argv[1] if len(sys.argv) > 1 else None

    comfyui_root = _find_comfyui_root(script_dir, explicit_root)
    if comfyui_root is None:
        print("ComfyUI root containing ComfyUI-nunchaku not found. Specify it as an argument if necessary.")
        return 1

    init_path = comfyui_root / "custom_nodes" / "ComfyUI-nunchaku" / "__init__.py"

    with io.open(init_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    new_content, removed = _remove_all_blocks(content)
    if removed == 0:
        print("Integration block not found. Nothing to remove.")
        return 0

    with io.open(init_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(new_content.rstrip() + "\n")

    print(f"Integration block removed ({removed} block(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
