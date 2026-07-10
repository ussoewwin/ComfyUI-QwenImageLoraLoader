"""
lora_cache.py — Pre-compiled LoRA structural cache utilities.

Provides save/load helpers for caching the result of the expensive
classify → fuse pipeline in compose_loras_v2.  The cache stores the
raw fused (A, B, alpha) tensor pairs *before* strength scaling so that
the strength slider still works correctly at inference time.

Cache layout on disk:
  ComfyUI/models/SVDQLora/
    <lora_stem>_precompiled.safetensors   ← fused tensors
    <lora_stem>_precompiled.safetensors.meta.json ← mtime sidecar

Tensor key format inside the .safetensors file:
  "<module_key>__A"           → fused A tensor  [rank, in_features]
  "<module_key>__B"           → fused B tensor  [out_features, rank]  (NOT strength-scaled)
  "<module_key>__alpha"       → alpha scalar as float32 1-D tensor (optional, omitted if None)

Sidecar JSON format:
  {
    "source_mtime": <float>,   # os.path.getmtime() of the original .safetensors
    "source_path":  <str>      # absolute path of original LoRA (informational)
  }
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Separator used to encode module_key and tensor role inside a flat safetensors dict.
# Double-underscore is safe because module keys use dot-notation (e.g. "transformer_blocks.0.attn.to_qkv").
_SEP = "__"

# Roles stored per module entry
_ROLE_A = "A"
_ROLE_B = "B"
_ROLE_ALPHA = "alpha"


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def get_cache_dir(comfy_base_dir: str) -> Path:
    """
    Return (and create if needed) the dedicated SVDQLora cache directory.

    Args:
        comfy_base_dir: Root of the ComfyUI installation, e.g. ``"C:/ComfyUI"``.

    Returns:
        ``Path`` pointing to ``<comfy_base_dir>/models/SVDQLora/``.
    """
    cache_dir = Path(comfy_base_dir) / "models" / "SVDQLora"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_path(lora_path: str | Path, cache_dir: Path) -> Path:
    """
    Derive the cache .safetensors path for a given LoRA source file.

    The cache filename includes a short hash of the LoRA's parent directory so
    that two LoRAs with identical filenames in different subdirectories (e.g.
    ``Qwen-Edit/v1/lora.safetensors`` vs ``Qwen-Edit/v2/lora.safetensors``)
    never collide to the same cache entry.

    Args:
        lora_path:  Full path to the original LoRA file.
        cache_dir:  Directory returned by :func:`get_cache_dir`.

    Returns:
        ``Path`` like ``<cache_dir>/<stem>_<dir_hash8>_precompiled.safetensors``.
    """
    lora_path = Path(lora_path)
    stem = lora_path.stem
    # Hash the parent directory path so same-named LoRAs in different dirs
    # produce distinct cache keys.
    parent_hash = hashlib.sha256(str(lora_path.parent).encode()).hexdigest()[:8]
    return cache_dir / f"{stem}_{parent_hash}_precompiled.safetensors"


def _get_meta_path(cache_path: Path) -> Path:
    """Return the sidecar JSON path for a given cache file."""
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_cache_valid(lora_path: str | Path, cache_path: Path) -> bool:
    """
    Return ``True`` iff a usable, up-to-date cache exists for *lora_path*.

    Validity criteria:
    1. ``cache_path`` exists on disk.
    2. The mtime sidecar JSON exists.
    3. The recorded ``source_mtime`` matches the current mtime of *lora_path*.

    A missing or unreadable sidecar is treated as invalid (forces re-fuse).

    Args:
        lora_path:   Full path to the original LoRA file.
        cache_path:  Path as returned by :func:`get_cache_path`.
    """
    if not cache_path.is_file():
        return False

    meta_path = _get_meta_path(cache_path)
    if not meta_path.is_file():
        logger.debug(f"[CACHE] No sidecar for {cache_path.name}, treating as invalid.")
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        recorded_mtime: float = meta["source_mtime"]
    except (KeyError, json.JSONDecodeError, OSError) as exc:
        logger.warning(f"[CACHE] Failed to read sidecar {meta_path.name}: {exc}. Treating as invalid.")
        return False

    try:
        current_mtime = os.path.getmtime(str(lora_path))
    except OSError as exc:
        logger.warning(f"[CACHE] Cannot stat source LoRA {lora_path}: {exc}. Treating cache as invalid.")
        return False

    if abs(current_mtime - recorded_mtime) > 1e-3:
        logger.info(
            f"[CACHE] Stale: source mtime changed for {Path(lora_path).name}. "
            f"Recorded={recorded_mtime:.3f}, current={current_mtime:.3f}. "
            "Will re-fuse and overwrite cache."
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_precompiled(
    processed_groups: Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    lora_path: str | Path,
    cache_path: Path,
) -> None:
    """
    Serialise the fused ``processed_groups`` dict to *cache_path*.

    ``processed_groups`` has the shape produced by the classify+fuse stage of
    ``compose_loras_v2``:  ``{ module_key: (A, B, alpha) }``.

    All tensors are moved to CPU before saving (the cache is always CPU-side;
    device placement happens later when ``_apply_lora_to_module`` runs).

    Args:
        processed_groups: Fused tensor pairs from the classify+fuse stage.
        lora_path:        Full path to the original LoRA file (used for mtime).
        cache_path:       Destination .safetensors file.
    """
    # Lazy import so safetensors is only required when actually saving.
    try:
        from safetensors.torch import save_file as st_save_file
    except ImportError as exc:
        logger.error(f"[CACHE] Cannot save precompiled cache: safetensors not available. {exc}")
        return

    if not processed_groups:
        logger.warning("[CACHE] processed_groups is empty — nothing to save.")
        return

    flat: Dict[str, torch.Tensor] = {}

    for module_key, (A, B, alpha) in processed_groups.items():
        if A is None or B is None:
            logger.debug(f"[CACHE] Skipping {module_key}: A or B is None.")
            continue

        # Validate tensor shapes before saving
        if A.ndim != 2 or B.ndim != 2:
            logger.warning(
                f"[CACHE] Skipping {module_key}: expected 2-D tensors, "
                f"got A.ndim={A.ndim}, B.ndim={B.ndim}."
            )
            continue

        # Move to CPU; safetensors does not accept CUDA tensors.
        flat[f"{module_key}{_SEP}{_ROLE_A}"] = A.detach().cpu().contiguous()
        flat[f"{module_key}{_SEP}{_ROLE_B}"] = B.detach().cpu().contiguous()

        if alpha is not None:
            # Normalise to a 1-D float32 scalar tensor for portability.
            if isinstance(alpha, torch.Tensor):
                flat[f"{module_key}{_SEP}{_ROLE_ALPHA}"] = alpha.detach().cpu().float().reshape(1)
            else:
                flat[f"{module_key}{_SEP}{_ROLE_ALPHA}"] = torch.tensor([float(alpha)], dtype=torch.float32)

    if not flat:
        logger.warning("[CACHE] No valid tensors to write after filtering — cache not saved.")
        return

    try:
        # Write to a temp file first; atomically rename to the final path.
        # This prevents a partially-written file from being mistaken as valid
        # if ComfyUI is interrupted mid-write (which would corrupt the cache).
        tmp_path = cache_path.with_suffix(".safetensors.tmp")
        st_save_file(flat, str(tmp_path))
        os.replace(str(tmp_path), str(cache_path))
        logger.info(f"[CACHE SAVE] Written {len(processed_groups)} module entries → {cache_path}")
    except Exception as exc:
        logger.error(f"[CACHE] Failed to write {cache_path}: {exc}")
        # Clean up temp file if it was created
        try:
            tmp_path = cache_path.with_suffix(".safetensors.tmp")
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return

    # Write mtime sidecar
    meta_path = _get_meta_path(cache_path)
    try:
        source_mtime = os.path.getmtime(str(lora_path))
        meta = {
            "source_mtime": source_mtime,
            "source_path": str(Path(lora_path).resolve()),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.debug(f"[CACHE] Sidecar written: {meta_path.name}")
    except OSError as exc:
        logger.warning(
            f"[CACHE] Could not write mtime sidecar {meta_path.name}: {exc}. "
            "Cache will be treated as invalid on next run."
        )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_precompiled(
    cache_path: Path,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """
    Load a precompiled cache file and reconstruct the ``processed_groups`` dict.

    Returns:
        ``{ module_key: (A, B, alpha_or_None) }`` — identical structure to
        what ``compose_loras_v2`` produces after the classify+fuse stage.
        Returns an empty dict on any error so the caller can fall back to
        a full re-fuse gracefully.
    """
    try:
        from safetensors.torch import load_file as st_load_file
    except ImportError as exc:
        logger.error(f"[CACHE] Cannot load precompiled cache: safetensors not available. {exc}")
        return {}

    try:
        flat: Dict[str, torch.Tensor] = st_load_file(str(cache_path), device="cpu")
    except Exception as exc:
        logger.error(f"[CACHE] Failed to load {cache_path}: {exc}")
        return {}

    # Reconstruct { module_key: { "A": T, "B": T, "alpha": T|None } }
    raw: Dict[str, Dict[str, torch.Tensor]] = {}
    for flat_key, tensor in flat.items():
        # flat_key format:  "<module_key>__<role>"
        # module_key itself may contain dots but never "__"
        sep_idx = flat_key.rfind(_SEP)
        if sep_idx == -1:
            logger.warning(f"[CACHE] Unrecognised flat key format: '{flat_key}' — skipping.")
            continue
        module_key = flat_key[:sep_idx]
        role = flat_key[sep_idx + len(_SEP):]
        raw.setdefault(module_key, {})[role] = tensor

    processed_groups: Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = {}
    for module_key, parts in raw.items():
        A = parts.get(_ROLE_A)
        B = parts.get(_ROLE_B)
        alpha_tensor = parts.get(_ROLE_ALPHA)  # 1-D float32 scalar tensor or None

        if A is None or B is None:
            logger.warning(f"[CACHE] Incomplete entry for '{module_key}' (missing A or B) — skipping.")
            continue

        # Convert alpha back to the same form compose_loras_v2 expects:
        # a torch.Tensor scalar (not a Python float).
        alpha: Optional[torch.Tensor] = None
        if alpha_tensor is not None:
            alpha = alpha_tensor  # keep as tensor; compose_loras_v2 calls .item() on it

        processed_groups[module_key] = (A, B, alpha)

    logger.info(f"[CACHE LOAD] Loaded {len(processed_groups)} module entries ← {cache_path.name}")
    return processed_groups
