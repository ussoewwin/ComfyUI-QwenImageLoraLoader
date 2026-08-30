# -*- coding: utf-8 -*-
"""
Patch ComfyUI SAM3Detector._detect to scalp the segmentation head inputs
together with the encoder.

SAM3 (non-multiplex) has 4 FPN levels but scalp=1 keeps only 3 for the
encoder. The stock code captures seg_features = features BEFORE the scalp and
passes all 4 levels to SegmentationHead. The head then replaces the smallest
36px level with a spatially-wrong crop of encoder_visual (built from only
1296 of the 5184 image tokens), which corrupts the pixel decoder and biases
every mask logit negative, producing empty / near-black masks.

Fix: pre-scalp the inputs and temporarily set self.scalp = 0 while calling
the original _detect, so seg_features ends up with the same 3 levels as the
encoder.

- Inert when the upstream fix is present (Comfy-Org/ComfyUI PR #15979):
  the wrapper pre-scales and the (fixed) original then keeps seg_features as-is.
- No-op for multiplex models (SAM3.1, scalp=0): the wrapper falls straight
  through to the original.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_TAG = "_qwen_lora_loader_sam3_seg_features_scalp_patch"
_ORIGINAL_DETECT = None


def apply_sam3_seg_features_scalp_patch() -> bool:
    """Install the SAM3 segmentation-head scalp wrapper on SAM3Detector._detect.

    Returns True when the wrapper is installed (or already installed); False when
    SAM3 is not available in the installed ComfyUI (wrapper stays absent).
    """
    global _ORIGINAL_DETECT
    if _ORIGINAL_DETECT is not None:
        return True

    try:
        import comfy.ldm.sam3.detector as detector_mod
    except Exception:
        logger.debug("SAM3 seg_features scalp patch skipped: comfy.ldm.sam3.detector not available")
        return False

    try:
        orig = detector_mod.SAM3Detector._detect
    except Exception as e:
        logger.debug("SAM3 seg_features scalp patch skipped: %s", e)
        return False

    if getattr(orig, _PATCH_TAG, False):
        _ORIGINAL_DETECT = orig
        return True

    def _detect_with_scalped_seg_features(self, features, positions, text_embeddings=None,
                                          text_mask=None, points=None, boxes=None):
        # SAM3 (scalp=1) keeps 3 of its 4 FPN levels for the encoder; the stock
        # code feeds all 4 to SegmentationHead, whose smallest level is replaced
        # by a spatially-wrong crop of encoder_visual (empty masks). Pre-scalp and
        # run the original with scalp=0 so seg_features matches the encoder.
        if self.scalp > 0:
            features = features[:-self.scalp]
            positions = positions[:-self.scalp]
            old_scalp = self.scalp
            self.scalp = 0
            try:
                return orig(self, features, positions, text_embeddings, text_mask, points, boxes)
            finally:
                self.scalp = old_scalp
        return orig(self, features, positions, text_embeddings, text_mask, points, boxes)

    _detect_with_scalped_seg_features.__name__ = orig.__name__
    _detect_with_scalped_seg_features.__doc__ = orig.__doc__
    setattr(_detect_with_scalped_seg_features, _PATCH_TAG, True)
    detector_mod.SAM3Detector._detect = _detect_with_scalped_seg_features
    _ORIGINAL_DETECT = orig
    logger.info("Applied SAM3 seg_features scalp patch (SAM3 empty-mask fix, PR #15979 parity)")
    return True
