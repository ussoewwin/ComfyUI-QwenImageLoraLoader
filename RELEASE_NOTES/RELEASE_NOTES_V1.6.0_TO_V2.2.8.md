# Previous Releases (v1.6.0 to v2.2.8)

This document contains release notes for versions v1.6.0 through v2.2.8 of ComfyUI-QwenImageLoraLoader.

---

## v2.2.8

- **Fixed**: Resolved [Issue #44](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/44) – Fixed performance slowdown issue when loading unsupported LoRA formats (LoKR, LoHa, IA3, etc.). Unsupported formats are now detected early and detailed key inspection is skipped to prevent console slowdown. Retry logic is also skipped for unsupported formats to prevent duplicate logging
- **Technical Details**: See [v2.2.8 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.8) for complete explanation

---

## v2.2.7

- **Performance**: LoRA loading speed improvement - Implemented optimization by eliminating duplicate file reads. The first LoRA file loaded for debug logging is now cached and reused in actual processing, eliminating duplicate file I/O and deserialization operations
- **Technical Details**: See [v2.2.7 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.7) for complete explanation
- **Acknowledgments**: 本次性能改进参考了知乎（zhihu.com）上 Jimmy 先生发表的文章《【closerAI ComfyUI】nunchaku V1.1 支持 z-image + 支持 zimage-LoRA 的修复方案，短板一次性补全，nunchaku zimage 全面高速生图解决方案！》。虽然本项目采用的实现方案在具体设计与实现细节上可能与 Jimmy 先生的方案有所不同，但该文章促使我重新审视 LoRA 加载流程中的性能瓶颈，并意识到在速度优化方面仍然存在改进空间。借此机会，谨向 Jimmy 先生表示诚挚的感谢。

---

## v2.2.6

- **Fixed**: Resolved [Issue #43](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/43) – Fixed `TypeError: QwenTimestepProjEmbeddings.forward() takes 3 positional arguments but 4 were given` error by forcing `guidance=None` after converting to `additional_t_cond` to prevent ComfyUI-nunchaku's buggy code path from executing
- **Technical Details**: See [v2.2.6 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.6) for complete explanation

---

## v2.2.5

- **Fixed**: Repository recovery - All updates after v2.0.8 were completely broken, and recovery work has been performed to restore all functionality. Related to [Issue #39](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.5).
- **Recovery Details**: Restored all deleted files (images/, nodes/, wrappers/, nunchaku_code/, js/, md/, LICENSE, pyproject.toml) from local backups
- **Feature Verification**: Verified and restored all features from v2.0.8 through v2.2.4 (NextDiT support, AWQ skip logic, toggle buttons, LoRA format detection)
- **Fixed**: [Issue #42](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/42) - Fixed toggle button logic in `NunchakuQwenImageLoraStackV3` and `NunchakuZImageTurboLoraStackV3` nodes. Individual LoRAs can now be enabled even when `toggle_all` is `False`.
- **Technical Details**: See [v2.2.5 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.5) for complete explanation

---

## v2.2.4

- **Added**: AWQ modulation layer detection and skip logic - `img_mod.1` and `txt_mod.1` layers are detected and LoRA application is skipped by default to prevent noise. Can be overridden with `QWENIMAGE_LORA_APPLY_AWQ_MOD=1` environment variable.
- **Removed**: `NunchakuZImageTurboLoraStackV2` node registration has been removed from ComfyUI node list to avoid confusion when using official Nunchaku Z-Image loader. The node file remains in the repository but is no longer registered. Users of the official loader should use `NunchakuZImageTurboLoraStackV3` instead. ([Issue #37](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/37))
- **Technical Details**: See [v2.2.4 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.4) for complete explanation

---

## v2.2.3

- **Added**: Toggle buttons to enable/disable individual LoRA slots and all LoRAs at once. Resolved [Issue #12](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/12) and [Issue #36](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/36)
- ⚠️ **DEVELOPMENT STATUS**: These features are currently experimental implementations for the `NunchakuQwenImageLoraStackV3` and `NunchakuZImageTurboLoraStackV3` nodes. ComfyUI Nodes 2.0 environment only. With current technical capabilities, it is not possible to fully implement all requested features in JavaScript.
- **Technical Details**: See [v2.2.3 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.3) for complete explanation

---

## v2.2.2

- **Added**: Diffsynth ControlNet support for Nunchaku Z-ImageTurbo models
- **Technical Details**: See [v2.2.2 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.2) for complete explanation

---

## v2.2.0

- **Added**: NunchakuZImageTurboLoraStackV3 node – Z-Image-Turbo LoRA stacker with dynamic UI for official Nunchaku Z-Image loader
- **Technical Details**: See [v2.2.0 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.2.0) for complete explanation

---

## v2.1.1

- **Fixed**: ComfyUI v0.6.0+ compatibility – Migrated from `guidance` to `additional_t_cond` parameter in `_execute_model` method to support ComfyUI v0.6.0+ API changes ([PR #34](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/34))
- **Technical Details**: See [v2.1.1 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.1.1) for complete explanation

---

## v2.1.0

- **Fixed**: Resolved [Issue #33](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/33) – Fixed `AttributeError: 'NoneType' object has no attribute 'to'` by adding None checks to `to_safely` and `forward` methods in `ComfyQwenImageWrapper`
- **Technical Details**: See [v2.1.0 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.1.0) for complete explanation

---

## v2.0.2 - v2.0.8

### v2.0.8

- **Fixed**: Resolved [Issue #30](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/30) – Fixed `TypeError: got multiple values for argument 'guidance'` error by adding final cleanup of kwargs before calling model forward
- **Technical Details**: See [v2.0.8 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.0.8) for complete explanation

### v2.0.7

- **Fixed**: Enhanced [Issue #32](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/32) fix by adding exclusion processing in `forward` method in addition to `_execute_model` method to prevent duplicate argument errors
- **Technical Details**: See [v2.0.7 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.0.7) for complete explanation

### v2.0.6

- **Fixed**: Excluded `ref_latents`, `transformer_options`, and `attention_mask` from kwargs to prevent duplicate argument errors
- **Technical Details**: See [v2.0.6 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.0.6) for complete explanation

### v2.0.5

- **Fixed**: Resolved [Issue #32](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/32) – Fixed `TypeError: got multiple values for argument 'guidance'` error by passing guidance as positional argument to match QwenImageTransformer2DModel.forward signature
- **Technical Details**: See [v2.0.5 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.0.5) for complete explanation

### v2.0.4

- **Fixed**: Resolved [Issue #32](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/32) – Fixed `TypeError: got multiple values for argument 'guidance'` error by removing guidance from transformer_options
- **Technical Details**: See [v2.0.4 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.0.4) for complete explanation

### v2.0.3

- **Fixed**: Resolved [Issue #31](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/31) – Fixed nodes not appearing when `comfy.ldm.lumina.controlnet` module is unavailable
- **Technical Details**: See [v2.0.3 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.0.3) for complete explanation

### v2.0.2

- **Fixed**: Resolved [Issue #30](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/30) and [Issue #32](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/32) – Fixed `TypeError: got multiple values for argument 'guidance'` error when using LoRA with KSampler
- **Technical Details**: See [v2.0.2 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.0.2) for complete explanation

---

## v2.0

- **MAJOR UPDATE**: Added diffsynth ControlNet support for Nunchaku Qwen Image models
- **New Node**: `NunchakuQI&ZITDiffsynthControlnet` - Enables diffsynth ControlNet to work with Nunchaku quantized Qwen Image models, Z Image Turbo BF16.safetensors, and Nunchaku Z Image Turbo models
- **Features**:
  - Full diffsynth ControlNet functionality for Nunchaku Qwen Image models
  - Automatic patch registration and application
- **Technical Details**: See [v2.0 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.0) for complete explanation

---

## v1.72

- **Fixed**: Resolved compatibility issue with kjai node updates – Added default value `"disable"` for `cpu_offload` parameter in LoRA loader methods ([PR #28](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/28))
- **Reported by**: [@enternalsaga](https://github.com/enternalsaga) ([PR #28](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/28))
- **Technical Details**: See [v1.72 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v1.72) for complete explanation

---

## v1.71

- **Fixed**: Resolved [Issue #27](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/27) – Fixed indentation error on line 882 in `lora_qwen.py` causing `SyntaxError: expected an indented block after 'else' statement` (reported by [@youyin400c-cpu](https://github.com/youyin400c-cpu))
- **Attempted Fix**: Addressed [Issue #25](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/25) – `AttributeError: 'NunchakuModelPatcher' object has no attribute 'pinned'` and deepcopy errors with `model_config`
- **Reported by**: [@LacklusterOpsec](https://github.com/LacklusterOpsec) ([Issue #25](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/25))
- **Current Status**: ⚠️ **This error does not occur in our stable ComfyUI environment** - The fix was implemented based on the reported issue, but we cannot guarantee it will completely resolve the issue as we cannot reproduce it in our environment. If you encounter this error, please report with your ComfyUI version and environment details.
- **Technical Details**: See [v1.71 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v1.71) for complete explanation

---

## v1.70

- **Added**: V2 loader with ComfyUI Nodes 2.0 (Beta) support
- **New Node**: `NunchakuQwenImageLoraStackV2` - V2 loader node added
- **Fixed**: Resolved [Issue #9](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/9) – The 10th LoRA control row no longer displays when `lora_count` is set to less than 10. Dynamic UI now correctly hides unused LoRA slots and adjusts node height automatically
- **Features**:
  - Full compatibility with ComfyUI Nodes 2.0 (Beta)
  - Complete feature parity with V1 implementation
  - Dynamic UI for adjusting slot count
  - Automatic node height adjustment
- **Technical Details**: See [v1.70 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v1.70) for complete explanation

---

## v1.63

- **Fixed**: Addressed [Issue #21](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/21) – User-configurable CPU offload setting
- **Problem**: CPU offload setting was hardcoded to `"auto"`, causing unnecessary slowdowns when VRAM was sufficient
- **Solution**: Added `cpu_offload` parameter to `INPUT_TYPES` allowing users to select from `["auto", "enable", "disable"]` with default `"disable"` for performance
- **Technical Details**: See [v1.63 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v1.63) for complete explanation

<img src="images/v1.63_cpu_offload_setting.png" alt="v1.63 CPU Offload Setting" width="50%">

---

## v1.62

- **Attempted Fix**: Addressed [Issue #14](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/14) – Multi-stage workflow cache not resetting when LoRAs change
- **Problem**: Cache was not being reset when switching between different LoRA sets in multi-stage workflows, causing incorrect results
- **Solution Attempted**: Cache invalidation logic was added to reset cache when LoRAs change
- **Current Status**: ⚠️ **Issue is still not fully resolved** - The fix was implemented but the problem persists in some multi-stage workflow scenarios
- **Technical Details**: See [v1.62 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v1.62) for complete explanation

---

## v1.60

- **MAJOR UPDATE**: Removed ComfyUI-nunchaku integration requirement - now a fully independent custom node
- **Simplified Installation**: No batch scripts or manual file editing needed - just `git clone` and restart
- **Cleaner Architecture**: Node registration happens automatically via ComfyUI's built-in mechanism
- **Backward Compatible**: All existing LoRA files and workflows continue to work
- **Technical Details**: See [v1.60 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v1.60) for complete explanation
- **Full release notes**: https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v1.60

---

## Related Documents

For release notes from v1.0.0 to v1.57, please see [RELEASE_NOTES_V1.0.0_TO_V1.57.md](RELEASE_NOTES_V1.0.0_TO_V1.57.md).
