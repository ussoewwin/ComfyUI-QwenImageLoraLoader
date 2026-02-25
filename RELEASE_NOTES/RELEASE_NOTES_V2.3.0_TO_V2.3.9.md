# Release Notes (v2.3.0 to v2.3.9)

This document contains release notes for versions v2.3.0 through v2.3.9 of ComfyUI-QwenImageLoraLoader.

---

## v2.3.9

- **Added**: Key Diffusion (key analysis logging) - Added detailed key analysis logging to all Qwen Image LoRA nodes (Loader, Legacy Stack, V2 Stack, V3 Stack). When enabled via `nunchaku_log=1` environment variable, displays key mapping information (`Key: <original key> -> Mapped to: <mapped target> (Group: <group>)`) for debugging and verification purposes. This feature matches the functionality already available in Z-Image Turbo LoRA nodes. **Note**: Logs are muted by default and only displayed when `nunchaku_log=1` is set.
- **Technical Details**: See [v2.3.9 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.3.9) for complete explanation

---

## v2.3.8

- **Fixed**: PEFT format LoRA detection - Fixed issue where LoRA files created with Hugging Face PEFT library (`.lora_A.default.weight` format) were incorrectly detected as "unsupported" and skipped. Changed pattern matching to support PEFT format while maintaining backward compatibility.
- **Added**: Warning log for skipped LoRA weights - Added logging to display which LoRA files had weights automatically skipped for AWQ modulation layers.
- **Merged**: [PR #48](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/48) - feat(lora_qwen): improve format detection and add safety skip logs (proposed by [avan06](https://github.com/avan06))
- **Related Issues**: [Issue #44](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/44) - Module resolution failures and unsupported LoRA format issues
- **Technical Details**: See [v2.3.8 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.3.8) for complete explanation

---

## v2.3.7

- **Updated**: V3 nodes - AWQ modulation layer LoRA application is now **always enabled** (no switch needed). V3 nodes (`NunchakuQwenImageLoraStackV3`) automatically apply LoRA to AWQ quantized modulation layers (`img_mod.1` / `txt_mod.1`) without requiring the `apply_awq_mod` toggle that was needed in V2 nodes.
- **Reason**: After v2.3.6 implementation, no noise issues have been reported, so we determined it is safe to always enable AWQ modulation layer LoRA application in V3 nodes.
- **Note**: The `apply_awq_mod` switch in V2 nodes is **currently retained** for backward compatibility and user preference. V2 users can still manually control AWQ modulation layer LoRA application via the toggle.
- **Impact**: V3 users no longer need to manually enable the AWQ modulation layer toggle. The Manual Planar Injection fix is automatically applied for all LoRAs in V3 nodes.
- **Technical Details**: See [v2.3.7 Release Notes](RELEASE_NOTES/RELEASE_NOTES_V2.3.7.md) for complete explanation

---

## v2.3.6

- **Fixed**: AWQ modulation layer LoRA application - Implemented Runtime Monkey Patch with Manual Planar Injection to fix noise issues when applying LoRA to AWQ quantized modulation layers (`img_mod.1` / `txt_mod.1`). The fix uses a runtime patch to inject LoRA weights directly into the transformer block forward pass, bypassing layout/transpose issues.
- **Impact**: This fix enables all standard format LoRA keys that were previously excluded from AWQ modulation layers to be fully applied. Previously, these layers were skipped to prevent noise, but now they work correctly with the Manual Planar Injection approach.
- ⚠️ **Warning**: **This is an experimental feature currently implemented only in V2 nodes.** If no issues are found, this feature will be applied to V1 and V3 nodes as well.
- **Related Issues**: [Issue #44](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/44) - Module resolution failures and unsupported LoRA format issues
- **Technical Details**: See [v2.3.6 Release Notes](RELEASE_NOTES/RELEASE_NOTES_V2.3.6.md) for complete explanation

---

## v2.3.5

- **Restored**: `NunchakuZImageTurboLoraStackV2` node registration has been restored
- ⚠️ **Important**: This node is **only compatible with the unofficial Nunchaku Z-Image-Turbo DiT Loader** provided by [ComfyUI-nunchaku-unofficial-loader](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader)
- ⚠️ **Not Compatible**: This node is **not compatible** with the official Nunchaku Z-Image-Turbo DiT Loader from ComfyUI-Nunchaku

---

## v2.3.1

- **Fixed**: Resolved [Issue #46](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/46) – Fixed `NameError: name 'ZIMAGETURBO_V4_NAMES' is not defined` error that occurred when `zimageturbo_v4` module import failed. Initialized `ZIMAGETURBO_V4_NAMES`, `ZIMAGETURBO_V4_NODES`, `QWEN_V3_NAMES`, and `QWEN_V3_NODES` before the import block to prevent NameError on import failure.
- **Fixed**: Resolved [Issue #47](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/47) – Added missing v4 files to Git repository (`nodes/lora/zimageturbo_v4.py`, `js/zimageturbo_lora_dynamic_v4.js`). These files were created but not committed to Git, causing `ModuleNotFoundError` in previous releases. The files are now included in the repository.

---

## v2.3.0

- **Removed**: Z-ImageTurbo Loader v3 registration - Removed from ComfyUI node registration
- **Updated**: Z-ImageTurbo Loader v4 - Conforms to standard ComfyUI LoRA loader format (CLIP input/output, no CPU offload parameter) while maintaining perfect mapping functionality
- **Removed**: Diffsynth ControlNet support - Removed `NunchakuQI&ZITDiffsynthControlnet` node registration and all related documentation. ComfyUI-Nunchaku now has native support for ZIT (Z-Image-Turbo) Diffsynth ControlNet, so this custom node is no longer needed.
- **Technical Details**: See [v2.3.0 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.3.0) for complete explanation
