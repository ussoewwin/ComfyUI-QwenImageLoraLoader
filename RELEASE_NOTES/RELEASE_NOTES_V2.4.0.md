# Release Notes v2.4.0: Nunchaku Qwen Image LoRA Stack V1

**Full technical documentation**: [md/V2.4.0_V1_RENAME_COMPLETE_EXPLANATION.md](../md/V2.4.0_V1_RENAME_COMPLETE_EXPLANATION.md)

---

## Summary

- **Added**: Nunchaku Qwen Image LoRA Stack V1 (rgthree-style UI) — merged from [PR #49](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/49)
- **Renamed**: V4 → V1 (node was introduced as V4 in PR; renamed to V1 as the first rgthree-style variant)
- **Breaking change**: Workflows using `NunchakuQwenImageLoraStackV4` must switch to `Nunchaku Qwen Image LoRA Stack V1` and reconfigure

---

## Files Changed

| Item | V4 (Before) | V1 (After) |
|------|-------------|------------|
| Python | `qwenimage_v4.py` | `qwenimage_v1.py` |
| JS | `z_qwen_lora_dynamic_v4.js` | `z_qwen_lora_dynamic_v1.js` |
| Node ID | `NunchakuQwenImageLoraStackV4` | `NunchakuQwenImageLoraStackV1` |

---

## References

- [PR #49](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/49)
- [Power Lora Loader (rgthree-comfy)](https://github.com/rgthree/rgthree-comfy)
