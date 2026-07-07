### v2.5.0 (latest)
- **Added**: Revived the previously unregistered `NunchakuQI&ZITDiffsynthControlnet` node.
- **Fixed**: Resolved an issue where ControlNet failed to apply to Nunchaku Qwen Image models due to custom forward loop skipping dynamic resizing.
- **Technical Details**: See [v2.5.0 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.5.0) for complete explanation

### v2.4.7
- **Fixed**: ComfyUI startup `[ERROR] loss` / `[ERROR] logits` messages from Hugging Face `transformers` `@auto_docstring` when importing Qwen3 VL / Qwen2.5 VL `*CausalLMOutputWithPast`. This is **not a defect in this node's LoRA loading logic**. Because it is unclear when Hugging Face will address this upstream, this node absorbs the issue by wrapping `get_args_doc_from_source` inside `prestartup_script.py` only (no `site-packages` edits, no stderr filtering).
- **Upstream auto-disable (fully automatic)**: On every ComfyUI start, the patch probes upstream `ModelOutputArgs` and runs a subprocess Qwen VL import test. Once `transformers` is fixed upstream, the patch **skips itself** automatically on the next start. **No environment variables or user toggles** (unlike v2.4.6 `apply_rotary_emb` compat, which still allows `QWENIMAGE_ROTARY_COMPAT` opt-out).
- **Note**: LoRA behavior is unchanged. The root cause is upstream `transformers` Qwen VL `@auto_docstring` validation when those `ModelOutput` classes are imported (often via other custom nodes or workflows).
- **Technical Details**: See [v2.4.7 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.7) for complete explanation

### v2.4.6
- **Fixed**: ComfyUI **0.24.x** startup failure when **ComfyUI-nunchaku** imports Qwen Image nodes (`ImportError: cannot import name 'apply_rotary_emb' from 'comfy.ldm.qwen_image.model'`). Adds an early `prestartup_script.py` shim that aliases `apply_rotary_emb` to ComfyUI's `apply_rope1` from this custom node only (no ComfyUI-nunchaku file edits).
- **Technical Details**: See [v2.4.6 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.6) for complete explanation

### v2.4.5
- **Added**: Chinese documentation pages under `zhmd/` (README and release notes), with bilingual language switchers on the English and Chinese README and release note pages.
- **Technical Details**: See [v2.4.5 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.5) for complete explanation

### v2.4.4
- **Fixed**: Restored v2.2.7 first LoRA duplicate file read elimination in `compose_loras_v2` (regression from v2.3.0 AWQ restructuring). The first LoRA is loaded once and reused in the main loop, cutting duplicate file I/O, deserialization, and key classification by 50% for Qwen Image and Z-Image-Turbo loaders. Fully compatible with the AWQ modulation layer monkey patch.
- **Technical Details**: See [v2.4.4 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.4) for complete explanation

### v2.4.3
- **Fixed**: Z-Image / SVDQ crash with ComfyUI deferred (lazy) `Linear` weights (`AttributeError: 'NoneType' object has no attribute 'dtype'`) by patching `SVDQW4A4Linear.from_linear` and `fuse_to_svdquant_linear`, including startup retries for load-order variance.
- **Technical Details**: See [v2.4.3 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.3) for complete explanation

### v2.4.2
- **Fixed**: Qwen Image ControlNet (e.g. Fun ControlNet) with Nunchaku Qwen Image model — `ComfyQwenImageWrapper` now exposes `process_img` and forwards ControlNet-required attributes (`patch_size`, `pe_embedder`, `img_in`, `txt_norm`, `txt_in`, `time_text_embed`) to the inner model so Union ControlNet works when the base model is the wrapper.
- **Fixed**: RecursionError when accessing `model_wrapper.model` (e.g. in NunchakuQwenImageLoraStackV3) — `__getattr__` now obtains the inner model via `_modules` instead of `self.model` to avoid infinite recursion.
- **Technical Details**: See [v2.4.2 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.2) for complete explanation

### v2.4.1
- **Added**: Nunchaku Z-Image-Turbo LoRA Stack V1 with rgthree-style UI - Same layout as Qwen Image LoRA Stack V1: toggle, LoRA name, and strength per row. For official Nunchaku Z-Image loader only. Uses compose_loras_v2. Does not work properly with ComfyUI Nodes 2.0; when using with Nodes 2.0, pressing F5 to refresh will reflect changes.
- **Related Issues**: [Issue #12](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/12) - Request for better LoRA option (rgthree-style UI), [Issue #36](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/36) - Request for enabling/disabling LoRA function

### v2.4.0
- **Added**: Nunchaku Qwen Image LoRA Stack V1 with rgthree-style UI - Clean, minimalist interface inspired by Power Lora Loader (rgthree-comfy). Toggle, LoRA name, and strength per row.
- **Merged**: [PR #49](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/49) - feat(qwen_lora): add Nunchaku Qwen Image LoRA Stack V4 with rgthree-style UI (proposed by [avan06](https://github.com/avan06))
- **Note**: Does not work properly with ComfyUI Nodes 2.0. Use the standard (LiteGraph) canvas.
- **Related Issues**: [Issue #12](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/12) - Request for better LoRA option (rgthree-style UI), [Issue #36](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/36) - Request for enabling/disabling LoRA function
- **Technical Details**: See [v2.4.0 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.0) for complete explanation
