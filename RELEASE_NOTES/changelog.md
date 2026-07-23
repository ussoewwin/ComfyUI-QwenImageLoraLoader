<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/changelogzh.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

### v2.5.3 (latest)
- **Changed**: Partial adoption of the PR #52 merge plan.
- **Technical Details**: See [v2.5.3 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.5.3) for complete explanation

### v2.5.2
- **Fixed**: Suppressed the cosmetic ComfyUI startup warning `WARNING: Potential Error in code: Torch already imported, torch should never be imported before this point.` ([Issue #53](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/issues/53)). The warning fires because the mandatory `apply_rotary_emb` prestartup shim must import `comfy.ldm` modules that import `torch` at module level (all CUDA env setup already ran, so it is harmless). A one-shot root-logger filter installed early in `prestartup_script.py` drops only that single message and lets every other log through. Opt out with `QWENIMAGE_SUPPRESS_TORCH_WARNING=0`.
- **Technical Details**: See [v2.5.2 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.5.2) for complete explanation

### v2.5.1
- **Added**: Krea2 depth ControlNet LoRA support via `Krea2ControlNetLoraLoader` and the Krea2 route in `NunchakuQI&ZITDiffsynthControlnet`. Load a Krea2 depth controlnet-lora file (for example `krea2-depth-control-lora.safetensors`) from the `controlnet` folder, connect its `MODEL_PATCH` output to the controlnet node `model_patch` input, and apply depth conditioning on Krea2 / SingleStreamDiT models.
- **Technical Details**: See [v2.5.1 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.5.1) for complete explanation

### v2.5.0
- **Added**: Revived the previously unregistered `NunchakuQI&ZITDiffsynthControlnet` node.
- **Fixed**: Resolved an issue where ControlNet failed to apply to Nunchaku Qwen Image models due to custom forward loop skipping dynamic resizing.
- **Technical Details**: See [v2.5.0 Release Notes](v2.5.0.md) for complete explanation

### v2.4.7
- **Fixed**: ComfyUI startup `[ERROR] loss` / `[ERROR] logits` messages from Hugging Face `transformers` `@auto_docstring` when importing Qwen3 VL / Qwen2.5 VL `*CausalLMOutputWithPast`. This is **not a defect in this node's LoRA loading logic**. Because it is unclear when Hugging Face will address this upstream, this node absorbs the issue by wrapping `get_args_doc_from_source` inside `prestartup_script.py` only (no `site-packages` edits, no stderr filtering).
- **Upstream auto-disable (fully automatic)**: On every ComfyUI start, the patch probes upstream `ModelOutputArgs` and runs a subprocess Qwen VL import test. Once `transformers` is fixed upstream, the patch **skips itself** automatically on the next start. **No environment variables or user toggles** (unlike v2.4.6 `apply_rotary_emb` compat, which still allows `QWENIMAGE_ROTARY_COMPAT` opt-out).
- **Note**: LoRA behavior is unchanged. The root cause is upstream `transformers` Qwen VL `@auto_docstring` validation when those `ModelOutput` classes are imported (often via other custom nodes or workflows).
- **Technical Details**: See [v2.4.7 Release Notes](v2.4.7.md) for complete explanation

### v2.4.6
- **Fixed**: ComfyUI **0.24.x** startup failure when **ComfyUI-nunchaku** imports Qwen Image nodes (`ImportError: cannot import name 'apply_rotary_emb' from 'comfy.ldm.qwen_image.model'`). Adds an early `prestartup_script.py` shim that aliases `apply_rotary_emb` to ComfyUI's `apply_rope1` from this custom node only (no ComfyUI-nunchaku file edits).
- **Technical Details**: See [v2.4.6 Release Notes](v2.4.6.md) for complete explanation

### v2.4.5
- **Added**: Chinese documentation pages under `zhmd/` (README and release notes), with bilingual language switchers on the English and Chinese README and release note pages.
- **Technical Details**: See [v2.4.5 Release Notes](v2.4.5.md) for complete explanation

### v2.4.4
- **Fixed**: Restored v2.2.7 first LoRA duplicate file read elimination in `compose_loras_v2` (regression from v2.3.0 AWQ restructuring). The first LoRA is loaded once and reused in the main loop, cutting duplicate file I/O, deserialization, and key classification by 50% for Qwen Image and Z-Image-Turbo loaders. Fully compatible with the AWQ modulation layer monkey patch.
- **Technical Details**: See [v2.4.4 Release Notes](v2.4.4.md) for complete explanation

### v2.4.3
- **Fixed**: Z-Image / SVDQ crash with ComfyUI deferred (lazy) `Linear` weights (`AttributeError: 'NoneType' object has no attribute 'dtype'`) by patching `SVDQW4A4Linear.from_linear` and `fuse_to_svdquant_linear`, including startup retries for load-order variance.
- **Technical Details**: See [v2.4.3 Release Notes](v2.4.3.md) for complete explanation

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
