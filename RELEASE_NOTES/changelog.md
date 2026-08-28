<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/changelogzh.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

### v2.6.0 (latest)
- **Changed**: ControlNet node renamed from `NunchakuQI&ZITDiffsynthControlnet` to **`Nunchaku ZI Diffsynth Controlnet&Krea2 LoRA ControlNet`** (internal class name unchanged: `NunchakuQwenImageDiffsynthControlnet`, so existing workflows load without modification).
- **Clarified supported model scope**:
  - **Z-Image-Turbo (ZI) route**: supports **Nunchaku** (quantized) and all non-quantized / quantized Z-Image models.
  - **Qwen Image (QI) route**: supports all **non-Nunchaku** Qwen Image models (both non-quantized and quantized, e.g. HSWQ ConvRot INT8). **Nunchaku Qwen Image is NOT supported** by the DiffSynth route - the quantized hidden-state scale mismatch produces broken output. Use a standard bf16 Qwen Image model for the QI route, or Nunchaku Z-Image-Turbo for the ZI route.
- **Fixed**: Route classification now recognizes the standard (non-Nunchaku) `QwenImageTransformer2DModel` and routes it to the `qwenimage_standard` path, so DiffSynth ControlNet works correctly with standard bf16 Qwen Image models.
- **Docs**: README (EN + ZH) updated - node image, accurate supported-scope notation, and usage notes.
- **Technical Details**: See [v2.6.0 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.6.0) for complete explanation

### v2.5.9
- **Fixed**: ComfyUI startup `[ERROR]` noise from Hugging Face `transformers` `@auto_docstring` - 13 "but not documented" lines for `DeepseekVLHybridImageProcessorKwargs` (`high_res_size`), `Kimi_K25ImageProcessorKwargs` (`merge_size`) and `PaddleOCRVLImageProcessorKwargs` (`min_pixels` / `max_pixels`). Upstream TypedDict docstrings fail validation: a stray leading space hides `high_res_size`, the kimi docstring documents `merge_kernel_size` instead of `merge_size`, and paddleocr never documents `min_pixels` / `max_pixels`.
- **Non-invasive fix**: No `site-packages` edits. `prestartup_script.py` wraps `get_args_doc_from_source` in-process (same design as the v2.4.7 CausalLM docstring patch) so the four fields resolve through the fallback source dict.
- **Upstream auto-disable (fully automatic)**: On every ComfyUI start the patch probes upstream; once `transformers` fixes the docstrings, the patch skips itself. No environment variables or user toggles.
- **Docs**: English technical guide added at `md/TRANSFORMERS_IMAGE_PROCESSOR_KWARGS_DOCSTRING_PATCH.md`.
- **Technical Details**: See [v2.5.9 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.5.9) for complete explanation

### v2.5.8
- **Fixed**: ComfyUI no longer crashes with a Windows fatal "page error" when a precompiled LoRA cache file is damaged on disk (readable header, unreadable data region). `load_precompiled()` now reads the cache as plain bytes instead of using the mmap-based `load_file`, so the same damage surfaces as an ordinary Python `OSError` and the loader falls back to a full re-fuse automatically.
- **Technical Details**: See [v2.5.8 Release Notes](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.5.8) for complete explanation

### v2.5.7
- **Removed**: Legacy node `NunchakuQwenImageLoraStack` ("Nunchaku Qwen Image LoRA Stack (Legacy)") removed from registered node mappings and documentation. Workflows should use `NunchakuQwenImageLoraStackV1`, `V2`, or `V3` instead.
- **Technical Details**: See [v2.5.7 Release Notes](v2.5.7.md) for complete explanation

### v2.5.6
- **Added**: AMD/ROCm compatibility - nunchaku requires NVIDIA CUDA and is unavailable on AMD/ROCm systems. The package now probes nunchaku availability at startup (`_NUNCHAKU_AVAILABLE`) and automatically disables every nunchaku-dependent node (`NunchakuQwenImageLoraLoader/Stack`, `V1`/`V2`/`V3`, `NunchakuZImageTurboLoraStackV1`/`V4`) on such systems; they are no longer registered.
- **Still available on AMD/ROCm**: The nunchaku-independent Krea2 ControlNet nodes remain fully functional - `Krea2ControlNetLoraLoader` and the Krea2 route of `NunchakuQI&ZITDiffsynthControlnet`.
- **Clear errors**: Runtime nunchaku imports are guarded, so legacy workflows get a clear `RuntimeError` ("nunchaku is required...") instead of an obscure `ImportError`. Startup logs a `[ROCm/AMD] nunchaku is not available...` warning when the dependency is missing.
- **Docs**: README (EN + ZH) updated with an AMD / ROCm compatibility section.
- **Technical Details**: See [v2.5.6 Release Notes](v2.5.6.md) for complete explanation

### v2.5.5
- **Added**: Krea2 **openpose** ControlNet LoRA support (`krea2_turbo_openpose_controlnet.safetensors`) in `NunchakuQI&ZITDiffsynthControlnet`.
- **Exclusive routing**: The Krea2 route now auto-detects the control sub-type from the LoRA file - **depth** (`first.weight` expansion) vs **openpose** (pure block LoRA). The openpose branch is fully isolated from the existing depth / Qwen Image / Z-Image / Nunchaku routes; no existing behavior is changed.
- **Native reference-latent conditioning**: The pose image is VAE-encoded and injected as a native Krea2 reference latent (`index_timestep_zero`) - the conditioning pathway the openpose control LoRA was trained against (256 block patches, rank 32, on all 28 DiT blocks + txtfusion layerwise/refiner blocks).
- **New `use_kv_cache` option** (default `True`): isolated reference K/V mode - one t=0 reference pass precomputes every block's K/V, injected as extra attention keys each step. Matches the ai-toolkit kv_cache training convention used by the reference workflow. Set to `False` to fall back to stock joint-mode refs.
- **Docs**: README (EN) and Chinese README updated with dedicated Krea2 Depth Control and Krea2 OpenPose Control sections.
- **Technical Details**: See [v2.5.5 Release Notes](v2.5.5.md) for complete explanation


### v2.5.3
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
