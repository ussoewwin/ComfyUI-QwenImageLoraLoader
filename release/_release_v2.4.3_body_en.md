<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/releases/tag/v2.4.3"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

## v2.4.3

### Highlights

- **ComfyUI-nunchaku** Z-Image / SVDQ: `AttributeError: 'NoneType' object has no attribute 'dtype'` when ComfyUI defers `Linear` weight init — mitigated in this pack via `patches/nunchaku_patch.py` (full write-up: [md/zhmd/v2.4.3.md](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.4.3.md)).
- **LoRA:** First-LoRA file read cache restored in `nodes/lora_loader.py` (v2.2.7 optimization that was lost in v2.3.0 AWQ fix, then restored in v2.4.3).
- **AWQ modulation layers:** `compose_loras_v2` applies LoRA to `img_mod.1` / `txt_mod.1` / `norm1` / `norm1_context` / `adaLN_modulation` (v2.3.0 regression fix).
- **Z-Image Turbo:** `transformer_options` / `attention_mask` / `ref_latents` gap filling in `nodes/model_loading.py` (v2.4.2).
- **ComfyUI 0.4.0:** `ModelPatcher.load` / `load_models_gpu` / `load_models_gpu` retry when `ModelPatcher` is missing (v2.4.3).

### Install

```bash
git pull
pip install -r requirements.txt
```

Restart ComfyUI after updating.

### Links

- [Full technical note (EN)](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/md/zhmd/v2.4.3.md)
- [COMFYUI_0.4.0 model management](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/md/COMFYUI_0.4.0_MODEL_MANAGEMENT_ERRORS.md)
- [ZIMAGETURBO_CONTROLNET_FIX.md](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/md/ZIMAGETURBO_CONTROLNET_FIX.md)
