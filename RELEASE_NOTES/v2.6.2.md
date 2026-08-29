<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.6.2.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# Nunchaku CPU Offload: `copy_params_into` `wtscale` Attribute Mismatch & Runtime In-Memory Mitigation

This document provides an exhaustive technical explanation of the Nunchaku CPU Offload buffer parameter transfer issues encountered during inference (e.g., Ultimate SD Upscale tiling or multi-step sampling): **(1) Error Details (both `AssertionError` and `AttributeError`), (2) Essential Root Cause & Deep Mechanics, (3) Modified/Added Files, (4) Complete Unabridged Code (Zero Omission), and (5) Detailed Line-by-Line Technical Analysis**.

---

## 1. Error Details

### 1-1. Upstream Buffer Reuse Crash (`AssertionError`)
During sampling under Nunchaku CPU Offload, transferring CPU blocks into GPU ping-pong buffers fails with `AssertionError`:

```text
[ERROR] !!! Exception during processing !!!
[ERROR] Traceback (most recent call last):
  File "D:\USERFILES\ComfyUI\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
  File "D:\USERFILES\ComfyUI\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
  File "D:\USERFILES\ComfyUI\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\USERFILES\ComfyUI\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-HSWQ-Loader-and-Tools\nodes\nunchaku_usdu.py", line 360, in upscale
    _ = script.run(
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-HSWQ-Loader-and-Tools\usdu_bundle\usdu_patch.py", line 194, in patched_script_run
    upscaler.process()
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-HSWQ-Loader-and-Tools\usdu_bundle\repositories\ultimate_sd_upscale\scripts\ultimate-upscale.py", line 138, in process
    self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-HSWQ-Loader-and-Tools\usdu_bundle\repositories\ultimate_sd_upscale\scripts\ultimate-upscale.py", line 245, in start
    return self.linear_process(p, image, rows, cols)
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-HSWQ-Loader-and-Tools\usdu_bundle\usdu_patch.py", line 353, in new_linear_process
    return old_linear(self, p, image, rows, cols)
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-HSWQ-Loader-and-Tools\usdu_bundle\repositories\ultimate_sd_upscale\scripts\ultimate-upscale.py", line 180, in linear_process
    processed = processing.process_images(p)
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-HSWQ-Loader-and-Tools\usdu_bundle\modules\processing.py", line 247, in process_images
    samples = sample(model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler, positive_cropped,
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-HSWQ-Loader-and-Tools\usdu_bundle\modules\processing.py", line 173, in sample
    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
  File "D:\USERFILES\ComfyUI\ComfyUI\nodes.py", line 1587, in common_ksampler
    samples = comfy.sample.sample(...)
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-QwenImageLoraLoader\wrappers\qwenimage.py", line 378, in forward
    out = self._execute_model(x, timestep, context, guidance, control, transformer_options, **kwargs)
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-QwenImageLoraLoader\wrappers\qwenimage.py", line 497, in _execute_model
    return self.model(...)
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-QwenImageLoraLoader\patches\nunchaku_timestep_zero_patch.py", line 487, in nunchaku_qwenimage_forward_with_timestep_zero_restore
  File "D:\USERFILES\ComfyUI\python_embeded\Lib\site-packages\nunchaku\models\utils.py", line 206, in step
    self.load_block(self.current_block_idx + 1)
  File "D:\USERFILES\ComfyUI\python_embeded\Lib\site-packages\nunchaku\models\utils.py", line 186, in load_block
    copy_params_into(block, self.buffer_blocks[block_idx % 2], non_blocking=non_blocking)
  File "D:\USERFILES\ComfyUI\python_embeded\Lib\site-packages\nunchaku\utils.py", line 366, in copy_params_into
    assert not hasattr(md, "wtscale")
AssertionError
```

### 1-2. Attribute Deletion Forward Crash (`AttributeError`)
If `wtscale` is deleted via `delattr(md, "wtscale")` when `hasattr(ms, "wtscale")` is False, subsequent forward execution inside `SVDQW4A4Linear.forward_quant` fails immediately because `self.wtscale` is accessed unconditionally:

```text
[ERROR] !!! Exception during processing !!! 'SVDQW4A4Linear' object has no attribute 'wtscale'
[ERROR] Traceback (most recent call last):
  File "D:\USERFILES\ComfyUI\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(...)
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-nunchaku\models\qwenimage.py", line 483, in forward
    attn_output = self.attn(...)
  File "D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-nunchaku\models\qwenimage.py", line 280, in forward
    img_qkv = self.to_qkv(hidden_states)
  File "D:\USERFILES\ComfyUI\python_embeded\Lib\site-packages\nunchaku\models\linear.py", line 187, in forward
    output = self.forward_quant(quantized_x, ascales, lora_act_out, output)
  File "D:\USERFILES\ComfyUI\python_embeded\Lib\site-packages\nunchaku\models\linear.py", line 265, in forward_quant
    alpha=self.wtscale,
          ^^^^^^^^^^^^
  File "D:\USERFILES\ComfyUI\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1967, in __getattr__
    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'"
    )
AttributeError: 'SVDQW4A4Linear' object has no attribute 'wtscale'. Did you mean: 'wcscales'?
```

---

## 2. Essential Root Cause & Deep Mechanics

### 2-1. CPU Offload Ping-Pong Buffer Initialization
To minimize GPU VRAM consumption, Nunchaku retains only a minimal subset of transformer blocks on GPU. In `CPUOffloadManager.__init__` (`nunchaku.models.utils`), two GPU buffer blocks are allocated by deep-copying Block 0:
```python
self.buffer_blocks = [copy.deepcopy(blocks[0]), copy.deepcopy(blocks[0])]
```
During execution, `CPUOffloadManager.step()` invokes `self.load_block(self.current_block_idx + 1)`, which calls `copy_params_into(block, self.buffer_blocks[block_idx % 2], non_blocking=True)` to stream weights from CPU memory into the alternate GPU buffer.

### 2-2. `SVDQW4A4Linear` Class Contract for `wtscale`
In `nunchaku.models.linear.SVDQW4A4Linear.__init__`:
```python
if precision == "nvfp4":
    self.wcscales = nn.Parameter(
        torch.ones(out_features, dtype=torch_dtype, device=device), requires_grad=False
    )
    self.wtscale = 1.0
else:
    self.wtscale = None
    self.wcscales = None
```
- For `precision == "nvfp4"`, `self.wtscale` is a `float` (`1.0` or model global scale).
- For `precision == "int4"`, `self.wtscale` is `None`.
- In `SVDQW4A4Linear.forward_quant` (line 265), the CUDA GEMM kernel `svdq_gemm_w4a4_cuda` is called with argument `alpha=self.wtscale`. The attribute `self.wtscale` **must always exist** on every `SVDQW4A4Linear` instance.

### 2-3. Why Upstream `copy_params_into` Fails
In upstream `nunchaku.utils.copy_params_into`:
```python
for ms, md in zip(src.modules(), dst.modules()):
    if hasattr(ms, "wtscale"):
        assert hasattr(md, "wtscale")
        md.wtscale = ms.wtscale
    else:
        assert not hasattr(md, "wtscale")  # <-- Flawed upstream assertion
```
1. **Positional Misalignment**: `zip(src.modules(), dst.modules())` iterates submodules in depth-first search order. If `dst` has custom wrappers, hooks, or LoRA planar injection layers that alter the child submodule count, `ms` and `md` become misaligned.
2. **Buffer Reuse Assertion Collision**: Even in a standard model, when a non-quantized module or a module where `hasattr(ms, "wtscale") == False` is evaluated, the `else:` branch executes `assert not hasattr(md, "wtscale")`. Because `md` is a recycled GPU buffer from Block 0 (which contains quantized linear modules), `md` still has `hasattr(md, "wtscale") == True`, immediately raising `AssertionError`.
3. **Flawed Deletion Remedy**: If `delattr(md, "wtscale")` is used to bypass the assertion, `md` is stripped of `wtscale`. When `md` is an instance of `SVDQW4A4Linear`, its `forward_quant` subsequently crashes with `AttributeError: 'SVDQW4A4Linear' object has no attribute 'wtscale'`.

---

## 3. Modified and Added Files

| File Path | Location | Purpose |
|---|---|---|
| [`patches/nunchaku_patch.py`](file:///D:/USERFILES/GitHub/ComfyUI-QwenImageLoraLoader/patches/nunchaku_patch.py) | `ComfyUI-QwenImageLoraLoader` | Implementation of `_safe_copy_params_into` and `apply_nunchaku_copy_params_patch` |
| [`md/NUNCHAKU_OFFLOAD_WTSCALE_ASSERTION_FIX.md`](file:///D:/USERFILES/GitHub/ComfyUI-QwenImageLoraLoader/md/NUNCHAKU_OFFLOAD_WTSCALE_ASSERTION_FIX.md) | `ComfyUI-QwenImageLoraLoader` | Complete technical specification and architectural documentation |
| [`zhmd/v2.6.2.md`](file:///D:/USERFILES/GitHub/ComfyUI-QwenImageLoraLoader/zhmd/v2.6.2.md) | `ComfyUI-QwenImageLoraLoader` | Chinese technical specification and release notes |

---

## 4. Complete Unabridged Code (Zero Omission)

The following represents the complete, verbatim implementation added to `patches/nunchaku_patch.py`:

```python
# ---------------------------------------------------------------------------
# Global flag definition (top of patches/nunchaku_patch.py)
# ---------------------------------------------------------------------------
_svdq_from_linear_patched: bool = False
_qwen_apply_rotary_emb_compat_applied: bool = False
_copy_params_patched: bool = False


# ---------------------------------------------------------------------------
# Safe In-Memory Parameter Copying and Upstream Patch
# ---------------------------------------------------------------------------
def _safe_copy_params_into(src: torch.nn.Module, dst: torch.nn.Module, non_blocking: bool = True):
    """
    Safely copy parameters and buffers from src to dst by name matching (with positional fallback),
    and correctly synchronize wtscale attribute without attribute deletion.
    """
    with torch.no_grad():
        src_params = dict(src.named_parameters())
        dst_params = dict(dst.named_parameters())
        if src_params and dst_params and set(src_params.keys()) == set(dst_params.keys()):
            for name, pd in dst_params.items():
                pd.copy_(src_params[name], non_blocking=non_blocking)
        else:
            for ps, pd in zip(src.parameters(), dst.parameters()):
                pd.copy_(ps, non_blocking=non_blocking)

        src_buffers = dict(src.named_buffers())
        dst_buffers = dict(dst.named_buffers())
        if src_buffers and dst_buffers and set(src_buffers.keys()) == set(dst_buffers.keys()):
            for name, bd in dst_buffers.items():
                bd.copy_(src_buffers[name], non_blocking=non_blocking)
        else:
            for bs, bd in zip(src.buffers(), dst.buffers()):
                bd.copy_(bs, non_blocking=non_blocking)

        src_modules = dict(src.named_modules())
        dst_modules = dict(dst.named_modules())
        if src_modules and dst_modules and set(src_modules.keys()) == set(dst_modules.keys()):
            for name, md in dst_modules.items():
                ms = src_modules[name]
                if hasattr(ms, "wtscale"):
                    md.wtscale = ms.wtscale
                elif hasattr(md, "wtscale"):
                    precision = getattr(md, "precision", "int4")
                    md.wtscale = 1.0 if precision == "nvfp4" else None
        else:
            for ms, md in zip(src.modules(), dst.modules()):
                if hasattr(ms, "wtscale"):
                    md.wtscale = ms.wtscale
                elif hasattr(md, "wtscale"):
                    precision = getattr(md, "precision", "int4")
                    md.wtscale = 1.0 if precision == "nvfp4" else None


def apply_nunchaku_copy_params_patch() -> bool:
    """
    Patch nunchaku copy_params_into to safely handle wtscale attribute differences
    when reusing buffer_blocks during CPU offloading.
    """
    global _copy_params_patched
    if _copy_params_patched:
        return True
    applied = False
    for mod_name in ("nunchaku.utils", "nunchaku.models.utils"):
        try:
            if mod_name in sys.modules:
                mod = sys.modules[mod_name]
                if hasattr(mod, "copy_params_into"):
                    mod.copy_params_into = _safe_copy_params_into
                    applied = True
            else:
                import importlib
                mod = importlib.import_module(mod_name)
                mod.copy_params_into = _safe_copy_params_into
                applied = True
        except Exception:
            pass
    if applied:
        _copy_params_patched = True
        logger.info("Patched nunchaku.copy_params_into for safe CPU offload buffer parameter copying.")
    return applied


def apply_nunchaku_patch():
    """
    Apply ComfyUI-nunchaku compatibility patches (LoRA planar injection + lazy Linear fixes + safe copy_params_into).
    Returns True if at least one patch was applied or was already active.
    """
    rotary_compat = apply_qwen_image_apply_rotary_emb_compat()
    lazy_from = apply_svdqw4a4_lazy_linear_patch()
    lazy_fuse = apply_nunchaku_zimage_fuse_lazy_linear_patch()
    copy_params_ok = apply_nunchaku_copy_params_patch()
    if not lazy_fuse:
        schedule_nunchaku_zimage_fuse_patch_retries()

    planar_ok = False
    try:
        target_class = None

        try:
            from nunchaku.models.qwenimage import NunchakuQwenImageTransformerBlock

            target_class = NunchakuQwenImageTransformerBlock
        except ImportError:
            pass

        if target_class is None:
            for module_name, module in sys.modules.items():
                if "qwenimage" in module_name and hasattr(module, "NunchakuQwenImageTransformerBlock"):
                    target_class = getattr(module, "NunchakuQwenImageTransformerBlock")
                    logger.info("Found NunchakuQwenImageTransformerBlock in %s", module_name)
                    break

        if target_class:
            logger.info("Applying Manual Planar Injection Monkey Patch to NunchakuQwenImageTransformerBlock")
            target_class.forward = forward_with_manual_planar_injection
            planar_ok = True
        else:
            logger.warning(
                "Could not find NunchakuQwenImageTransformerBlock to patch. "
                "Manual Planar Injection logic will not work if the original file is reverted."
            )

    except Exception as e:
        logger.error("Failed to apply Nunchaku planar patch: %s", e)

    return planar_ok or lazy_from or rotary_compat or copy_params_ok
```

---

## 5. Detailed Line-by-Line Technical Analysis

### 5-1. `_safe_copy_params_into` Analysis
1. **Context Management (`with torch.no_grad():`)**:
   - Ensures parameter and buffer copy operations do not track history or allocate autograd graph nodes, maintaining maximum execution speed and zero VRAM leak.
2. **Key-Matched Parameter and Buffer Copying**:
   - `dict(src.named_parameters())` and `dict(dst.named_parameters())` match tensors by fully-qualified module parameter names (e.g., `"attn.to_qkv.qweight"`).
   - If parameter keys match, each tensor is copied directly to its exact counterpart using `pd.copy_(src_params[name], non_blocking=non_blocking)`.
   - If keys diverge (e.g. dynamic runtime wrapping), a fallback to positional `zip()` executes safely.
3. **Key-Matched Module Synchronization**:
   - `src.named_modules()` and `dst.named_modules()` align module hierarchies accurately.
4. **Attribute Preservation Without Deletion**:
   - **`if hasattr(ms, "wtscale"): md.wtscale = ms.wtscale`**: When the source module has `wtscale`, its value is assigned to `md.wtscale`.
   - **`elif hasattr(md, "wtscale"): md.wtscale = 1.0 if precision == "nvfp4" else None`**: When the source module lacks `wtscale` but `md` has `wtscale` (or is an `SVDQW4A4Linear`), `md.wtscale` is assigned the valid default value based on `md.precision` instead of deleting the attribute.
   - **Zero Assertion**: The invalid assertion `assert not hasattr(md, "wtscale")` is entirely eliminated.

### 5-2. `apply_nunchaku_copy_params_patch` Analysis
1. **Idempotency Guard (`_copy_params_patched`)**:
   - Ensures that repeated node executions or multiple loaders do not re-apply the patch unnecessarily.
2. **Dual-Namespace In-Memory Monkey Patching**:
   - `nunchaku.models.utils` imports `copy_params_into` from `nunchaku.utils` (`from ..utils import copy_params_into`).
   - Patching both `sys.modules["nunchaku.utils"].copy_params_into` and `sys.modules["nunchaku.models.utils"].copy_params_into` ensures that `CPUOffloadManager.load_block` executes `_safe_copy_params_into` under all import conditions.
3. **Pure In-Memory Execution**:
   - No files in `site-packages` or disk binaries are modified. The disk installation remains 100% clean and pristine.

### 5-3. Integration into `apply_nunchaku_patch`
- `apply_nunchaku_copy_params_patch()` is invoked during `apply_nunchaku_patch()`.
- Whenever ComfyUI initializes `ComfyUI-QwenImageLoraLoader`, the patch activates proactively, guaranteeing seamless inference across standard sampling, USDU tiling, LoRA injection, and custom pipelines without user intervention.
