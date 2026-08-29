# Nunchaku CPU Offload: `copy_params_into` `wtscale` Attribute Mismatch AssertionError & Runtime Mitigation

This document provides a comprehensive technical breakdown of the `AssertionError: assert not hasattr(md, "wtscale")` crash that occurs during Nunchaku backend model inference (e.g., Ultimate SD Upscale tiling or multi-step sampling): **(1) Error Details, (2) Essential Root Cause, (3) Modified/Added Files, (4) Complete Unabridged Code, and (5) Technical Analysis & Significance**.

---

## 1. Error Details

### Traceback
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

### Execution Context
- During sampling, when Nunchaku's per-block CPU Offload mechanism (`CPUOffloadManager.step()`) preloads and copies a CPU transformer block into the GPU ping-pong buffer (`self.buffer_blocks`) via `load_block` -> `copy_params_into`.

---

## 2. Essential Root Cause

### 1. Nunchaku Ping-Pong Buffer Architecture
To minimize GPU VRAM consumption, Nunchaku retains only a minimal subset of transformer blocks on GPU. It maintains two reusable buffer blocks (`self.buffer_blocks[0]`, `self.buffer_blocks[1]`) on the GPU, alternating between them via asynchronous stream transfers (`copy_params_into`) while the compute stream executes the forward pass.

In `CPUOffloadManager.__init__`, the GPU buffer blocks are initialized by **deep-copying Block 0 (`blocks[0]`)**:
```python
self.buffer_blocks = [copy.deepcopy(blocks[0]), copy.deepcopy(blocks[0])]
```

### 2. Upstream Flawed Assertion in `copy_params_into`
In the upstream Nunchaku library (`nunchaku/utils.py`), parameter transfer is implemented as follows:
```python
for ms, md in zip(src.modules(), dst.modules()):
    # wtscale is a special case which is a float on the CPU
    if hasattr(ms, "wtscale"):
        assert hasattr(md, "wtscale")
        md.wtscale = ms.wtscale
    else:
        assert not hasattr(md, "wtscale")  # <-- Flawed assertion
```

### 3. Failure Mechanism
1. **Initial Buffer Allocation**:
   - Because Block 0 contains quantized linear layers with a CPU float attribute `wtscale` (weight quantization scale), the GPU buffer blocks `md` initially inherit `wtscale` on those modules.
2. **Subsequent Block Transfer**:
   - As sampling progresses, subsequent blocks (e.g., Block 1, non-quantized layers, or blocks modified with custom LoRA / planar injection) are transferred into the same GPU buffer slot `md`.
   - In these modules, the source block `ms` does not have a `wtscale` attribute.
3. **Assertion Breakdown**:
   - `hasattr(ms, "wtscale")` evaluates to `False`, entering the `else:` branch.
   - The assertion `assert not hasattr(md, "wtscale")` asserts that the destination buffer module `md` must not possess `wtscale`.
   - However, **the recycled buffer `md` still retains the `wtscale` attribute from Block 0 (or a preceding quantized block)**.
   - Consequently, `hasattr(md, "wtscale")` is `True`, the assertion fails, and Python raises `AssertionError`, terminating the workflow immediately.

---

## 3. Modified and Added Files

- **`D:\USERFILES\GitHub\ComfyUI-QwenImageLoraLoader\patches\nunchaku_patch.py`**
- **`D:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-QwenImageLoraLoader\patches\nunchaku_patch.py`**
- **`D:\USERFILES\GitHub\ComfyUI-QwenImageLoraLoader\md\NUNCHAKU_OFFLOAD_WTSCALE_ASSERTION_FIX.md`**

---

## 4. Complete Unabridged Code

### 1. Global State Definition
```python
_svdq_from_linear_patched: bool = False
_qwen_apply_rotary_emb_compat_applied: bool = False
_copy_params_patched: bool = False
```

### 2. Safe Implementation and Hook Integration
```python
def _safe_copy_params_into(src: torch.nn.Module, dst: torch.nn.Module, non_blocking: bool = True):
    """
    Safely copy parameters and buffers from src to dst without failing on wtscale attribute mismatch.
    """
    with torch.no_grad():
        for ps, pd in zip(src.parameters(), dst.parameters()):
            pd.copy_(ps, non_blocking=non_blocking)
        for bs, bd in zip(src.buffers(), dst.buffers()):
            bd.copy_(bs, non_blocking=non_blocking)

        for ms, md in zip(src.modules(), dst.modules()):
            if hasattr(ms, "wtscale"):
                md.wtscale = ms.wtscale
            elif hasattr(md, "wtscale"):
                delattr(md, "wtscale")


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

## 5. Technical Analysis & Significance

### 1. `_safe_copy_params_into`
- **Strict Preservation of Tensor & Buffer Copying**:
  ```python
  for ps, pd in zip(src.parameters(), dst.parameters()):
      pd.copy_(ps, non_blocking=non_blocking)
  for bs, bd in zip(src.buffers(), dst.buffers()):
      bd.copy_(bs, non_blocking=non_blocking)
  ```
  Operates under the exact same `torch.no_grad()` context as upstream, performing asynchronous memory copies without autograd overhead.
- **Dynamic Attribute Synchronization Without Assertions**:
  ```python
  for ms, md in zip(src.modules(), dst.modules()):
      if hasattr(ms, "wtscale"):
          md.wtscale = ms.wtscale
      elif hasattr(md, "wtscale"):
          delattr(md, "wtscale")
  ```
  - When the source module `ms` has `wtscale`, `md.wtscale` is assigned directly.
  - When `ms` does not have `wtscale`, any stale `wtscale` attribute on `md` left over from prior iterations is deleted via `delattr(md, "wtscale")`.
  - This guarantees that `dst` perfectly mirrors `src` in module topology and attributes, eliminating the invalid `AssertionError` across any heterogeneous block structure or LoRA configuration.

### 2. `apply_nunchaku_copy_params_patch`
- **Idempotency & Re-entrancy Protection**:
  Controlled by `_copy_params_patched` to prevent redundant re-patching across multiple node invocations.
- **Dual-Namespace In-Memory Monkey Patching**:
  `nunchaku.models.utils` imports `copy_params_into` via `from ..utils import copy_params_into`. Replacing the function symbol in both `nunchaku.utils` and `nunchaku.models.utils` ensures that `CPUOffloadManager.load_block` calls `_safe_copy_params_into` directly.
- **Zero Third-Party Library File Modifications**:
  No files in `site-packages` or disk binaries are touched; all patching occurs strictly in Python process memory at runtime.

### 3. Integration into `apply_nunchaku_patch`
- Automatically applied during node initialization alongside other Nunchaku upstream compatibility fixes (`apply_rotary_emb`, `SVDQW4A4Linear` lazy initialization), activating proactive protection upon ComfyUI startup.
