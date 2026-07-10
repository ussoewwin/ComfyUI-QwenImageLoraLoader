# PR #52 Merge Plan — Technical Analysis & Adoption Items

## Overview

- **PR**: [#52 feat: GPU compile acceleration, atomic cache writes, and V2 TE loaders with CPU offload](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/52)
- **Author**: [Sniper199999](https://github.com/Sniper199999)
- **Merge-base**: `14f1e32` (v2.5.1 established main HEAD)
- **Scope**: +1465 / -364 across 17 files
- **Policy**: **Approve the proposal, overwrite problematic sections with current v2.5.1 code**
- **Conclusion**: No malice — valuable contributions mixed with stale v2.4.x working-copy artifacts due to rebase omission

---

## Premise

- The PR is correctly based on v2.5.1 `main`, but the working copy was from the v2.4.x era and was not rebased before commit.
- The PR adds genuinely valuable features: TE V2 CPU offload, LoRA precompiled cache, GPU pack/unpack optimization, and a real bug fix for bypass-path wrapper corruption.
- The PR also accidentally reverts v2.5.0/v2.5.1 established features (version bump, Krea2/Diffsynth ControlNet registration, rotary compat patch, `.gitignore` entries, README structure).
- **Policy**: Approve the proposal itself, then overwrite the problematic sections with current v2.5.1 code so v2.5.x achievements are preserved while new features are absorbed.

---

## Adoption Items (7 adopted, 1 rejected)

### Adoption 1: `_has_ever_had_loras` bypass-path corruption fix

**Target file**: `wrappers/qwenimage.py`

#### Current code problem

```python:159:162:wrappers/qwenimage.py
        model_is_dirty = (
            not self.loras and # We expect no LoRA
            hasattr(self.model, "_lora_slots") and self.model._lora_slots # But the model actually has LoRA
        )
```

```python:183:186:wrappers/qwenimage.py
        if loras_changed or model_is_dirty or device_changed:
            # The compose function handles resetting before applying the new stack
            reset_lora_v2(self.model)
            self._applied_loras = self.loras.copy()
```

`load_lora()` has a side effect: it permanently mutates the input model's `diffusion_model` to a `ComfyQwenImageWrapper(loras=[])` — call this W1. The returned model's wrapper (W2) holds the actual LoRAs and **shares the same `NunchakuQwenImageTransformer2DModel` instance** as W1.

When the LoRA node is bypassed, ComfyUI routes through the input model (with W1). On W1's first forward pass:

1. `_applied_loras = None` → `loras_changed = True`
2. `reset_lora_v2(shared_transformer)` → **wipes the LoRA composition that W2 had applied**
3. On the next run with LoRA active again, W2 sees `_applied_loras == loras` → skips recompose → LoRA is silently missing

The result is intermittent LoRA corruption: LoRA appears to stop working randomly, and toggling bypass restores it briefly before it breaks again.

#### PR improvement

```python
# __init__ — add flag
self._has_ever_had_loras = False

# forward() — update flag before dirty check
if self.loras:
    self._has_ever_had_loras = True

model_is_dirty = (
    self._has_ever_had_loras and  # We were (or are) a LoRA manager
    not self.loras and            # We currently expect no LoRA
    hasattr(self.model, "_lora_slots") and self.model._lora_slots  # But the model has LoRA
)

if loras_changed or model_is_dirty or device_changed:
    is_lora_manager = bool(self.loras) or self._has_ever_had_loras
    if is_lora_manager:
        reset_lora_v2(self.model)
    # ...
```

W1 (`_has_ever_had_loras = False`) now skips the entire transformer-mutation block. Only wrappers that actually manage LoRAs can reset and recompose the shared transformer.

#### Effects

- **Fixes**: Intermittent LoRA silent-disappearance on bypass toggle
- **Fixes**: LoRA not re-applying after bypass restore because W2 skips recompose
- **Safety**: `_has_ever_had_loras` is monotonic (once True, never reverts), so W1 permanently avoids reset

#### Relation to md docs

- `QWEN_IMAGE_CONTROLNET_AND_GETATTR_FIX.md` (v2.4.2 `__getattr__` recursion fix) is a different location, but the same bug family: "wrapper incorrectly mutates shared transformer"
- `PR28_FIX_EXPLANATION.md` recorded an `IS_CHANGED` fix; this is the same category of correct adaptation to ComfyUI's caching/execution model

---

### Adoption 2: Remove `model` argument from `IS_CHANGED`

**Target files**: `nodes/lora/qwenimage.py`, `nodes/lora/qwenimage_v3.py`, `nodes/lora/qwenimage_v2.py` (v2 is missing the fix in the PR — must apply manually)

#### Current code problem

```python:36:nodes/lora/qwenimage.py
    def IS_CHANGED(s, model, lora_name, lora_strength, cpu_offload="disable", *args, **kwargs):
```

ComfyUI's caching system calls `IS_CHANGED` **without** the `model` argument. MODEL-type inputs (outputs from other nodes) are not passed. Result:

1. `model` argument is not provided → `TypeError`
2. ComfyUI converts unhandled exceptions in `IS_CHANGED` to `NaN`
3. `NaN` is the sentinel that tells ComfyUI to always re-execute the node
4. **Full LoRA recomposition on every queue run, even when nothing changed**

#### PR improvement

```python
def IS_CHANGED(s, lora_name, lora_strength, cpu_offload="disable", save_precompiled_lora=False, **kwargs):
    import hashlib
    m = hashlib.sha256()
    m.update(lora_name.encode())
    m.update(str(lora_strength).encode())
    m.update(cpu_offload.encode())
    m.update(str(save_precompiled_lora).encode())
    return m.digest().hex()
```

`model` is removed from the signature; excess arguments are absorbed by `**kwargs`. Hash is computed from lora_name, lora_strength, cpu_offload, and save_precompiled_lora only.

#### Effects

- **Fixes**: Every-generation LoRA recomposition is no longer forced
- **Reduces**: 1-3 seconds per generation for 500MB LoRA, 5-10 seconds for 2GB LoRA
- **Side effect**: As long as non-LoRA parameters don't change, hash is identical → compose is skipped

#### Relation to md docs

- **Evolution of** `PR28_FIX_EXPLANATION.md` section ② "Fix 2: Additional Fixes Applied (IS_CHANGED Methods)". PR #28 added a default value for `cpu_offload` but kept `model`. PR #52 removes `model` entirely — a more fundamental fix. They are compatible.

#### Note

The PR does **not** apply this fix to `qwenimage_v2.py`. When adopting, the same fix must be applied to v2 as well.

---

### Adoption 3: LoRA precompiled cache

**Target files**: `nunchaku_code/lora_cache.py` (new), `nunchaku_code/lora_qwen.py`, node files

#### Current code problem

`compose_loras_v2` Section 4, every run:

1. `_load_lora_state_dict_robust(lora_path)` — load safetensors from disk
2. `_classify_and_map_key` — apply 30 regex patterns to each key
3. `_fuse_*_lora` — fuse A/B tensors
4. Accumulate into `aggregated_weights`
5. `_apply_lora_to_module` — apply to module

When the same LoRA is used repeatedly, steps 1-3 repeat every generation. Significant for 500MB-2GB LoRAs.

#### PR improvement

New module `lora_cache.py` provides:

- `get_cache_dir(comfy_base)` → `ComfyUI/models/SVDQLora/`
- `get_cache_path(lora_path, cache_dir)` → collision-avoiding filename with SHA256[:8] of parent directory
- `is_cache_valid(lora_path, cache_path)` → file existence + mtime sidecar JSON check
- `save_precompiled(processed_groups, lora_path, cache_path)` → atomic write of `{module_key__A, module_key__B, module_key__alpha}`
- `load_precompiled(cache_path)` → reconstruct `{module_key: (A, B, alpha)}`

`compose_loras_v2` gains a cache fast path:

```python
if _cache_dir_path is not None and isinstance(lora_path_or_dict, (str, Path)):
    _cache_path = _get_cache_path(lora_path_or_dict, _real_cache_dir)
    if _is_cache_valid(lora_path_or_dict, _cache_path):
        processed_groups = _load_precompiled(_cache_path)
        if processed_groups:
            # skip classify + fuse
            for module_key, (A, B, alpha) in processed_groups.items():
                aggregated_weights[module_key].append(...)
            continue
```

#### Effects

- **Speed**: Skips classify + fuse stages. Significantly reduces disk I/O and CPU computation on 2nd+ generation
- **Safety**: B tensor is saved **unscaled**; strength is re-applied at inference time → strength slider remains compatible
- **Atomic write**: `tmp + os.replace` prevents corruption on interruption
- **mtime validation**: Cache auto-invalidates when LoRA file is updated
- **Collision avoidance**: Same-named LoRAs in different directories don't collide

#### Relation to md docs

- **Evolution of** `PERFORMANCE_OPTIMIZATION_PLAN.md` item 1 ("add `lru_cache` to `_load_lora_state_dict_robust`"). The plan was session-scoped; the PR's precompiled cache is persistent disk-scoped — more powerful.
- **Complementary to** `LOST_OPTIMIZATION_v227_v230.md` ("first LoRA duplicate read elimination"). When precompiled cache hits, the double-read problem doesn't arise at all.

#### Concerns

1. No integrity verification (hash signature) of cache contents. Low risk for local use, but tamper risk in shared environments.
2. mtime sidecar may produce false-positive invalidation on copy/sync. Acceptable in practice.
3. **`_cached_first_lora_state_dict` mechanism is absent from the PR's `compose_loras_v2`**. When cache is invalid (default), the first LoRA is read twice. This re-loses the optimization restored in `LOST_OPTIMIZATION_v227_v230.md`. **Must be restored when adopting.**

---

### Adoption 4: TE V2 CPU offload — REJECTED (not adopted)

**Target files**: `nodes/te_offload/nunchaku_te_v2.py` (new), `__init__.py`

#### PR claim

Adds three V2 loader nodes (`NunchakuQwenImageEditEncoderLoaderV2`, `NunchakuQwenImageTextEncoderLoaderV2`, `NunchakuQwen3TextEncoderLoaderV2`) with an `offload_after_encode` toggle that moves the text encoder to CPU after encode.

#### Why rejected

1. **Referenced loader classes do not exist.** The PR's `_delegate_load` method attempts to import `NunchakuQwenImageEditEncoderLoader`, `NunchakuQwenImageTextEncoderLoader`, and `NunchakuQwen3TextEncoderLoader` from a module path `custom_nodes/ComfyUI-nunchaku/nodes/models/qwen_text_encoder.py`. Verified against the installed ComfyUI-nunchaku: **this file does not exist**. The actual TE module is `nodes/models/text_encoder.py` and contains only `NunchakuTextEncoderLoader` / `NunchakuTextEncoderLoaderV2` (FLUX T5 — not Qwen). The three Qwen-specific classes the PR references **do not exist in ComfyUI-nunchaku or ComfyUI core**.

2. **CPU offload is already available via existing ComfyUI mechanisms.** Qwen Image text encoders are loaded through ComfyUI's standard CLIP loader pipeline, which uses `ModelPatcher` for VRAM management. CPU offload is already supported through ComfyUI's built-in model management — no custom node is needed.

3. **The PR's monkey-patch approach (`encode_token_weights` replacement) is unnecessary** when ComfyUI's existing offload infrastructure handles this case.

#### Conclusion

Not adopted. The `nodes/te_offload/` package and its `__init__.py` registration in `__init__.py` are excluded from the merge.

---

### Adoption 5: GPU pack/unpack optimization

**Target files**: `nunchaku_code/lora_qwen.py` (`_awq_lora_forward`, `_apply_lora_to_module` nunchaku branch, `reset_lora_v2`)

#### Current code problem

`pack_lowrank_weight` / `unpack_lowrank_weight` execute on CPU. When nunchaku module's `proj_down.data` / `proj_up.data` are on CUDA:

1. CUDA → CPU tensor copy (via paged staging buffer)
2. CPU unpack / cat / pack
3. CPU → CUDA write-back

This CPU paging stall accumulates across many modules (Qwen Image has dozens of transformer blocks × attn/mlp).

#### PR improvement

Adds `_get_compute_device()`, preferring `comfy.model_management.get_torch_device()`:

```python
def _get_compute_device() -> torch.device:
    try:
        import comfy.model_management as _mm
        return _mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

In `_awq_lora_forward`:

```python
compute_device = _get_compute_device()
pd_gpu = pd.to(compute_device)
pu_gpu = pu.to(compute_device)
A_gpu = A.to(compute_device)
B_gpu = B.to(compute_device)

pd_unpacked = unpack_lowrank_weight(pd_gpu, down=True)
pu_unpacked = unpack_lowrank_weight(pu_gpu, down=False)
# ... cat, pack on compute_device ...
module.proj_down.data = packed_down.to(pd.device)  # restore to original device
module.proj_up.data = packed_up.to(pu.device)
```

`reset_lora_v2` applies the same optimization.

#### Effects

- **Reduces**: CPU paging stall. pack/unpack completes on CUDA.
- **Reduces**: Number of CPU ↔ GPU tensor round-trips.

#### Relation to md docs

- **Same direction as** `PERFORMANCE_OPTIMIZATION_PLAN.md` item 2 ("batch device transfers"). The plan was "cat on CPU then one transfer"; the PR is "complete pack/unpack on GPU". Same goal (reduce device transfers).

#### Concerns

- The nunchaku branch of `_apply_lora_to_module` itself does **not** have this optimization. A/B are passed as `all_A.append(A.to(dtype=target_dtype))` (staying on CPU), then `torch.cat([pd, A], dim=0)` combines CPU `pd` with CPU `A`. The PR's A/B pre-transfer optimization (skipping `target_device` in the nunchaku branch) and the `_apply_lora_to_module` GPU pack/unpack optimization are **not fully consistent**. Must verify during adoption.

---

### Adoption 6: `set_lora_strength_v2` `base_rank` guard

**Target file**: `nunchaku_code/lora_qwen.py`

#### Current code problem

```python:1562:1564:nunchaku_code/lora_qwen.py
    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None or info.get("appended", 0) <= 0:
            continue

        base_rank, appended = info["base_rank"], info["appended"]
```

AWQ modulation slots (`img_mod.1` / `txt_mod.1`) are registered with `type: "awq_w4a16"` and do not have `base_rank`. Accessing `info["base_rank"]` raises `KeyError`.

#### PR improvement

```python
        if module is None or info.get("appended", 0) <= 0 or "base_rank" not in info:
            continue
```

#### Effects

- **Fixes**: `KeyError` on AWQ modulation slots
- **Safety**: nunchaku/linear slots have `base_rank` and are processed as before

#### Relation to md docs

- No direct mention in existing md docs. Supplements the AWQ modulation handling recorded in `V2.4.0_V1_RENAME_COMPLETE_EXPLANATION.md`.

---

### Adoption 7: Dynamic UI `save_precompiled_lora` widget support

**Target files**: `js/z_qwen_lora_dynamic.js`, `js/z_qwen_lora_dynamic_v3.js`

#### Current code problem

V1/V3 dynamic LoRA stack nodes' `onConfigure` caches and restores the `apply_awq_mod` widget, but if `save_precompiled_lora` widget is added, it would be missing on node redraw.

#### PR improvement

```javascript
// Cache widget
const savePrecompiledWidget = all.find(w => w.name === "save_precompiled_lora");
if (savePrecompiledWidget) {
    node.cachedSavePrecompiled = savePrecompiledWidget;
}

// Restore in onConfigure
if (node.cachedSavePrecompiled) {
    this.widgets.push(node.cachedSavePrecompiled);
}

// Reflect in size calculation
const SAVE_PRECOMPILED_H = node.cachedSavePrecompiled ? 30 : 0;
const targetH = HEADER_H + CPU_OFFLOAD_H + APPLY_AWQ_MOD_H + SAVE_PRECOMPILED_H + (count * SLOT_H) + PADDING;
```

#### Effects

- **Fixes**: `save_precompiled_lora` widget missing / size mismatch
- **Cosmetic**: Node height is computed correctly, UI doesn't break

---

### Adoption 8: Logging unification — `print()` removal and Key Diffusion gating

**Target files**: `nodes/lora/qwenimage.py`, `nodes/lora/qwenimage_v1.py`, `nodes/lora/qwenimage_v2.py`, `nodes/lora/qwenimage_v3.py`, `nodes/lora/zimageturbo_v2.py`, `nodes/lora/zimageturbo_v3.py`, `nodes/lora/zimageturbo_v4.py`

**Status**: **Already implemented and pushed** (commit `33ba82e`, 2026-07-10)

#### Current code problem (before this commit)

Three categories of uncontrolled logging existed:

1. **Startup `[DEBUG]` print** — 6 files (qwenimage, qwenimage_v1, qwenimage_v2, qwenimage_v3, zimageturbo_v2, zimageturbo_v3) emitted 3 `print()` lines each at module import time (ComfyUI startup). These were temporary debug leftovers for import-path resolution and were never gated by any env var.

2. **Key Diffusion `print()`** — 4 files (qwenimage × 2 blocks, qwenimage_v2, qwenimage_v3) used `print()` for per-key mapping logs (`Key: ... -> Mapped to: ... (Group: ...)`). These were inside `if NUNCHAKU_LOG_ENABLED:` guards, so `nunchaku_log=0` suppressed them, but `print()` bypasses the logger framework entirely — no flush control, no level filtering, console freeze risk.

3. **Key Diffusion `logger.info` without guard** — `zimageturbo_v3.py` and `zimageturbo_v4.py` used `logger.info()` for Key Diffusion but did **not** gate with `NUNCHAKU_LOG_ENABLED`. These would output even when `nunchaku_log=0`.

#### Implemented fix

| File | Change |
|------|--------|
| `qwenimage.py` | Removed 3 startup `print()` lines; converted 2 Key Diffusion blocks (Loader + Stack) from `print()` to `logger.info`/`logger.warning`; kept inside `if NUNCHAKU_LOG_ENABLED:` |
| `qwenimage_v1.py` | Removed 3 startup `print()` lines |
| `qwenimage_v2.py` | Removed 3 startup `print()` lines; converted 1 Key Diffusion block from `print()` to `logger.info`/`logger.warning`; kept inside `if NUNCHAKU_LOG_ENABLED:` |
| `qwenimage_v3.py` | Removed 3 startup `print()` lines; converted 1 Key Diffusion block from `print()` to `logger.info`/`logger.warning`; kept inside `if NUNCHAKU_LOG_ENABLED:` |
| `zimageturbo_v2.py` | Removed 3 startup `print()` lines |
| `zimageturbo_v3.py` | Removed 3 startup `print()` lines; added `NUNCHAKU_LOG_ENABLED` import; gated Key Diffusion block with `and NUNCHAKU_LOG_ENABLED` |
| `zimageturbo_v4.py` | Added `NUNCHAKU_LOG_ENABLED` import; gated Key Diffusion block with `if idx == 0 and NUNCHAKU_LOG_ENABLED:` |

#### Resulting log policy

| Category | `nunchaku_log=0` (default) | `nunchaku_log=1` |
|----------|---------------------------|------------------|
| Status logs (`[LoRA Stack Status]`, `🔍`, `✅`, `📦`, `🔧`, `Composing`, `LoRA Format Detection`, `Sampled LoRA composition complete`, `Total LoRAs`, etc.) | **Always output** | Always output |
| Key Diffusion per-key mapping (`Key: ... -> Mapped to: ...`) | Suppressed | Output |
| Startup `[DEBUG]` path-verification print | Removed entirely | Removed entirely |

#### Effects

- **All `print()` calls eliminated** from `nodes/lora/` — zero remaining
- **Key Diffusion** gated behind `nunchaku_log` env var across all 7 node files
- **Status logs** remain always-on per owner's policy ("ユーザーが何と言おうと出す")
- **Console freeze** risk from `print()` eliminated (logger has flush control)
- **`nunchaku_log=0`** (default) suppresses only per-key mapping details; status logs still flow

#### Relation to md docs

- **Supersedes** the policy recorded in `KEY_DIFFUSION_RESTORE_QI_LORA_LOADER.md` ("QI side outputs via `print()`"). All nodes now use `logger.info`/`logger.warning` uniformly, matching the `zimageturbo_v4.py` pattern.
- **Complementary to** `PERFORMANCE_OPTIMIZATION_PLAN.md` (logging is a separate axis from performance optimization, but both reduce per-generation CPU overhead).

---

## Overwrite (keep current v2.5.1) — 6 items

These are rebase-omission artifacts where the PR's working copy was stale. They must be overwritten with current v2.5.1 code:

1. **Version `2.5.1`** (`pyproject.toml` + `__init__.py`) — PR downgrades to 2.4.0/2.4.3
2. **`Krea2ControlNetLoraLoader`** import & registration — PR removes it
3. **`NunchakuQwenImageDiffsynthControlnet`** import & registration — PR removes it
4. **`apply_qwen_image_apply_rotary_emb_compat`** patch (`patches/nunchaku_patch.py`, 122 lines) — PR deletes it
5. **`.gitignore`** entries for `.cursor/`, `反省文*`, `backups/`, `scripts/` — PR removes them
6. **README** language switcher table, Diffsynth/Krea2 sections, Latest release v2.5.1 URL — PR reverts to v2.4.2

---

## Additional fixes required during adoption (not in PR)

### Additional-1: Restore `_cached_first_lora_state_dict` ★Most critical

The optimization restored in `LOST_OPTIMIZATION_v227_v230.md` does not exist in the PR's `compose_loras_v2`. When the precompiled cache is invalid (default), the first LoRA is read twice. **Must merge into PR's `compose_loras_v2` Section 3-4 when adopting.**

### Additional-2: Apply `IS_CHANGED` `model` removal to `qwenimage_v2.py`

The PR applies the fix to `qwenimage.py` and `qwenimage_v3.py` but misses `qwenimage_v2.py`. Must apply the same fix when adopting.

### Additional-3: Remove `reset_lora_v2` dead 1st loop

```python:1579:1582:nunchaku_code/lora_qwen.py
    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None:
            continue
```

This loop does nothing. The 2nd loop is the real body. Present in both PR and current. Remove during adoption.

---

## File-by-file adoption matrix

| # | File | Action | Details |
|---|------|--------|---------|
| 1 | `nodes/te_offload/__init__.py` | **Reject** | TE V2 offload — referenced loaders don't exist; CPU offload already available via ComfyUI |
| 2 | `nodes/te_offload/nunchaku_te_v2.py` | **Reject** | Same — references non-existent `NunchakuQwenImageEditEncoderLoader` etc. |
| 3 | `nunchaku_code/lora_cache.py` | **Adopt fully** | New precompiled cache module |
| 4 | `__init__.py` | **Partial adopt** | Add TE_V2_NODES/NAMES **excluded**; keep Krea2/Diffsynth registration; keep `__version__ = "2.5.1"` |
| 5 | `pyproject.toml` | **Overwrite** | Keep `version = "2.5.1"` |
| 6 | `.gitignore` | **Overwrite** | Keep `.cursor/`, `反省文*`, `backups/`, `scripts/` ignores |
| 7 | `README.md` | **Overwrite** | Keep language switcher, Diffsynth/Krea2 sections, v2.5.1 URL |
| 8 | `patches/nunchaku_patch.py` | **Overwrite** | Keep `apply_qwen_image_apply_rotary_emb_compat` (122 lines) |
| 9 | `js/z_qwen_lora_dynamic.js` | **Adopt** | `save_precompiled_lora` widget cache/size |
| 10 | `js/z_qwen_lora_dynamic_v3.js` | **Adopt** | Same as above |
| 11 | `nodes/lora/qwenimage.py` | **Partial adopt** | `IS_CHANGED` model removal + `save_precompiled_lora` optional + 3-tuple. **Logging already unified** (commit `33ba82e`) |
| 12 | `nodes/lora/qwenimage_v1.py` | **Adopt** | `save_precompiled_lora` optional + 3-tuple. **Startup print removed** (commit `33ba82e`) |
| 13 | `nodes/lora/qwenimage_v2.py` | **Partial adopt + fix** | `save_precompiled_lora` optional + 3-tuple; **apply `IS_CHANGED` model removal (PR missed this)**. **Logging already unified** (commit `33ba82e`) |
| 14 | `nodes/lora/qwenimage_v3.py` | **Adopt** | `IS_CHANGED` model removal + `save_precompiled_lora` + 3-tuple. **Logging already unified** (commit `33ba82e`) |
| 15 | `nunchaku_code/lora_qwen.py` | **Partial adopt** | `_get_compute_device`, GPU pack/unpack, `compose_loras_v2` cache args, `set_lora_strength_v2` guard, `reset_lora_v2` `_applied_loras` delattr; **restore `_cached_first_lora_state_dict`**; remove dead 1st loop |
| 16 | `wrappers/qwenimage.py` | **Adopt** | `_has_ever_had_loras` flag, `is_lora_manager` guard, `self.model._applied_loras` sharing, cache args to `compose_loras_v2`, 3-tuple support |
| 17 | `wrappers/qwenimage_v2.py` | **Adopt** | `self.model._applied_loras` sharing, cache args to `compose_loras_v2_v2`, 3-tuple support |

---

## Cross-reference with md/ documents

| md document | Relationship |
|---|---|
| `PERFORMANCE_OPTIMIZATION_PLAN.md` | Item 1 (lru_cache) → superseded by precompiled cache (Adoption 3). Item 2 (batch device transfer) → same direction as GPU pack/unpack (Adoption 5). Item 3 (AWQ forward `.to()` skip) → different location, same goal, coexists. Item 4 (`_get_module_by_name` cache) → not in PR, remains as future plan. |
| `LOST_OPTIMIZATION_v227_v230.md` | `_cached_first_lora_state_dict` restoration must be merged into PR's `compose_loras_v2` (Additional-1). |
| `PR28_FIX_EXPLANATION.md` | PR #28 added `cpu_offload` default to `IS_CHANGED`; PR #52 removes `model` entirely — evolution (Adoption 2). |
| `KEY_DIFFUSION_RESTORE_QI_LORA_LOADER.md` | **Superseded** — all nodes now use `logger.info`/`logger.warning` uniformly (commit `33ba82e`). Per-key mapping gated behind `nunchaku_log` env var. |
| `QWEN_IMAGE_CONTROLNET_AND_GETATTR_FIX.md` | Different location, same bug family as Adoption 1. |
| `QWEN_IMAGE_APPLY_ROTARY_EMB_COMPAT_FIX.md` | Must keep (overwrite item 4). |
| `V2.4.0_V1_RENAME_COMPLETE_EXPLANATION.md` | AWQ modulation handling supplemented by Adoption 6. |
| `COMFYUI_0.4.0_*`, `COMFYUI_0.7.0_*`, `COMFYUI_PR_PROPOSAL`, `MGPU_MM_LOG_*` | Unrelated — PR doesn't touch these areas. |
| `ZIMAGE_SVDQ_LAZY_LINEAR_AND_POP_DEFAULT_FIX.md` | PR preserves — no conflict. |
| `pr48_peft_lora_format_fix.md` | Unrelated — PR doesn't touch `_detect_lora_format`. |
| `tech.md` | PR doesn't touch `_execute_model` — no conflict. |
| `DIFFSYNTH_OFFICIAL_SUPPORT_EXPLANATION`, `KREA2_*`, `ZIMAGETURBO_CONTROLNET_FIX` | PR deletes; we keep (overwrite items 2-3, 6-7). |
| `V2.0_ROOT_CAUSE_ANALYSIS`, `V3/V4_DEVELOPMENT`, `UPGRADE_GUIDE`, `technical_explanation`, `installation` | Unrelated. |

---

## Audit anchors

| Item | Value |
|---|---|
| PR | [#52](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/52) |
| Merge-base | `14f1e32` (v2.5.1 main HEAD) |
| PR head | `3b50696` |
| Merge state | CLEAN / MERGEABLE |
| Current main version | `2.5.1` (`pyproject.toml` + `__init__.py`) |
| PR version (rejected) | `2.4.0` / `2.4.3` (mismatched) |
| Files changed | 17 (+1465 / -364) |
| Adoption items | 7 features (8 in PR, 1 rejected) |
| Overwrite items | 6 (rebase-omission rollbacks) |
| Additional fixes | 3 (not in PR, required during adoption) |
| Logging unification | **Done** — commit `33ba82e` (7 files, `print()` eliminated, Key Diffusion gated behind `nunchaku_log`) |

---

*Document created as the merge plan for PR #52. All adoption decisions are based on cross-reference with existing md/ technical documents and current v2.5.1 main code.*
