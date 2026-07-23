# PR #52 Merge Plan — Technical Analysis and Adoption Items

## Overview

- **PR**: [#52 feat: GPU compile acceleration, atomic cache writes, and V2 TE loaders with CPU offload](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/52)
- **Author**: [Sniper199999](https://github.com/Sniper199999)
- **PR base (merge-base)**: `14f1e32` (v2.5.1 main HEAD at PR creation time)
- **PR head**: `601c79d` (branch `feature/gpu-perf-and-te-offload-v2`, single commit)
- **Remote main HEAD**: `930fd82` (message `Update README.md`. Tree identical to previous `f82ccd1` at `ba4dedb` — tip-only update with no content diff)
- **Version**: main has **`2.5.2`** in both `pyproject.toml` and `__init__.py`. PR `__init__.py` alone is **`2.4.0`** (must discard)
- **Change size**: +1310 / -330, **15 files** (matches `gh pr view` / `git diff origin/main...601c79d`)
- **Merge state (remote re-check)**: `mergeable=CONFLICTING` / `mergeStateStatus=DIRTY` / `state=OPEN`
- **Measured conflict files (`origin/main` ← `601c79d` `--no-commit` merge)**:
  1. `__init__.py`
  2. `nodes/lora/qwenimage_v3.py`
  3. `patches/nunchaku_patch.py`
- **Policy**: **Keep the adoptable parts of the PR. One-click GitHub Merge is impossible due to conflicts. Bring the PR in with conflict resolution, then overwrite regressions / stale code with the correct current v2.5.2 shape.**
- **Conclusion**: Valuable contributions are mixed with leftover v2.4.x working-copy residue from a missing rebase. The PR cannot be merged as-is. Keep valuable proposals; repair regressions by overwriting with v2.5.2. Conflict resolution centers on the three files in the table below.

---

## Premise

### Why this plan exists

**This plan exists to make sure the adoptable parts of the PR actually survive.** Merging the PR is the intake path for the contribution; the later overwrite is only site preparation so those proposals are not dragged into regressions. The remote is CONFLICTING, so intake requires conflict resolution.

- Valuable proposals (Adoption 1–3, 5–7 below) are the payload of this plan.
- “Overwrite” and “Additional fixes” exist only to protect those adoption items from stale v2.4.x code bundled in the same PR.

### Why the PR is worth keeping

The PR includes contributions worth taking:

- LoRA precompiled cache (disk persistence, atomic writes, mtime invalidation)
- GPU pack/unpack optimization (pack/unpack/cat completed on CUDA)
- `_has_ever_had_loras` bypass-path corruption fix (real bug from ComfyUI’s caching model)
- Removing the `model` argument from `IS_CHANGED` (stops forced re-compose every generation)
- `set_lora_strength_v2` `base_rank` guard, and dynamic UI support for the `save_precompiled_lora` widget

### Why the PR cannot be merged as-is

- **Remote state**: `mergeable=CONFLICTING` / `mergeStateStatus=DIRTY`. GitHub UI Merge will not work. Conflicts are `__init__.py`, `qwenimage_v3.py`, and `nunchaku_patch.py` (measured).
- The PR correctly bases on v2.5.1 `main` (`14f1e32`), but the working copy was still from the v2.4.x era and was not rebased before commit.
- As a result it wrongly rolls back features established in v2.5.0/v2.5.1 (version bump, Krea2/Diffsynth ControlNet registration, rotary compat patch, `.gitignore` entries).
- main has since advanced to **v2.5.2** (tip `930fd82`), including `suppress_torch_preimport_warning`, `transformers_qwen_vl_docstring_patch`, and logging unification (`33ba82e`) that the PR never saw. A bad conflict resolution would delete these.

### Policy (core of this plan)

**Merge the PR as the intake path (conflict resolution required because the remote is CONFLICTING). Then overwrite every regressing / stale section with the correct current v2.5.2 shape so the proposals can survive on a clean tree.**

- “Intake” = accept the contributor’s intent and put adoption items into the tree. The green GitHub Merge button is currently unavailable. CLI / manual conflict-resolved merge (or equivalent intake) is required.
- “Overwrite” = after intake, restore every regressing / stale place to the correct v2.5.2 form. This does not void the PR; it **protects the adopted parts** from stale code shipped with them.
- Goal: keep valuable proposals in the tree, repair regressions, and preserve v2.5.x achievements (including v2.5.2).

---

## Adoption items (7 adopt, 1 reject)

### Adoption 1: `_has_ever_had_loras` bypass-path corruption fix

**Target file**: `wrappers/qwenimage.py`

#### Problem in current code

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

`load_lora()` has a side effect: it permanently rewrites the input model’s `diffusion_model` to `ComfyQwenImageWrapper(loras=[])`. Call that W1. The returned model’s wrapper (W2) holds the real LoRAs and **shares the same `NunchakuQwenImageTransformer2DModel` instance** with W1.

When a LoRA node is bypassed, ComfyUI routes through the input model (W1). On W1’s first forward path:

1. `_applied_loras = None` → `loras_changed = True`
2. `reset_lora_v2(shared_transformer)` → **wipes the LoRA configuration W2 had applied**
3. Next time LoRA is enabled again, W2 sees `_applied_loras == loras` and skips recompose → LoRA silently disappears

Result: intermittent LoRA corruption. LoRA appears to stop at random, briefly recovers when toggling bypass, then breaks again.

#### Improvement from the PR

```python
# __init__ — add flag
self._has_ever_had_loras = False

# forward() — update flag before dirty check
if self.loras:
    self._has_ever_had_loras = True

model_is_dirty = (
    self._has_ever_had_loras and  # Was (or is) a LoRA manager
    not self.loras and            # Currently needs no LoRA
    hasattr(self.model, "_lora_slots") and self.model._lora_slots  # But the model still has LoRA
)

if loras_changed or model_is_dirty or device_changed:
    is_lora_manager = bool(self.loras) or self._has_ever_had_loras
    if is_lora_manager:
        reset_lora_v2(self.model)
    # ...
```

W1 (`_has_ever_had_loras = False`) skips the entire transformer mutation block. Only wrappers that actually manage LoRA may reset / recompose the shared transformer.

#### Effects

- **Fix**: intermittent silent LoRA loss when toggling bypass
- **Fix**: after restoring bypass, W2 skips recompose so LoRA is not reapplied
- **Safety**: `_has_ever_had_loras` is monotonic (once True, stays True). W1 permanently avoids reset

#### Relation to md docs

- `QWEN_IMAGE_CONTROLNET_AND_GETATTR_FIX.md` (v2.4.2 `__getattr__` recursion fix) is a different site, but same bug family: “wrapper wrongly mutates shared transformer”
- `PR28_FIX_EXPLANATION.md` records `IS_CHANGED` fixes. This item is the same category: correct adaptation to ComfyUI’s cache / execution model

---

### Adoption 2: Remove `model` from `IS_CHANGED`

**Target files**: `nodes/lora/qwenimage.py`, `nodes/lora/qwenimage_v3.py`, `nodes/lora/qwenimage_v2.py` (v2 is missing the PR fix — manual apply required)

#### Problem in current code

```python:33:nodes/lora/qwenimage.py
    def IS_CHANGED(s, model, lora_name, lora_strength, cpu_offload="disable", *args, **kwargs):
```

ComfyUI’s cache system calls `IS_CHANGED` without passing `model`. MODEL-type inputs (outputs of other nodes) are not passed. Result:

1. `model` not passed → `TypeError`
2. ComfyUI turns the unhandled exception inside `IS_CHANGED` into `NaN`
3. `NaN` is ComfyUI’s sentinel for “always re-run the node”
4. **LoRA is re-composed on every queue run even when nothing changed**

#### Improvement from the PR

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

Remove `model` from the signature; absorb extras via `**kwargs`. Hash only lora_name, lora_strength, cpu_offload, and save_precompiled_lora.

#### Effects

- **Fix**: eliminate forced LoRA re-compose every generation
- **Reduction**: ~1–3 s/generation for a 500MB LoRA, ~5–10 s for a 2GB LoRA
- **Side effect**: if non-LoRA parameters are unchanged, the hash matches → compose is skipped

#### Relation to md docs

- Evolution of `PR28_FIX_EXPLANATION.md` section ② “Fix 2: Additional Fixes Applied (IS_CHANGED Methods)”. PR #28 only added a `cpu_offload` default and kept `model`. PR #52 removes `model` entirely — a more fundamental fix. Both are compatible.

#### Notes

- **Remote main (`930fd82`) at plan time**: `IS_CHANGED` in `qwenimage.py` / `qwenimage_v2.py` / `qwenimage_v3.py` still had `model` (Adoption 2 not yet applied).
- **PR (`601c79d`)**: `model` removed from `qwenimage.py` and `qwenimage_v3.py`. **`qwenimage_v2.py` still has `model` in the PR** → Additional-2 must apply it manually.
- **Conflict**: `qwenimage_v3.py` content-conflicts on merge (logging unification + Adoption 2). Synthesis policy is in the “Conflict watch during merge” table.

---

### Adoption 3: LoRA precompiled cache

**Target files**: `nunchaku_code/lora_cache.py` (new), `nunchaku_code/lora_qwen.py`, node files

#### Problem in current code

Every `compose_loras_v2` section 4 run:

1. `_load_lora_state_dict_robust(lora_path)` — load safetensors from disk
2. `_classify_and_map_key` — apply ~30 regexes per key
3. `_fuse_*_lora` — fuse A/B tensors
4. Accumulate into `aggregated_weights`
5. `_apply_lora_to_module` — apply to modules

When the same LoRA is reused, steps 1–3 repeat every generation. Cost is significant for 500MB–2GB LoRAs.

#### Improvement from the PR

New module `lora_cache.py` provides:

- `get_cache_dir(comfy_base)` → `ComfyUI/models/SVDQLora/`
- `get_cache_path(lora_path, cache_dir)` → collision-safe filename via parent-dir SHA256[:8]
- `is_cache_valid(lora_path, cache_path)` → file exists + mtime sidecar JSON match
- `save_precompiled(processed_groups, lora_path, cache_path)` → atomic write of `{module_key__A, module_key__B, module_key__alpha}`
- `load_precompiled(cache_path)` → rebuild `{module_key: (A, B, alpha)}`

`compose_loras_v2` gets a cache fast path:

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

- **Speedup**: skip classify + fuse. Large cut in disk I/O and CPU work from the 2nd generation onward
- **Safety**: B tensors stored **unscaled**. Strength is reapplied at inference → strength slider stays compatible
- **Atomic writes**: `tmp + os.replace` prevents corruption on interrupt
- **mtime check**: cache auto-invalidates when the LoRA file is updated
- **Collision avoidance**: same-named LoRAs in different directories do not collide

#### Relation to md docs

- Evolution of `PERFORMANCE_OPTIMIZATION_PLAN.md` item 1 (`lru_cache` on `_load_lora_state_dict_robust`). That plan was session-scoped; the PR’s precompiled cache is disk-persistent and stronger
- Complementary to `LOST_OPTIMIZATION_v227_v230.md` (eliminate double-read of first LoRA). A precompiled-cache hit avoids the double-read problem entirely

#### Concerns

1. No integrity check (hash signature) of cache contents. Low risk locally; tamper risk in shared environments
2. mtime sidecars can false-invalidate after copy/sync. Acceptable in practice
3. **PR `compose_loras_v2` lacks `_cached_first_lora_state_dict`**. When the cache is disabled (default), the first LoRA is read twice. That re-loses the optimization restored in `LOST_OPTIMIZATION_v227_v230.md`. **Must restore on adoption**

---

### Adoption 4: TE V2 CPU offload — REJECTED

**Target files**: `nodes/te_offload/nunchaku_te_v2.py` (new), `__init__.py`

#### PR claim

Adds three V2 loader nodes (`NunchakuQwenImageEditEncoderLoaderV2`, `NunchakuQwenImageTextEncoderLoaderV2`, `NunchakuQwen3TextEncoderLoaderV2`) and claims an `offload_after_encode` toggle moves the text encoder to CPU after encode.

#### Rejection reasons

1. **Referenced loader classes do not exist.** The PR’s `_delegate_load` tries to import `NunchakuQwenImageEditEncoderLoader`, `NunchakuQwenImageTextEncoderLoader`, and `NunchakuQwen3TextEncoderLoader` from `custom_nodes/ComfyUI-nunchaku/nodes/models/qwen_text_encoder.py`. Verified against installed ComfyUI-nunchaku: **that file does not exist**. The real TE module is `nodes/models/text_encoder.py`, which only has `NunchakuTextEncoderLoader` / `NunchakuTextEncoderLoaderV2` (FLUX T5 — not Qwen). The three Qwen-specific classes the PR references exist in **neither ComfyUI-nunchaku nor ComfyUI core**.

2. **CPU offload is already available via ComfyUI.** Qwen Image text encoders load through ComfyUI’s standard CLIP loader pipeline and use `ModelPatcher` for VRAM. Built-in model management already supports CPU offload; a custom node is unnecessary.

3. **The PR’s monkey-patch (`encode_token_weights` swap) is unnecessary.** ComfyUI’s existing offload stack covers this case.

#### Conclusion

Rejected. Exclude the `nodes/te_offload/` package and its `__init__.py` registration from the merge.

---

### Adoption 5: GPU pack/unpack optimization

**Target file**: `nunchaku_code/lora_qwen.py` (`_awq_lora_forward`, nunchaku branch of `_apply_lora_to_module`, `reset_lora_v2`)

#### Problem in current code

`pack_lowrank_weight` / `unpack_lowrank_weight` run on CPU. When nunchaku modules’ `proj_down.data` / `proj_up.data` are on CUDA:

1. CUDA → CPU tensor copy (via paged staging buffer)
2. unpack / cat / pack on CPU
3. CPU → CUDA write-back

This CPU paging stall accumulates across many modules (Qwen Image: dozens of transformer blocks × attn/mlp).

#### Improvement from the PR

Add `_get_compute_device()` preferring `comfy.model_management.get_torch_device()`:

```python
def _get_compute_device() -> torch.device:
    try:
        import comfy.model_management as _mm
        return _mm.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

In the nunchaku branch of `_apply_lora_to_module`:

```python
compute_device = _get_compute_device()
pd_gpu = pd.to(compute_device)
pu_gpu = pu.to(compute_device)
A_gpu = A.to(compute_device)
B_gpu = B.to(compute_device)

pd_unpacked = unpack_lowrank_weight(pd_gpu, down=True)   # CUDA
pu_unpacked = unpack_lowrank_weight(pu_gpu, down=False)  # CUDA
new_proj_down = torch.cat([pd_unpacked, A_gpu], dim=0)    # CUDA
packed_down = pack_lowrank_weight(new_proj_down, down=True)  # CUDA
module.proj_down.data = packed_down.to(pd.device)  # restore to original device
```

Apply the same optimization in `reset_lora_v2`.

#### Effects

- **Reduction**: CPU paging stalls. pack/unpack finishes on CUDA
- **Reduction**: CPU ↔ GPU tensor round trips

#### Relation to md docs

- Same direction as `PERFORMANCE_OPTIMIZATION_PLAN.md` item 2 (“batch device transfers”). The plan said “cat on CPU then transfer once”; the PR says “finish pack/unpack on GPU”. Same goal (fewer device transfers).

#### Data-flow verification (validated)

The PR’s two-stage optimization is **internally consistent**:

| Stage | Where | What | Device |
|------|------|------|--------|
| 1 | `compose_loras_v2` apply loop | `all_A.append(A.to(dtype=target_dtype))` — dtype only; **device intentionally omitted** | CPU (A/B stay on CPU) |
| 2 | `torch.cat(all_A, dim=0)` | concatenate all A tensors | CPU |
| 3 | `_apply_lora_to_module` nunchaku branch | `A_gpu = A.to(compute_device)` → `torch.cat([pd_unpacked, A_gpu], dim=0)` on CUDA → `pack_lowrank_weight` on CUDA → restore via `.to(pd.device)` | CUDA → original device |

Stage 1 intentionally omits `device=target_device` only in the nunchaku branch (only that branch does this). Comment: “~960 redundant CPU→CPU memcpys per generation”. Stage 3 moves A/B with `pd`/`pu` to `compute_device` and runs pack/unpack/cat all on CUDA. The two stages are designed together; no device mismatch.

---

### Adoption 6: `set_lora_strength_v2` `base_rank` guard

**Target file**: `nunchaku_code/lora_qwen.py`

#### Problem in current code

```python:1562:1564:nunchaku_code/lora_qwen.py
    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None or info.get("appended", 0) <= 0:
            continue

        base_rank, appended = info["base_rank"], info["appended"]
```

AWQ modulation slots (`img_mod.1` / `txt_mod.1`) register as `type: "awq_w4a16"` and have no `base_rank`. Accessing `info["base_rank"]` raises `KeyError`.

#### Improvement from the PR

```python
        if module is None or info.get("appended", 0) <= 0 or "base_rank" not in info:
            continue
```

#### Effects

- **Fix**: `KeyError` on AWQ modulation slots
- **Safety**: nunchaku/linear slots that have `base_rank` still process as before

#### Relation to md docs

- No direct mention in existing md. Complements AWQ modulation handling recorded in `V2.4.0_V1_RENAME_COMPLETE_EXPLANATION.md`.

---

### Adoption 7: Dynamic UI `save_precompiled_lora` widget support

**Target files**: `js/z_qwen_lora_dynamic.js`, `js/z_qwen_lora_dynamic_v3.js`

#### Problem in current code

V1/V3 dynamic LoRA stack nodes’ `onConfigure` caches/restores the `apply_awq_mod` widget, but when `save_precompiled_lora` is added it is missing after redraw.

#### Improvement from the PR

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

// Size calculation
const SAVE_PRECOMPILED_H = node.cachedSavePrecompiled ? 30 : 0;
const targetH = HEADER_H + CPU_OFFLOAD_H + APPLY_AWQ_MOD_H + SAVE_PRECOMPILED_H + (count * SLOT_H) + PADDING;
```

#### Effects

- **Fix**: missing / mis-sized `save_precompiled_lora` widget
- **UI**: node height calculates correctly; UI does not break

---

### Adoption 8: Logging unification — remove `print()` and gate Key Diffusion

**Target files**: `nodes/lora/qwenimage.py`, `nodes/lora/qwenimage_v1.py`, `nodes/lora/qwenimage_v2.py`, `nodes/lora/qwenimage_v3.py`, `nodes/lora/zimageturbo_v2.py`, `nodes/lora/zimageturbo_v3.py`, `nodes/lora/zimageturbo_v4.py`

**Status**: **Already implemented and pushed** (commit `33ba82e`, 2026-07-10)

#### Problem before that commit

Three kinds of uncontrolled logging:

1. **Startup `[DEBUG]` print** — 6 files (qwenimage, qwenimage_v1, qwenimage_v2, qwenimage_v3, zimageturbo_v2, zimageturbo_v3) each emitted 3 `print()` lines at module import (ComfyUI startup). Temporary debug leftovers for import-path resolution; no env gate.

2. **Key Diffusion `print()`** — 4 files (qwenimage × 2 blocks, qwenimage_v2, qwenimage_v3) used `print()` for per-key mapping logs (`Key: ... -> Mapped to: ... (Group: ...)`). Inside `if NUNCHAKU_LOG_ENABLED:` so `nunchaku_log=0` suppresses them, but `print()` bypasses the logging framework entirely — no flush control, no level filter, console-stick risk.

3. **Ungated Key Diffusion `logger.info`** — `zimageturbo_v3.py` and `zimageturbo_v4.py` used `logger.info()` for Key Diffusion **without** gating on `NUNCHAKU_LOG_ENABLED`. Still printed with `nunchaku_log=0`.

#### Implemented fix

| File | Change |
|------|--------|
| `qwenimage.py` | Removed 3 startup `print()` lines. Converted Key Diffusion blocks ×2 (Loader + Stack) from `print()` → `logger.info`/`logger.warning`. Kept inside `if NUNCHAKU_LOG_ENABLED:` |
| `qwenimage_v1.py` | Removed 3 startup `print()` lines |
| `qwenimage_v2.py` | Removed 3 startup `print()` lines. Converted Key Diffusion ×1 `print()` → `logger.info`/`logger.warning`. Kept inside `if NUNCHAKU_LOG_ENABLED:` |
| `qwenimage_v3.py` | Removed 3 startup `print()` lines. Converted Key Diffusion ×1 `print()` → `logger.info`/`logger.warning`. Kept inside `if NUNCHAKU_LOG_ENABLED:` |
| `zimageturbo_v2.py` | Removed 3 startup `print()` lines |
| `zimageturbo_v3.py` | Removed 3 startup `print()` lines. Import `NUNCHAKU_LOG_ENABLED`. Gate Key Diffusion with `and NUNCHAKU_LOG_ENABLED` |
| `zimageturbo_v4.py` | Import `NUNCHAKU_LOG_ENABLED`. Gate Key Diffusion with `if idx == 0 and NUNCHAKU_LOG_ENABLED:` |

#### Resulting logging policy

| Category | `nunchaku_log=0` (default) | `nunchaku_log=1` |
|----------|----------------------------|------------------|
| Status logs (`[LoRA Stack Status]`, `🔍`, `✅`, `📦`, `🔧`, `Composing`, `LoRA Format Detection`, `Sampled LoRA composition complete`, `Total LoRAs`, etc.) | **Always on** | Always on |
| Key Diffusion per-key mapping (`Key: ... -> Mapped to: ...`) | Suppressed | Printed |
| Startup `[DEBUG]` path-check print | Fully removed | Fully removed |

#### Effects

- **All `print()` calls removed from `nodes/lora/`** — zero remaining
- **Key Diffusion** gated behind `nunchaku_log` in all 7 node files
- **Status logs** always on by design (independent of user settings)
- Eliminates **console-stick risk from `print()`** (logger has flush control)
- **`nunchaku_log=0`** (default) suppresses only per-key mapping detail. Status logs still flow

#### Relation to md docs

- **Overrides** the policy in `KEY_DIFFUSION_RESTORE_QI_LORA_LOADER.md` (“QI side uses `print()`”). All nodes follow the `zimageturbo_v4.py` pattern with `logger.info`/`logger.warning`
- Complementary to `PERFORMANCE_OPTIMIZATION_PLAN.md` (logging is a separate axis from perf, but both reduce per-generation CPU overhead)

#### Post-merge note (mandatory)

If Adoption 1–3 / 5–7 replace node files with PR versions, **`33ba82e` logging unification can regress**. After merge, re-check `nodes/lora/` for leftover `print(` and Key Diffusion `NUNCHAKU_LOG_ENABLED` gating (Additional-4).

## Overwrite (keep correct current v2.5.2 shape) — 5 items

These are stale v2.4.x leftovers bundled in the PR. Overwrite with current v2.5.2 code so adopted proposals survive on a clean tree.

1. **`__init__.py` `__version__`** — PR downgrades to `2.4.0`. Restore current v2.5.2 `2.5.2`. PR does not touch `pyproject.toml`, so that stays `2.5.2` automatically. Leaving `__init__.py` at `2.4.0` desyncs ComfyUI Manager (reads `__init__.py`) from `pyproject.toml` — always restore `2.5.2` on the same path.
2. **`Krea2ControlNetLoraLoader` import/registration** — PR deletes this. Keep v2.5.2 registration.
3. **`NunchakuQwenImageDiffsynthControlnet` import/registration** — PR deletes this. Keep v2.5.2 registration.
4. **`apply_qwen_image_apply_rotary_emb_compat` patch (`patches/nunchaku_patch.py`)** — PR deletes this. The same file also has v2.5.2 `suppress_torch_preimport_warning`, so overwrite must keep **both** rotary compat and v2.5.2 warning suppression.
5. **`.gitignore`** `.cursor/`, `backups/`, `scripts/`, local-only entries — PR deletes these. Keep v2.5.2 entries.

### Out of scope (auto-kept because PR does not touch them)

- **`pyproject.toml`** — not in PR diff. Stays `2.5.2` with no action. (Previously listed with Overwrite 1; removed. No overwrite needed.)
- **`README.md`** — not in PR diff. v2.5.2 README (language switcher, Diffsynth/Krea2 sections, v2.5.2 release URL) remains. (Previous Overwrite 6 removed. No overwrite needed.)
- **`prestartup_script.py`** — not in PR diff. v2.5.2 docstring-patch and warning-suppression calls are kept automatically. No work.

### Conflict watch during merge (remote measured — mandatory)

`--no-commit` merge of `601c79d` onto `origin/main` (`930fd82`) produced **content conflicts in only these 3 files**. Other PR-changed files (`qwenimage.py` / `v1` / `v2` / `lora_qwen.py` / wrappers / js, etc.) auto-merged, but **auto-merge ≠ adoption policy** (especially logging unification — re-verify with Additional-4).

| Conflict file | Why it conflicts | Resolution |
|---------------|------------------|------------|
| `__init__.py` | PR sets `__version__=2.4.0`, adds TE V2 registration, deletes Krea2/Diffsynth. main has `2.5.2` and keeps Krea2/Diffsynth | **Keep main (v2.5.2) entirely**. Do **not** add TE V2 node registration (Adoption 4 rejected) |
| `patches/nunchaku_patch.py` | PR deletes rotary compat. main has rotary **and** `suppress_torch_preimport_warning` | **Keep main** (rotary + warning suppression). Discard PR’s rotary deletion |
| `nodes/lora/qwenimage_v3.py` | PR removes `model` from `IS_CHANGED` + `save_precompiled_lora`. main has `33ba82e` logging unification (`print()` removal / Key Diffusion gate) | **Synthesize both**: take PR `IS_CHANGED`/`save_precompiled_lora`, keep main logging unification. Do not crush either side |

---

## Additional fixes required on adoption (not in the PR)

### Additional-1: Restore `_cached_first_lora_state_dict` ★ critical

The optimization restored in `LOST_OPTIMIZATION_v227_v230.md` is missing from PR `compose_loras_v2`. When the precompiled cache is disabled (default), the first LoRA is read twice. **Must merge into PR `compose_loras_v2` sections 3–4 on adoption.**

### Additional-2: Also remove `model` from `IS_CHANGED` in `qwenimage_v2.py`

The PR fixes `qwenimage.py` and `qwenimage_v3.py` but misses `qwenimage_v2.py`. Apply the same fix on adoption.

### Additional-3: Delete the dead first loop in `reset_lora_v2`

```python:1579:1582:nunchaku_code/lora_qwen.py
    for name, info in model._lora_slots.items():
        module = _get_module_by_name(model, name)
        if module is None:
            continue
```

This loop does nothing. The second loop is the real work. Present in both PR and current. Delete on adoption.

### Additional-4: Re-verify logging unification (`33ba82e`) after merge ★ mandatory

Replacing node files with PR versions can regress main’s `33ba82e` (`print()` removal / Key Diffusion `NUNCHAKU_LOG_ENABLED` gate). **Remote re-check**: `origin/main` `nodes/lora/` has **0** `print(` (logging unification is maintained on main). After merge completes, always verify:

1. No leftover `print(` under `nodes/lora/`
2. Every Key Diffusion block is gated by `NUNCHAKU_LOG_ENABLED`
3. Startup `[DEBUG]` path-check prints are not reintroduced
4. Especially in conflict-resolved `qwenimage_v3.py`, PR adoption (`IS_CHANGED` / `save_precompiled_lora`) coexists with main logging

If regressing, re-apply `33ba82e`-equivalent changes onto the adoption tree.

---

## Per-file adoption matrix (15 PR-changed files)

| # | File | Policy | Detail |
|---|------|--------|--------|
| 1 | `nodes/te_offload/__init__.py` | **Reject** | TE V2 offload — referenced loaders missing; CPU offload available via ComfyUI |
| 2 | `nodes/te_offload/nunchaku_te_v2.py` | **Reject** | Same — references non-existent `NunchakuQwenImageEditEncoderLoader`, etc. |
| 3 | `nunchaku_code/lora_cache.py` | **Adopt fully** | New precompiled-cache module |
| 4 | `__init__.py` | **Partial (conflict)** | Measured conflict. Resolve by **keeping main entirely** (`2.5.2`, Krea2/Diffsynth). Do not add TE_V2 registration. Discard PR `2.4.0` |
| 5 | `.gitignore` | **Overwrite-keep** | Keep `.cursor/`, `backups/`, `scripts/`, local-only entries as in v2.5.2 (expect auto-merge) |
| 6 | `patches/nunchaku_patch.py` | **Overwrite-keep (conflict)** | Measured conflict. Resolve by **keeping main** (rotary + `suppress_torch_preimport_warning`). Discard PR rotary deletion |
| 7 | `js/z_qwen_lora_dynamic.js` | **Adopt** | Cache/size for `save_precompiled_lora` widget |
| 8 | `js/z_qwen_lora_dynamic_v3.js` | **Adopt** | Same |
| 9 | `nodes/lora/qwenimage.py` | **Partial** | Remove `model` from `IS_CHANGED` + optional `save_precompiled_lora` + 3-tuple. **Logging already unified in `33ba82e` — re-verify with Additional-4 after merge** |
| 10 | `nodes/lora/qwenimage_v1.py` | **Adopt** | Optional `save_precompiled_lora` + 3-tuple. **Startup prints already removed in `33ba82e` — Additional-4** |
| 11 | `nodes/lora/qwenimage_v2.py` | **Partial + fix** | Optional `save_precompiled_lora` + 3-tuple. **Also remove `model` from `IS_CHANGED` (PR miss)**. Logging: Additional-4 |
| 12 | `nodes/lora/qwenimage_v3.py` | **Adopt (conflict)** | Measured conflict. **Must synthesize**: take PR `IS_CHANGED` model removal + `save_precompiled_lora` + 3-tuple; keep main `33ba82e` logging. Additional-4 |
| 13 | `nunchaku_code/lora_qwen.py` | **Partial** | `_get_compute_device`, GPU pack/unpack, `compose_loras_v2` cache args, `set_lora_strength_v2` guard, `reset_lora_v2` `_applied_loras` delattr. **Restore `_cached_first_lora_state_dict`**. Delete dead first loop |
| 14 | `wrappers/qwenimage.py` | **Adopt** | `_has_ever_had_loras` flag, `is_lora_manager` guard, share `self.model._applied_loras`, cache args into `compose_loras_v2`, 3-tuple support |
| 15 | `wrappers/qwenimage_v2.py` | **Adopt** | Share `self.model._applied_loras`, cache args into `compose_loras_v2_v2`, 3-tuple support |

### Auto-kept (outside matrix — PR does not touch)

| File | Reason |
|------|--------|
| `pyproject.toml` | Not in PR diff. Stays `version = "2.5.2"` |
| `README.md` | Not in PR diff. v2.5.2 README remains |
| `prestartup_script.py` | Not in PR diff. Docstring patch / warning-suppression calls auto-kept |

---

## Cross-check against md/ docs

| md doc | Relation |
|--------|----------|
| `PERFORMANCE_OPTIMIZATION_PLAN.md` | Item 1 (lru_cache) → precompiled cache (Adoption 3) is a stronger replacement. Item 2 (batch device transfers) → same direction as GPU pack/unpack (Adoption 5). Item 3 (skip `.to()` in AWQ forward) → different site, same goal, can coexist. Item 4 (`_get_module_by_name` cache) → not in PR; remains future plan |
| `LOST_OPTIMIZATION_v227_v230.md` | Must merge `_cached_first_lora_state_dict` restore into PR `compose_loras_v2` (Additional-1) |
| `PR28_FIX_EXPLANATION.md` | PR #28 added `cpu_offload` default to `IS_CHANGED`. PR #52 removes `model` itself — evolution (Adoption 2) |
| `KEY_DIFFUSION_RESTORE_QI_LORA_LOADER.md` | **Already overridden** — all nodes unified on `logger.info`/`logger.warning` (`33ba82e`). Per-key mapping gated by `nunchaku_log`. Re-verify after merge with Additional-4 |
| `QWEN_IMAGE_CONTROLNET_AND_GETATTR_FIX.md` | Same bug family as Adoption 1, different site |
| `QWEN_IMAGE_APPLY_ROTARY_EMB_COMPAT_FIX.md` | Must keep (Overwrite 4). Also keep v2.5.2 `suppress_torch_preimport_warning` |
| `V2.4.0_V1_RENAME_COMPLETE_EXPLANATION.md` | Adoption 6 complements AWQ modulation handling |
| `COMFYUI_0.4.0_*`, `COMFYUI_0.7.0_*`, `COMFYUI_PR_PROPOSAL`, `MGPU_MM_LOG_*` | Unrelated — PR does not touch these areas |
| `ZIMAGE_SVDQ_LAZY_LINEAR_AND_POP_DEFAULT_FIX.md` | PR keeps — no conflict |
| `pr48_peft_lora_format_fix.md` | Unrelated — PR does not touch `_detect_lora_format` |
| `tech.md` | PR does not touch `_execute_model` — no conflict |
| `DIFFSYNTH_OFFICIAL_SUPPORT_EXPLANATION`, `KREA2_*`, `ZIMAGETURBO_CONTROLNET_FIX` | PR deletes; we keep (Overwrite 2–3). README auto-kept because PR does not touch it |
| `V2.0_ROOT_CAUSE_ANALYSIS`, `V3/V4_DEVELOPMENT`, `UPGRADE_GUIDE`, `technical_explanation`, `installation` | Unrelated |

---

## Audit anchors

| Item | Value |
|------|-------|
| PR | [#52](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/52) |
| Merge-base | `14f1e32` (PR branch point; then-main) |
| Remote main HEAD | `930fd82` (`Update README.md`. Tree identical to `f82ccd1` at `ba4dedb`) |
| PR head | `601c79d` (unchanged) |
| Merge state | **CONFLICTING** / **DIRTY** (`gh pr view` re-check). One-click Merge impossible |
| Measured conflict files (3) | `__init__.py` / `nodes/lora/qwenimage_v3.py` / `patches/nunchaku_patch.py` |
| Current main version | `2.5.2` (`pyproject.toml` + `__init__.py`, both confirmed on origin/main) |
| PR version (discard) | `__init__.py` only `2.4.0` (`pyproject.toml` untouched by PR) |
| Changed file count | **15** (+1310 / -330; matches `gh` files and `git diff origin/main...601c79d`) |
| Adoption items | 7 features (of 8 in the PR; 1 rejected) |
| Overwrite-keep items | **5** (stale rollback. `pyproject.toml` / `README.md` out of scope) |
| Additional fixes | **4** (outside PR. Additional-4 = post-merge logging re-verify) |
| Logging unification | **Already on main** — commit `33ba82e` (7 files, `print()` removed, Key Diffusion gated by `nunchaku_log`). After merge: Additional-4 |

---

*This document is the merge plan for PR #52. Adoption decisions are based on cross-checking existing md/ technical docs against current v2.5.2 main (remote tip `930fd82`). Remote was CONFLICTING/DIRTY. The goal is not overwrite for its own sake, but to keep adoptable proposals alive on the current tree.*
