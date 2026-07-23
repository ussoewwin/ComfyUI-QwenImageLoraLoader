<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.5.3.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## Summary

**v2.5.3** partially adopts [PR #52](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/52) (*feat: GPU compile acceleration, atomic cache writes, and V2 TE loaders with CPU offload*, author [Sniper199999](https://github.com/Sniper199999)).

The PR could not be merged with one-click GitHub Merge (`mergeable=CONFLICTING`). Valuable LoRA performance and correctness patches were taken in; TE V2 CPU-offload loaders were **rejected**; v2.5.0–v2.5.2 features that the PR would have rolled back were kept; and follow-up fixes required by the merge plan were applied after intake.

Full technical analysis and decision matrix:

- [md/PR52_MERGE_PLAN.md](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/md/PR52_MERGE_PLAN.md)

Merge commit: `2e15390`  
Follow-up: `b683f8a`  
Tag: `v2.5.3` (`74552a8` version bump + changelog)

---

## What was adopted (from PR #52)

### 1. Bypass-path LoRA corruption fix (`_has_ever_had_loras`)

**Files:** `wrappers/qwenimage.py` (and related wrapper paths)

ComfyUI bypass can route through a wrapper that previously had LoRAs cleared (`loras=[]`) while still sharing the same transformer instance as the real LoRA-managing wrapper. The old dirty check could call `reset_lora_v2` on that shared transformer and wipe the active LoRA stack. After bypass was turned off again, the manager wrapper often skipped recompose because `_applied_loras` still matched the desired list — LoRA appeared to vanish intermittently.

**Adopted fix:** monotonic `_has_ever_had_loras`. Only wrappers that have ever managed LoRA may reset / recompose the shared transformer. Bypass shells that never owned LoRA no longer mutate the shared model.

### 2. Remove `model` from `IS_CHANGED`

**Files:** `nodes/lora/qwenimage.py`, `qwenimage_v3.py`, and (post-merge) `qwenimage_v2.py`

ComfyUI’s cache calls `IS_CHANGED` **without** MODEL inputs. A `model` parameter caused `TypeError` → ComfyUI’s `NaN` “always dirty” sentinel → **LoRA re-compose on every queue run** even when nothing changed (large cost for 500MB–2GB LoRAs).

**Adopted fix:** drop `model` from the signature; hash only LoRA-relevant controls (`lora_name`, strength, `cpu_offload`, `save_precompiled_lora`, etc.).

### 3. Disk-persistent LoRA precompiled cache

**Files:** new `nunchaku_code/lora_cache.py`; wired from `nunchaku_code/lora_qwen.py` and LoRA nodes

When the same LoRA file is reused, classify + fuse no longer need to re-run from scratch every generation:

- Cache directory under ComfyUI models (`SVDQLora/`)
- Collision-safe paths (parent-dir hash)
- Atomic writes (`tmp` + `os.replace`)
- mtime sidecar invalidation when the source LoRA file changes
- B tensors stored unscaled so the strength slider remains correct at apply time

Optional UI: `save_precompiled_lora` widget (see item 7).

### 4. GPU pack / unpack for nunchaku LoRA apply

**File:** `nunchaku_code/lora_qwen.py`

`pack_lowrank_weight` / `unpack_lowrank_weight` / `cat` for nunchaku proj_down/proj_up now prefer the Comfy compute device (CUDA) instead of bouncing through CPU paging for every module. Same idea applied on the reset path.

### 5. `set_lora_strength_v2` `base_rank` guard

**File:** `nunchaku_code/lora_qwen.py`

AWQ modulation slots (`img_mod.1` / `txt_mod.1`) have no `base_rank`. Accessing `info["base_rank"]` raised `KeyError`. Slots without `base_rank` are skipped safely.

### 6. Dynamic UI support for `save_precompiled_lora`

**Files:** `js/z_qwen_lora_dynamic.js`, `js/z_qwen_lora_dynamic_v3.js`

Cache / restore / height accounting for the `save_precompiled_lora` widget so it does not disappear or mis-size after redraw/`onConfigure`.

---

## What was rejected

### TE V2 CPU-offload loaders (Adoption 4)

**Excluded:** `nodes/te_offload/` and any `__init__.py` registration for those nodes.

Reasons (see merge plan for detail):

1. Referenced Qwen TE loader classes do not exist in installed ComfyUI-nunchaku / ComfyUI core (wrong import targets).
2. Qwen Image text encoders already go through ComfyUI’s CLIP / `ModelPatcher` offload path.
3. Extra encode monkey-patches are unnecessary for this use case.

---

## Conflict resolution / overwrite (kept current main)

PR head still carried v2.4.x residue. After conflict-resolved intake, these **stayed on main’s v2.5.2 shape**:

| Area | Decision |
|------|----------|
| `__init__.py` version + node map | Keep 2.5.x version path; keep Krea2 / Diffsynth ControlNet registration; **do not** register TE V2 |
| `patches/nunchaku_patch.py` | Keep rotary `apply_rotary_emb` compat **and** `suppress_torch_preimport_warning` (v2.5.2) |
| `nodes/lora/qwenimage_v3.py` | Synthesize PR `IS_CHANGED` / precompiled UI with main’s logging unification |
| `.gitignore` | Keep `.cursor/`, `backups/`, `scripts/`, and other local-only entries |

Logging unification from earlier main (`print()` removal / Key Diffusion gated by `NUNCHAKU_LOG_ENABLED`) was preserved across node files.

---

## Post-merge additional fixes (`b683f8a`)

Required by the merge plan; not present in the original PR head:

1. **Restore `_cached_first_lora_state_dict` in `compose_loras_v2`**  
   When precompiled cache is off (default), the first LoRA must not be read from disk twice (restore the v2.2.7 / v2.4.4 first-LoRA reuse optimization).

2. **`qwenimage_v2.py` `IS_CHANGED`**  
   Same `model` removal as V1/V3; also stop hashing `str(model)` (object repr changes every run and forced re-execution).

---

## Practical effects for users

- More stable LoRA when toggling **bypass** on LoRA nodes
- Faster repeated generations when LoRA inputs are unchanged (`IS_CHANGED` no longer always dirty)
- Optional **precompiled LoRA cache** for large LoRAs (2nd+ generation savings)
- Lower CPU↔GPU thrash during nunchaku pack/unpack
- No new TE V2 offload nodes (rejected on purpose)
- v2.5.0–v2.5.2 features (Krea2 depth ControlNet LoRA, Diffsynth ControlNet, rotary compat, torch-preimport warning suppression) remain intact

---

## Credits

- PR author: [Sniper199999](https://github.com/Sniper199999) — [PR #52](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/52)
- Merge analysis / adoption matrix: [PR52_MERGE_PLAN.md](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/md/PR52_MERGE_PLAN.md)

---

## Upgrade notes

1. Update / reload this custom node to **v2.5.3** (ComfyUI Manager reads `__init__.py` `__version__`).
2. Fully restart ComfyUI after update.
3. Existing workflows keep working. New optional widget: `save_precompiled_lora` on supported dynamic LoRA nodes when you want disk cache writes.
4. If you previously saw intermittent LoRA loss after bypass toggles, re-test with this build.
