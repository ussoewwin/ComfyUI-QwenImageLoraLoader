<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.4.3.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

This document explains, a crash that can occur when loading Nunchaku-quantized Z-Image (Lumina2 / NextDiT) via **ComfyUI-nunchaku**: **what the error is, why it happens, the underlying causes, why ComfyUI-QwenImageLoraLoader absorbs the fix, which files change, and what the added code does**.



---



## Target error (typical user report)



### Exception message



```text

AttributeError: 'NoneType' object has no attribute 'dtype'

```



### Stack trace essentials (representative)



Execution enters from ComfyUI’s node pipeline and fails in **ComfyUI-nunchaku Z-Image model build → Nunchaku `SVDQW4A4Linear.from_linear`**.



```text

File ".../ComfyUI/execution.py", line 525, in execute

  ...

File ".../ComfyUI-nunchaku/nodes/models/zimage.py", line 215, in load_model

  model = _load(sd, metadata=metadata)

File ".../ComfyUI-nunchaku/nodes/models/zimage.py", line 148, in _load

  model = model_config.get_model(patched_sd, "", torch_dtype=torch_dtype)

File ".../ComfyUI-nunchaku/model_configs/zimage.py", line 71, in get_model

  patch_model(out.diffusion_model, ...)

File ".../ComfyUI-nunchaku/models/zimage.py", line 320, in patch_model

  _patch_transformer_block(diffusion_model.layers)

File ".../ComfyUI-nunchaku/models/zimage.py", line 317, in _patch_transformer_block

  block.attention = ComfyNunchakuZImageAttention(block.attention, **kwargs)

File ".../ComfyUI-nunchaku/models/zimage.py", line 133, in __init__

  self.qkv = SVDQW4A4Linear.from_linear(orig_attn.qkv, **kwargs)

File ".../site-packages/nunchaku/models/linear.py", line 152, in from_linear

  torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)

                                            ^^^^^^^^^^^^^^^^^^^

AttributeError: 'NoneType' object has no attribute 'dtype'

```



**Where it actually breaks:** `linear.weight` is `None`, but the code reads `linear.weight.dtype`.



---



## Symptom summary



| Item | Detail |

|------|--------|

| **When** | While the Z-Image (Nunchaku DiT) loader builds the model and `patch_model` swaps Attention / FF to SVDQ layers |

| **Where** | `nunchaku.models.linear.SVDQW4A4Linear.from_linear` (and ComfyUI-nunchaku’s `fuse_to_svdquant_linear` with the same pattern) |

| **Environment** | **`disable_weight_init.Linear`** used when **Windows** and ComfyUI **AIMDO (memory optimization)** are on — keeps weights **`None` until `state_dict` is applied |

| **What you see** | The `AttributeError` above during workflow execution |



---



## Call path (logical flow)



### 1. ComfyUI-nunchaku Z-Image load (outline)



Typical order (implementation gist):



1. Load `state_dict`  

2. Call `NunchakuZImage.get_model(patched_sd, "", torch_dtype=...)`  

3. **`super().get_model(...)`** builds `model_base.Lumina2` → **NextDiT is constructed** (weights may **not yet be on each `Linear`**)  

4. **`patch_model(diffusion_model, ...)`** replaces e.g. `JointAttention` with **`ComfyNunchakuZImageAttention`**  

5. Inside that constructor, **`SVDQW4A4Linear.from_linear(orig_attn.qkv, **kwargs)`** runs  

6. **After that** (depending on loader flow), `load_model_weights` etc. fills weights  



So if **“when quantized modules are patched”** and **“when ComfyUI’s `Linear` gets real weights”** diverge, **`weight is None` at patch time** is possible.



### 2. ComfyUI core: meaning of `disable_weight_init.Linear`



In `comfy/ops.py`, **`disable_weight_init.Linear`**, when **Windows and `aimdo_enabled`**, roughly does:



- Skips normal `torch.nn.Linear` `super().__init__(...)`; only `torch.nn.Module.__init__`  

- Initial **`self.weight = None`**, **`self.bias = None`**  

- Real **`Parameter`**s are assigned in **`_load_from_state_dict`**  



**Design intent (summary):** allocating dummy weight tensors with `torch.empty` can spike commit charge on Windows and destabilize the system. ComfyUI defers / zero-copies from `state_dict` instead.



So **`weight is None` is not a bug** under those conditions — it is **intended ComfyUI behavior**.



### 3. Nunchaku: what `from_linear` assumed



`SVDQW4A4Linear.from_linear` is conceptually written to:



- Read **in/out features, bias, dtype, device** from an existing `nn.Linear`  

- Build a same-shaped quantized placeholder `SVDQW4A4Linear` (trained weights arrive later via `state_dict`)  



If it **implicitly assumes `linear.weight` is always non-`None`**, it clashes with ComfyUI’s deferred `Linear`.



---



## What was really wrong (two layers)



This is best understood as **composite**, not a single root cause.



### Layer A: Python semantics — `dict.pop(key, default)` always evaluates the default



The problematic line (**typical pre-fix nunchaku shape**):



```python

torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)

```



Many developers intuit:



> “If `torch_dtype` is in `kwargs`, the default isn’t used, so `linear.weight.dtype` is never evaluated.”



**In Python that is false.** **All argument expressions are evaluated before the call.** Therefore:



- Even if **`"torch_dtype": torch.bfloat16` is in `kwargs`**  

- **`linear.weight.dtype` is still evaluated**  



So:



- When **`linear.weight is None`** → **`None.dtype` → `AttributeError`**  

- Even if the caller **passed `torch_dtype=` correctly** → **same `AttributeError`** (the real reason it “crashes despite kwargs”)



**Language takeaway:** using `pop`’s default as a “lazy fallback” is **wrong**. If the fallback is expensive, has side effects, or can fail, use **`if "k" in kwargs`** or **`try/except KeyError`**.



### Layer B: Ordering — ComfyUI deferred `Linear` vs ComfyUI-nunchaku “patch first”



Even after fixing layer A, **`linear.weight is None`** still leaves **where to get dtype / device**.



- ComfyUI-nunchaku often does **`patch_model` inside `get_model`, then `load_model_weights`**  

- At the **`patch_model` instant**, deferred ComfyUI `Linear` may still have **`weight is None`**  



**Integration takeaway:** ComfyUI’s **“Linear may have no weight at patch time”** did not match Nunchaku reading **dtype/device from weight**.  

Plus **layer A’s `pop` bug** meant **callers passing `torch_dtype` still crashed**.



---



## Responsibility split (what upstream “should” fix)



| Area | Closest component | Notes |

|------|----------------|------|

| Deferred `Linear` on Windows + AIMDO | **ComfyUI core** | Deliberate optimization; behavior should be documented as spec |

| `from_linear` using `pop(..., linear.weight.dtype)` | **Nunchaku (library)** | Wrong for Python semantics; can be written safely when `weight is None` |

| Order of `patch_model` vs `load_model_weights` | **ComfyUI-nunchaku** | Design choice: patch without weights vs reorder |

| Guaranteed quick fix in every user env | **Nobody** | Waiting on PRs/releases/review blocks users for a long time |



**Absorbing the fix in this node pack** matches the pragmatic pattern in `md/COMFYUI_0.4.0_MODEL_MANAGEMENT_ERRORS.md` and similar notes.



---



## Why ComfyUI-QwenImageLoraLoader absorbs it



### 1. Fits existing architecture



This repo already **`patches/nunchaku_patch.py` on `__init__.py` startup**:



- **Nunchaku Qwen:** **Manual Planar Injection** on `NunchakuQwenImageTransformerBlock.forward` (LoRA)  

- **`sys.modules` scans** for flaky import paths  



This change is a **natural extension: more compatibility under the same `apply_nunchaku_patch()` umbrella**.



### 2. Fewer places for users to touch



- Hand-editing **`site-packages/nunchaku`** → lost on upgrade, hard to reproduce  

- Patching **ComfyUI core** → update conflicts, heavy to explain to others  

- Pinning a **ComfyUI-nunchaku fork** → maintenance cost  



People who install the **LoRA node pack already trust this package** — **centralizing fixes here** minimizes operational cost.



### 3. Load-order handling was already needed



Folder name order can load **this package before ComfyUI-nunchaku**.  

Then **`models.zimage` may be missing from `sys.modules` at startup**.



A **one-shot patch is not enough**; **deferred retries** are required — the same “environment-dependent” pattern this repo already handles.



---



## Files changed



| File | Change |

|------|--------|

| **`ComfyUI-QwenImageLoraLoader/patches/nunchaku_patch.py`** | Replace `SVDQW4A4Linear.from_linear`, dynamic replace of `fuse_to_svdquant_linear`, retry scheduler, wired from `apply_nunchaku_patch` |

| **`ComfyUI-QwenImageLoraLoader/__init__.py`** | Success log text updated to mention lazy-`Linear` mitigation |



**Note:** This repo **does not edit ComfyUI-nunchaku or nunchaku in-tree** (avoid drifting from upstream).



---



## Added / changed code and meaning (by section)



Below is **for explanation**; line numbers may drift — treat the repo source as canonical.



### A. `_torch_device_fallback`



**Role:** When **`linear.weight is None`**, **infer `device`**.



1. Prefer **`comfy.model_management.get_torch_device()`** (aligns with ComfyUI’s chosen inference device)  

2. On failure (tests etc.), simple **`cuda` / `cpu` fallback**  



**Why:** Even if `load_model_weights` fills tensors later, **consistent device at construction** is safer for quantized `Parameter`s.



### B. `_patched_svdqw4a4_from_linear` (replacement for `SVDQW4A4Linear.from_linear`)



**Fixes:**



1. Stop misusing **`pop`**; branch with **`"torch_dtype" in kwargs`** → **no touch of `linear.weight` when `torch_dtype` is provided**  

2. When **`weight is None`**, require **`torch_dtype` in kwargs** and raise a clear **`TypeError`** (debuggability)  

3. Same pattern for **`device`**; if missing, use **`_torch_device_fallback()`**  



**Bound as `classmethod`:** original `from_linear` is a class method, so `SVDQW4A4Linear.from_linear = classmethod(_patched_svdqw4a4_from_linear)`.



### C. `_make_patched_fuse_to_svdquant_linear` and replacement



ComfyUI-nunchaku’s **`fuse_to_svdquant_linear`** (FF w1/w3 fusion for Z-Image) can use the **same `pop(..., weight.dtype)` pattern**.  

Fixing Attention `qkv` / `out` alone can still **fail on FF** in the next block.



**Approach:**



- Pull **`add_comfy_cast_weights_attr`** from the target module and preserve original behavior (dummy attrs for offload compatibility)  

- Only the inner logic uses the **same safe branches as `from_linear`**  



### D. `apply_svdqw4a4_lazy_linear_patch`



- After successful import of **`nunchaku.models.linear.SVDQW4A4Linear`**, **override `from_linear`**  

- **`_svdq_from_linear_patched` flag** prevents double application (reload / re-exec resilience)  



### E. `apply_nunchaku_zimage_fuse_lazy_linear_patch`



**Module identification heuristics** (reduce false positives):



- Has **`fuse_to_svdquant_linear`**  

- Has **`ComfyNunchakuZImageAttention`** (fingerprint for Z-Image nunchaku patch module)  

- Has **`add_comfy_cast_weights_attr`** (helper from the same file)  



Patched functions get **`_qwen_lora_loader_lazy_linear_patch`** to skip re-patching.



### F. `schedule_nunchaku_zimage_fuse_patch_retries`



**Problem:** If this package imports **first** (e.g. alphabetical order), **`zimage` may not be loaded** at startup.



**Mitigation:**



- **Retry `apply_nunchaku_zimage_fuse_lazy_linear_patch()`** after short delays  

- Up to **~6 seconds** (0.25s × 24), then replace **`fuse`** once **ComfyUI-nunchaku** has loaded  



**Trade-off:** background **`threading.Timer`**s. They are **not daemon** but **short-lived and bounded**, so they should not behave like a permanent leak.



### G. `apply_nunchaku_patch` integration



Lazy-`Linear` fixes run **together with** existing **Manual Planar Injection** (lazy path first, then planar in the same function).



**Return value:**



- **`True`** if **Planar patch** succeeded **or** **`from_linear` patch** succeeded  

- **`False`** if neither (e.g. nunchaku not installed)  



### H. `__init__.py` logging



Success logs explicitly mention **Z-Image / SVDQ / ComfyUI deferring `Linear` weights** so you can tell **what activated** from startup logs.



---



## How to verify in startup logs



If ComfyUI startup shows lines like these, **lazy-`Linear` patches are likely active**:



- `Patched SVDQW4A4Linear.from_linear for ComfyUI lazy Linear ...`  

- `Patched fuse_to_svdquant_linear in ...`  



The **`fuse`** patch may appear **a few seconds later** due to **deferred retries**.



---



## Remaining limitations / caveats



1. If **`torch_dtype` is missing from kwargs and `weight is None`**, this patch raises a **`TypeError` on purpose**.  

   - Prefer **loud failure** over **silent breakage**.  

   - As long as ComfyUI-nunchaku’s Z-Image loader keeps passing **`torch_dtype=`**, you should not hit this in normal use.  



2. **Monkey-patching in general:** if upstream changes the same entrypoints (`SVDQW4A4Linear.from_linear`, ComfyUI-nunchaku’s `fuse_to_svdquant_linear`), **unexpected behavioral drift** is possible. See **§4** below.



3. **`sys.modules` heuristics:** a different module could theoretically satisfy the checks; risk is low and mitigated by a **strong Z-Image fingerprint** (`ComfyNunchakuZImageAttention`).  



### 4. If upstream (Nunchaku / ComfyUI-nunchaku) merges an equivalent fix later



This mitigation **replaces** installed nunchaku **classmethods** and ComfyUI-nunchaku **module functions at startup**. The following organizes what happens if upstream **merges a proper fix** (same as “side effects” item **6** elsewhere).



#### 4.1 Double patching?



- **Possible.** `apply_nunchaku_patch()` **still replaces `from_linear` whenever it can**, regardless of upstream version.  

- If the **intent matches** (remove `pop` misuse, branch on `weight is None`), **harm is usually small**: this patch runs first, then logic aligns with a correct upstream path — **unlikely to be a result-changing conflict**.  

- If upstream chooses a **different solution** (e.g. `from_linear` API change, move to another class), **stale replacement here** can cause **only one layer to apply / different exceptions**.



#### 4.2 Upstream API / signature changes



- Examples: extra args on `SVDQW4A4Linear.from_linear(cls, linear, **kwargs)`, no longer a `classmethod`, return type change, delegation to another helper.  

- Then **`_patched_svdqw4a4_from_linear` stuck on an old contract** may **pass startup but fail at runtime** or **silently diverge**.  

- **Mitigation:** **update this patch**, or **remove / disable lazy-`Linear` patching** where upstream is sufficient, or **gate by nunchaku / ComfyUI-nunchaku version** and skip the patch on new releases.



#### 4.3 Maintenance recommendation



- When release notes / issues say **“lazy Linear + `from_linear`” is fixed upstream**, **re-evaluate whether this patch is still needed** for that version onward.  

- To remove it, clean **`patches/nunchaku_patch.py`**, **`apply_nunchaku_patch` call sites**, and **`schedule_nunchaku_zimage_fuse_patch_retries`**, leaving **Planar Injection** only if desired.  

- Record in this doc’s **changelog**: “removed ○○ after upstream absorbed fix” for future readers.



#### 4.4 What users see



- Often **only updating upstream while keeping this package old** still **works** because this patch **overwrites** at load.  

- If something breaks, suspect the **nunchaku / ComfyUI-nunchaku / this package combo**; **update this package** or **disable it temporarily** to reproduce with upstream alone.



---



## Relation to other docs



| Document | Relation |

|----------|----------|

| `COMFYUI_0.4.0_MODEL_MANAGEMENT_ERRORS.md` | Same **premise**: core fixes lag; **user env breaks first** |

| `QWEN_IMAGE_CONTROLNET_AND_GETATTR_FIX.md` | Same **strategy**: **wrappers / patches** to satisfy upstream contracts |

| `ZIMAGETURBO_CONTROLNET_FIX.md` | Z-Image family: **`transformer_options`** etc., **pipeline gap filling** |



This note adds a **compatibility layer** at the intersection of **Z-Image × Nunchaku SVDQ × ComfyUI `Linear` × Python `pop`**.



---



## One-line summary



**ComfyUI can keep `Linear.weight` intentionally `None` under some conditions**, and **Nunchaku’s `from_linear` misused `pop`’s default**, together causing **`None.dtype`**; **ComfyUI-nunchaku’s FF fusion can share the same fragility**; **this repo extends `apply_nunchaku_patch` with monkey-patches and deferred retries so users are not blocked waiting for upstream releases.**



---



## Document changelog



- Initial: document lazy `Linear`, `pop` default evaluation, and absorption in this repo  

- Addendum: **§4** — double patch, API drift, and maintenance when upstream merges the same fix (expanded from side-effects item 6)

