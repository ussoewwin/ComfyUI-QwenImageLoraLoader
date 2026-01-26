# PR #48: Improved PEFT Format LoRA Detection

> **Merged Date**: 2026-01-26  
> **Proposed by**: [avan06](https://github.com/avan06)  
> **Commit**: `d8f819bc0e1826861dd41d0b6dbc12080ac9db23`

---

## 1. Proposal Overview

### 1.1 Summary

PR #48 fixes the issue where LoRA files created with the **PEFT (Parameter-Efficient Fine-Tuning)** library were incorrectly skipped with a "Skipping unsupported LoRA" warning.

### 1.2 Target LoRA

Example LoRA that becomes compatible after this fix:
- [qwen-image-edit-lowres-fix-input-image-repair](https://civitai.com/models/1889350/qwen-image-edit-lowres-fix-input-image-repair)

---

## 2. Previous Issues

### 2.1 Key Format Differences

LoRA files have multiple key naming conventions:

| Format | Example | Generation Tool |
|--------|---------|----------------|
| **Standard Format** | `.lora_A.weight` | Kohya-ss, Diffusers |
| **PEFT Format** | `.lora_A.default.weight` | Hugging Face PEFT |

### 2.2 Previous Code (Problem Location)

```python
# main branch (before fix)
standard_patterns = (
    ".lora_up.weight",
    ".lora_down.weight",
    ".lora_A.weight",      # ← Exact match: requires ending with .weight
    ".lora_B.weight",
    ".lora.up.weight",
    ".lora.down.weight",
    ".lora.A.weight",
    ".lora.B.weight",
)
```

### 2.3 Problem Mechanism

PEFT format LoRA key:
```
transformer_blocks.0.attn.add_k_proj.lora_A.default.weight
                                          ^^^^^^^^
                                          Additional tag (.default)
```

The previous pattern `.lora_A.weight` expected **suffix exact match**, so it does not match `.lora_A.default.weight`.

### 2.4 Verification Test

```python
>>> import re
>>> pattern = r"\.lora_A\.weight"
>>> key = "transformer_blocks.0.attn.add_k_proj.lora_A.default.weight"
>>> re.search(pattern, key)
None  # No match
```

→ **Result**: "Skipping unsupported LoRA" warning is output, and the LoRA is ignored

---

## 3. Proposal Verification

### 3.1 LoRA File Analysis

Target file: `Qwen-Image-Edit-Lowres-Fix.safetensors`

```
Total keys: 1440
Key format: transformer_blocks.N.xxx.lora_A.default.weight
          transformer_blocks.N.xxx.lora_B.default.weight
```

**Sample keys (partial)**:
```
transformer_blocks.0.attn.add_k_proj.lora_A.default.weight
transformer_blocks.0.attn.add_k_proj.lora_B.default.weight
transformer_blocks.0.attn.add_q_proj.lora_A.default.weight
transformer_blocks.0.img_mod.1.lora_A.default.weight
transformer_blocks.0.txt_mod.1.lora_B.default.weight
...
```

### 3.2 Match Test After Fix

```python
>>> new_patterns = (".lora_A.", ".lora_B.")
>>> key = "transformer_blocks.0.attn.add_k_proj.lora_A.default.weight"
>>> any(p in key for p in new_patterns)
True  # Matches
```

### 3.3 Backward Compatibility Verification

| Key Format | Old Pattern | New Pattern |
|------------|-------------|-------------|
| `.lora_A.weight` | ✅ Match | ✅ Match |
| `.lora_B.weight` | ✅ Match | ✅ Match |
| `.lora_A.default.weight` | ❌ No match | ✅ Match |
| `.lora_B.default.weight` | ❌ No match | ✅ Match |

**Conclusion**: The new pattern **fully includes** the old pattern, so backward compatibility is maintained.

### 3.4 Real-World Test Results

ComfyUI test results (log excerpt):

```
✅ Standard LoRA (Rank-Decomposed)
[APPLY] LoRA applied to: transformer_blocks.0.attn.add_qkv_proj
[APPLY] LoRA applied to: transformer_blocks.0.attn.to_qkv
[AWQ_MOD] transformer_blocks.0.img_mod.1: Storing LoRA weights for manual Planar injection
...
Sampled LoRA composition complete. Applied: 360, Mod-patched: 120, Skipped: 0
```

---

## 4. Modified Files

### 4.1 Changed Files List

| File | Change Type |
|------|-------------|
| `nunchaku_code/lora_qwen.py` | Modified |
| `README.md` | Updated (minor) |

---

## 5. Code Changes

### 5.1 `_detect_lora_format` Function Fix

#### Change Location (diff format)

```diff
@@ -42,14 +42,14 @@ def _detect_lora_format(lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, A
     keys = list(lora_state_dict.keys())
 
     standard_patterns = (
-        ".lora_up.weight",
-        ".lora_down.weight",
-        ".lora_A.weight",
-        ".lora_B.weight",
-        ".lora.up.weight",
-        ".lora.down.weight",
-        ".lora.A.weight",
-        ".lora.B.weight",
+        ".lora_up.",
+        ".lora_down.",
+        ".lora_A.",
+        ".lora_B.",
+        ".lora.up.",
+        ".lora.down.",
+        ".lora.A.",
+        ".lora.B.",
     )
```

#### Code After Fix

```python
standard_patterns = (
    ".lora_up.",      # Changed to partial match
    ".lora_down.",
    ".lora_A.",       # Matches both .lora_A.weight and .lora_A.default.weight
    ".lora_B.",
    ".lora.up.",
    ".lora.down.",
    ".lora.A.",
    ".lora.B.",
)
```

### 5.2 Skip Log Addition

#### Change Location (diff format)

```diff
@@ -1274,6 +1274,8 @@ def compose_loras_v2(
         _log_lora_format_detection(str(first_lora_path_or_dict)[:50], _first_detection)
 
     aggregated_weights: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
+    # Track names of LoRA files that contain skipped weights
+    skipped_lora_names = set()
 
     # --- 4. Main Loading Loop ---
     for idx, (lora_path_or_dict, strength) in enumerate(lora_configs):
@@ -1402,6 +1404,7 @@ def compose_loras_v2(
                      # mod_layer_applied_count += 1  <-- Removed to avoid double counting per LoRA
                 else:
                     skipped_modules_count += 1
+                    skipped_lora_names.update(w["source"] for w in weight_list)
             elif is_awq_w4a16:
```

#### Log Output Code After Fix

```python
if skipped_lora_names:
    logger.warning("""
⚠️  Some weights from the following LoRAs were automatically skipped 
    because modulation layers (img_mod/txt_mod) are extremely sensitive to Nunchaku quantization. 
    To override this and force loading, set environment variable QWENIMAGE_LORA_APPLY_AWQ_MOD=1:""")
    for source in sorted(list(skipped_lora_names)):
        short_name = os.path.basename(source) if os.path.sep in source else source
        logger.warning(f"   - {short_name}")
```

---

## 6. Meaning of the Fix

### 6.1 Pattern Change Intent

| Change | Old | New | Effect |
|--------|-----|-----|--------|
| Suffix match → Partial match | `.lora_A.weight` | `.lora_A.` | Allows additional tags in PEFT format |

### 6.2 Technical Details

#### Old Pattern Behavior

```python
".lora_A.weight" in "...lora_A.default.weight"  # False
```
→ Requires the string `.lora_A.weight` to be **continuously contained**

#### New Pattern Behavior

```python
".lora_A." in "...lora_A.default.weight"  # True
```
→ **OK** if the string `.lora_A.` is contained (anything can follow)

### 6.3 Skip Log Addition Intent

In Nunchaku quantized models, applying LoRA to AWQ modulation layers (`img_mod.1` / `txt_mod.1`) causes noise issues. By default, these layers are skipped, but explicitly showing users **which LoRA files were skipped** enables:

1. Easier debugging
2. Guidance on using the `QWENIMAGE_LORA_APPLY_AWQ_MOD=1` environment variable to force loading

---

## 7. Relationship to Existing Code

### 7.1 `_detect_lora_format` vs `_RE_LORA_SUFFIX`

| Function/Regex | Purpose | Scope |
|----------------|---------|-------|
| `_detect_lora_format` | **Format detection for log display** | UI/Logs only |
| `_RE_LORA_SUFFIX` | Used for **actual key mapping** | LoRA application logic |

`_RE_LORA_SUFFIX` definition:
```python
_RE_LORA_SUFFIX = re.compile(r"\.(?P<tag>lora(?:[._](?:A|B|down|up)))(?:\.[^.]+)*\.weight$")
```

This regex **already supports PEFT format**:
- `(?:\.[^.]+)*` → Allows additional tags like `.default`

Therefore, PR #48's change fixes **format detection for log display** and does not affect actual LoRA application logic (already supported).

---

## 8. Summary

| Item | Content |
|------|---------|
| **Problem** | PEFT format LoRA (`.lora_A.default.weight`) was skipped as "unsupported" |
| **Cause** | `_detect_lora_format` pattern required suffix exact match |
| **Fix** | Changed pattern to partial match (`.lora_A.weight` → `.lora_A.`) |
| **Additional Feature** | Display skipped LoRA file names in warning logs |
| **Backward Compatibility** | ✅ Maintained (new pattern includes old pattern) |
| **Test Results** | ✅ 360 layers applied successfully, 0 skipped |

---

## 9. Related Links

- **PR**: https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/pull/48
- **Target LoRA**: https://civitai.com/models/1889350/qwen-image-edit-lowres-fix-input-image-repair
- **PEFT Official**: https://github.com/huggingface/peft
