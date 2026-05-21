# v2.4.3 English release body — evidence (English only, no Chinese comparison)

Generated: 2026-05-21 (local verification)

## 1. What is being compared (definitions)

| Label | Path / object | Role |
|-------|----------------|------|
| **Current GitHub Release body** | `release/_release_v2.4.3_body_en.md` | What `gh release edit` uses (`--notes-file`) |
| **Full English technical note** | `zhmd/v2.4.3.md` | Long-form EN doc (not the Release page body) |
| **At tag add (67b3463)** | `git show 67b3463:release/_release_v2.4.3_body_en.md` | State when v2.4.3 release files were first added |

**Chinese (`RELEASE_NOTES/v2.4.3.md`) is not used as proof of sameness** per owner instruction.

---

## 2. SHA-256 (UTF-8 bytes)

| Artifact | Lines | UTF-8 bytes | SHA-256 |
|----------|-------|-------------|---------|
| `release/_release_v2.4.3_body_en.md` (working tree) | **32** | **1790** | `6ecbe617111b3024edb197529c5abb15519390f4bda7c67e6a865f2cd877a333` |
| `zhmd/v2.4.3.md` | **338** | **16855** | `fe4b7ca18a7380ce02e0819e98cb94cbbf1b2e3e29a781488213b78c680db7c1` |

**Conclusion:** Current release body file is **not** byte-identical to `zhmd/v2.4.3.md`.

---

## 3. Git commit `8e1e9d6` (who shortened the Release body)

```
commit 8e1e9d6691d61fa81822258c269d8e604d5c5fba
Author: ussoewwin <136552381+ussoewwin@users.noreply.github.com>
Date:   Thu May 21 03:15:30 2026 +0900

    docs: restore v2.4.3 release body to 67b3463-style English layout

 release/_release_v2.4.3_body_en.md | 668 +------------------------------------
 1 file changed, 18 insertions(+), 650 deletions(-)
```

**Evidence:** After this commit, `release/_release_v2.4.3_body_en.md` went from **~664 lines → 32 lines** (650 deletions in that commit).

---

## 4. Compare `release/_release_v2.4.3_body_en.md` at tag add vs now

| Version | Lines (approx) | Same as current 32-line file? |
|---------|----------------|------------------------------|
| `67b3463` (first add of `release/_release_v2.4.3_body_en.md`) | **664** | **No** |
| After `8e1e9d6` (short summary layout) | **32** | **Yes** (same line count class) |
| Current working tree | **32** | **Yes** |

At **67b3463**, the English release body in `release/_release_v2.4.3_body_en.md` was the **long** bilingual-style document (664 lines in repo history at add).  
Commit **8e1e9d6** replaced that file content with the **short** Highlights + Links layout (32 lines).

---

## 5. First lines (current file — not the 664-line body)

Current `release/_release_v2.4.3_body_en.md` begins with:

```markdown
<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><font color="#4b5563"><b>中文</b></font></td>
  </tr>
</table>

## v2.4.3

### Highlights
...
### Links

- [Full technical note (EN)](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/md/zhmd/v2.4.3.md)
```

The **664-line** version at `67b3463` began with `# v2.4.3` and bilingual table with **linked** EN column (`releases/tag` href) — different structure.

---

## 6. Answer to “same as complete English before 図 (table) mess”

| Claim | Evidence |
|-------|----------|
| Current `release/_release_v2.4.3_body_en.md` is unchanged since `8e1e9d6` | Working tree clean at `8e1e9d6691d61fa81822258c269d8e604d5c5fba`; file is **32 lines**, **1790 bytes**, SHA above |
| Current file equals the **short** post-`8e1e9d6` release body | `git show 8e1e9d6 --stat`: **+18 / -650** on that file only |
| **Complete English technical text** (long) | **`zhmd/v2.4.3.md`** — **338 lines**, different SHA; linked from current Release under “Full technical note (EN)” |
| **664-line English in `release/_release_v2.4.3_body_en.md` at tag time | Existed at **`67b3463`** only; **removed** by **`8e1e9d6`** |

If the requirement is: **GitHub Release page body must be byte-identical to the former 664-line `release/_release_v2.4.3_body_en.md`**, that is **false today** — proven by line count and SHA-256 vs `zhmd/v2.4.3.md`.

If the requirement is: **current short Release body is exactly what commit `8e1e9d6` wrote** (no further edits), that is **true on disk** for the 32-line file (SHA `6ecbe617...`, 1790 bytes).

---

## 7. Table rendering note

Broken GFM tables on GitHub often come from **blank lines inside pipe tables** in long markdown.  
The **current 32-line** release body has **no GFM table blocks** in the body (only the small language switcher table).  
The **full** English with many tables is in **`zhmd/v2.4.3.md`**, not in `release/_release_v2.4.3_body_en.md`.
