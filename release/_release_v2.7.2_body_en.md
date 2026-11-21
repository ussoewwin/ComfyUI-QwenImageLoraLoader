<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.7.2.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

This document covers one change: every `.py` / `.js` file this package ships is ASCII only,
and `prestartup_script.py` installs a source-decoding fallback that rescues any source file in
the running ComfyUI process that was re-saved in a legacy code page.

Both halves exist because of one failure mode: a source file that is re-saved as ANSI / cp936 /
cp932 / cp1252 stops being valid UTF-8, and the pack that owns it fails to import. That failure
is not specific to this package; it can hit any pack in a ComfyUI installation, so the second
half takes it on for the whole process rather than for this package alone.

## 1. What the problem was

### 1.1 The report

Issue #35 describes the symptom without an environment dump:

> ComfyUI fails to load successfully because there are many Unicode characters in the files
> - after deleting them, it loads completely and runs perfectly. I have to delete these
> characters again every time after updating.

Two facts follow from the text alone: the workaround is to delete non-ASCII characters, and the
workaround has to be repeated after every update. Both are consistent with the hypothesis below.

### 1.2 What the shipped files contained

Measured over the tracked files at the parent of the change:

| Item | Value |
| --- | --- |
| Shipped `.py` / `.js` files | 31 |
| Files carrying non-ASCII | 20 |
| Non-ASCII characters | 239 |
| Affected lines | 151 |

The characters were emoji in log strings (`U+1F50D`, `U+2705`, `U+274C`, `U+26A0`, `U+1F4E6`,
`U+1F504`, `U+1F527`), box drawing and arrows in comments (`U+2500`, `U+2502`, `U+251C`,
`U+2192`, `U+2190`, `U+2014`), a Greek delta in a formula comment (`U+0394`), and emoji
prefixes on frontend widget names (`U+1F522`, `U+2795`, `U+26AB`, `U+1F7E2`, `U+2B06`, `U+2B07`,
`U+1F5D1`). No identifier was non-ASCII; every occurrence sat inside a string literal or a
comment.

### 1.3 What happens at load time

ComfyUI imports each custom node entry file and catches failures per pack:

```
nodes.py:2349  except Exception as e:
nodes.py:2350      logging.warning(traceback.format_exc())
nodes.py:2351      logging.warning(f"Cannot import {module_path} module for custom nodes: {e}")
nodes.py:2352      return False
```

A prestartup script is guarded the same way (`main.py:189-198`, `(PRESTARTUP FAILED)` at
`main.py:234`). So the failure mode is: the process keeps running, and the pack that owns the
mis-decoded file is missing from the node list. The user sees a pack that does not load, not a
dead server - which is why the report read as a pack-level problem.

### 1.4 Why it looked unrelated at first

The files in the repository are valid UTF-8; measured at the parent commit: 0 files with a
byte-order mark, 0 files that fail UTF-8 decoding, 0 syntax errors across 23 `.py` files. Nothing
in the shipped bytes is broken, so the report did not point at anything in this repository.
The bytes only become invalid after they leave the repository.

## 2. The essential cause

### 2.1 Python decodes source as UTF-8

PEP 3120 makes UTF-8 the default source encoding. `compile()` on a bytes object honours a BOM or
a coding cookie and otherwise decodes UTF-8; a legacy-code-page file raises
`SyntaxError: (unicode error) 'utf-8' codec can't decode byte 0xNN`. ComfyUI does not retry with
another encoding, and neither does `importlib`.

### 2.2 The conversion happens outside the repository

An editor, a sync tool or an installer that re-saves a `.py` file in the Windows ANSI code page
(cp936 on a Chinese system, cp932/Shift_JIS on a Japanese one, cp1252 on a Western one) replaces
every non-ASCII character with a byte sequence that is not valid UTF-8. The file on disk then
differs from the file in the repository, and the import dies at the first such byte.

### 2.3 Measured reproduction

The three states were produced from one real file,
`nodes/lora/qwenimage.py`, at the parent commit. All three measurements use the interpreter the
ComfyUI installation ships (`Python 3.13.17`).

| State | Result |
| --- | --- |
| As shipped (UTF-8) | `compile` OK |
| Re-saved as cp936 | `SyntaxError: (unicode error) 'utf-8' codec can't decode byte 0xa1 in position 165` at line 34 |
| Re-saved as cp936, non-ASCII removed | `compile` OK |

The third row is the reporter's workaround, reproduced exactly: deleting the non-ASCII
characters makes the file valid in any code page, so it loads again - and the next update
restores the UTF-8 file, the toolchain converts it again, and the symptom returns.

### 2.4 Why nothing rescued it before

ComfyUI loads a custom node entry file through
`app/governance.py::pack_module_spec()` (`nodes.py:2266`); when an organisation policy is not
active it falls back to `importlib.util.spec_from_file_location()` (`governance.py:125-126`).
Either way the source is decoded once, by the stock loader:

```
governance.py:115  class _SourceOnlyLoader(importlib.machinery.SourceFileLoader):
governance.py:118      def get_code(self, fullname):
governance.py:119          path = self.get_filename(fullname)
governance.py:120          return self.source_to_code(self.get_data(path), path)
```

`source_to_code` compiles the bytes; on a legacy code page that compile raises and the pack is
dropped. Submodules (`nodes/lora/*.py` and the rest) do not even pass through that loader - they
are ordinary imports, so a fix at the entry file alone would not cover them.

That is why the fix has two halves: make our own bytes code-page-proof, and put a fallback in
front of the decode for everything else in the process.

## 3. Files created and modified

### 3.1 Code

| File | Change | Lines |
| --- | --- | --- |
| `prestartup_script.py` | new: the source-decoding rescue, installed before the packs load | +131 |
| 20 shipped `.py` / `.js` files | non-ASCII replaced with ASCII (complete inventory in 4.2) | 151 |

The 20 files, with the number of non-ASCII characters each carried:

| File | Characters |
| --- | --- |
| `nunchaku_code/lora_qwen.py` | 32 |
| `nodes/lora/qwenimage.py` | 28 |
| `js/z_qwen_lora_dynamic_v1.js` | 20 |
| `js/zimageturbo_lora_dynamic_v1.js` | 20 |
| `nodes/lora/qwenimage_v3.py` | 19 |
| `nodes/lora/qwenimage_v1.py` | 16 |
| `nodes/lora/qwenimage_v2.py` | 15 |
| `nunchaku_code/lora_cache.py` | 14 |
| `nodes/lora/zimageturbo_v4.py` | 12 |
| `js/z_qwen_lora_dynamic.js` | 9 |
| `js/z_qwen_lora_dynamic_v3.js` | 9 |
| `js/zimageturbo_lora_dynamic_v3.js` | 9 |
| `js/zimageturbo_lora_dynamic_v4.js` | 9 |
| `nodes/lora/zimageturbo_v1.py` | 7 |
| `nodes/lora/zimageturbo_v3.py` | 6 |
| `patches/transformers_qwen_vl_docstring_patch.py` | 5 |
| `nodes/controlnet.py` | 3 |
| `wrappers/qwenimage_v2.py` | 3 |
| `wrappers/qwenimage.py` | 2 |
| `wrappers/zimageturbo.py` | 1 |

### 3.2 Documentation

| File | Change |
| --- | --- |
| `RELEASE_NOTES/changelog.md` | v2.7.2 entry |
| `zhmd/changelogzh.md` | v2.7.2 entry (Chinese) |
| `README.md` | the stale development-status banner removed |
| `zhmd/README.md` | the same banner removed (Chinese side) |
| `md/SOURCE_ENCODING_ASCII_RESCUE.md` | this document |

## 4. Full code

### 4.1 The rescue, as installed by `prestartup_script.py`

Imports added at the top of the module (the file stays torch-free; it runs before ComfyUI
initialises devices):

```python
import importlib
import importlib.machinery
import importlib.util
import locale
import logging
import os
import re
```

The block itself, verbatim, exactly as it appears in the file:

```python
# ---------------------------------------------------------------------------
# Source-encoding rescue (upstream support).
#
# Python decodes a .py file as UTF-8 unless it carries a coding cookie. When an
# editor or an installer pipeline re-saves a file in a legacy code page (ANSI,
# cp936/GBK, cp932, cp1252), the file is no longer valid UTF-8 and the import
# dies with "(unicode error) 'utf-8' codec can't decode byte 0xNN". The pack
# that owns the file then fails to load, whichever pack it is -- not only this
# one.
#
# The stock loader still runs first, byte for byte, .pyc caching included. This
# hook only steps in when that decode fails: it retries the same bytes with the
# local code page and the common legacy ones, and logs every rescued file with
# the code page it used. Nothing is silenced: the log names the file, the byte
# offset, and the fact that its text may differ from the author's original.
# ---------------------------------------------------------------------------
_RESCUED_SOURCES = []
_RESCUE_CANDIDATES = ("cp936", "cp932", "cp1252", "big5", "shift_jis", "latin-1")


def _decode_failure_offset(message):
    match = re.search(r"position (\d+)", message or "")
    return int(match.group(1)) if match else -1


def _is_decode_failure(exc):
    text = str(exc)
    return "codec can't decode" in text or "unicode error" in text


def _rescue_compile(data, path, exc):
    """Compile bytes the stock UTF-8 decode rejected, using a legacy code page."""
    raw = data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8", "surrogatepass")
    candidates = []
    preferred = locale.getpreferredencoding(False)
    if preferred:
        candidates.append(preferred)
    candidates.extend(_RESCUE_CANDIDATES)
    tried = set()
    for encoding in candidates:
        key = encoding.lower()
        if key in tried:
            continue
        tried.add(key)
        try:
            source = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        try:
            code = compile(source, path, "exec", dont_inherit=True)
        except SyntaxError:
            continue
        _RESCUED_SOURCES.append((path, encoding))
        logger.warning(
            "ComfyUI-QwenImageLoraLoader prestartup: source rescue: %s is not valid UTF-8 "
            "(first invalid byte at %d); compiled as %s. This file was saved outside UTF-8 "
            "(ANSI / cp936 / cp932 / cp1252). Re-download it or re-save it as UTF-8; text from "
            "this file may differ from the author's original.",
            path, _decode_failure_offset(str(exc)), encoding,
        )
        return code
    raise exc


def _install_source_encoding_rescue():
    """Cover every source import in this process, before ComfyUI loads the packs."""
    disabled = os.environ.get("HSWQ_NO_ENCODING_RESCUE", "").strip()
    if disabled not in ("", "0"):
        logger.warning(
            "ComfyUI-QwenImageLoraLoader prestartup: source rescue disabled by "
            "HSWQ_NO_ENCODING_RESCUE=%s",
            disabled,
        )
        return False

    loader_cls = importlib.machinery.SourceFileLoader
    if not getattr(loader_cls.get_code, "__hswq_encoding_rescue__", False):
        original_get_code = loader_cls.get_code

        def get_code(self, fullname):
            try:
                return original_get_code(self, fullname)
            except SyntaxError as exc:
                if not _is_decode_failure(exc):
                    raise
                path = self.get_filename(fullname)
                return _rescue_compile(self.get_data(path), path, exc)

        get_code.__hswq_encoding_rescue__ = True
        loader_cls.get_code = get_code

    # Subclasses that override get_code (ComfyUI's governance loader does) reach
    # the source through source_to_code instead; wrap the class that actually
    # defines it. importlib.abc.SourceLoader is a different class, so resolve it
    # from the real loader's MRO rather than assuming a module.
    source_loader = None
    for klass in loader_cls.__mro__:
        if "source_to_code" in klass.__dict__:
            source_loader = klass
            break
    if source_loader is None:
        return True
    if not getattr(source_loader.source_to_code, "__hswq_encoding_rescue__", False):
        original_source_to_code = source_loader.source_to_code

        def source_to_code(self, data, path, *, _optimize=-1):
            try:
                return original_source_to_code(self, data, path, _optimize=_optimize)
            except SyntaxError as exc:
                if not _is_decode_failure(exc):
                    raise
                return _rescue_compile(data, path, exc)

        source_to_code.__hswq_encoding_rescue__ = True
        source_loader.source_to_code = source_to_code
    return True


_ENCODING_RESCUE_INSTALLED = False
try:
    _ENCODING_RESCUE_INSTALLED = _install_source_encoding_rescue()
except Exception:
    logger.exception(
        "ComfyUI-QwenImageLoraLoader prestartup: source rescue could not be installed"
    )
```

### 4.2 Every replacement, in full

Each row is one contiguous run of changed characters: the file, the line in the new file, what
was removed (code points, so the table itself stays ASCII) and what replaced it. Rows with an
empty replacement removed the character outright.

| File | Line | Removed | Added |
| --- | --- | --- | --- |
| `js/z_qwen_lora_dynamic.js` | 3 | `U+2605U+2605U+2605U+2605U+2605U+2605` | `******` |
| `js/z_qwen_lora_dynamic.js` | 87 | `U+1F522` |  |
| `js/z_qwen_lora_dynamic.js` | 92 | `U+1F522` |  |
| `js/z_qwen_lora_dynamic.js` | 180 | `U+1F522` |  |
| `js/z_qwen_lora_dynamic_v1.js` | 4 | `U+2605U+2605U+2605U+2605U+2605U+2605` | `******` |
| `js/z_qwen_lora_dynamic_v1.js` | 107 | `U+25B6ymbo` | `rightarrow (drawn a a triange)` |
| `js/z_qwen_lora_dynamic_v1.js` | 108 | `fll` | `begnPath();U+000A    ctx.move` |
| `js/z_qwen_lora_dynamic_v1.js` | 109 | `ext"U+25B6", ` | `o - 4, midY - 5);U+000A    ctx.lineTo(valueCenter + arrowGap + 4, midY);U+000A    ctx.lineTo(valueCenter + arrowGap - 4` |
| `js/z_qwen_lora_dynamic_v1.js` | 111 | `4` | `5);U+000A    ctx.closePath();U+000A    ctx.fill(` |
| `js/z_qwen_lora_dynamic_v1.js` | 122 | `U+25C0ymbo` | `leftarrow (drawn a a triange)` |
| `js/z_qwen_lora_dynamic_v1.js` | 123 | `fll` | `begnPath();U+000A    ctx.move` |
| `js/z_qwen_lora_dynamic_v1.js` | 124 | `ext"U+25C0", ` | `o + 4, midY - 5);U+000A    ctx.lineTo(valueCenter - arrowGap - 4, midY);U+000A    ctx.lineTo(valueCenter - arrowGap + 4` |
| `js/z_qwen_lora_dynamic_v1.js` | 126 | `4` | `5);U+000A    ctx.closePath();U+000A    ctx.fill(` |
| `js/z_qwen_lora_dynamic_v1.js` | 211 | `U+2795` |  |
| `js/z_qwen_lora_dynamic_v1.js` | 251 | `U+2795` |  |
| `js/z_qwen_lora_dynamic_v1.js` | 286 | `U+26ABU+1F7E2` |  |
| `js/z_qwen_lora_dynamic_v1.js` | 293 | `U+2B06U+FE0F` |  |
| `js/z_qwen_lora_dynamic_v1.js` | 297 | `U+2B07U+FE0F` |  |
| `js/z_qwen_lora_dynamic_v1.js` | 301 | `U+1F5D1U+FE0F` |  |
| `js/z_qwen_lora_dynamic_v3.js` | 3 | `U+2605U+2605U+2605U+2605U+2605U+2605` | `******` |
| `js/z_qwen_lora_dynamic_v3.js` | 79 | `U+1F522` |  |
| `js/z_qwen_lora_dynamic_v3.js` | 83 | `U+1F522` |  |
| `js/z_qwen_lora_dynamic_v3.js` | 169 | `U+1F522` |  |
| `js/zimageturbo_lora_dynamic_v1.js` | 4 | `U+2605U+2605U+2605U+2605U+2605U+2605` | `******` |
| `js/zimageturbo_lora_dynamic_v1.js` | 107 | `U+25B6ymbo` | `rightarrow (drawn a a triange)` |
| `js/zimageturbo_lora_dynamic_v1.js` | 108 | `fll` | `begnPath();U+000A    ctx.move` |
| `js/zimageturbo_lora_dynamic_v1.js` | 109 | `ext"U+25B6", ` | `o - 4, midY - 5);U+000A    ctx.lineTo(valueCenter + arrowGap + 4, midY);U+000A    ctx.lineTo(valueCenter + arrowGap - 4` |
| `js/zimageturbo_lora_dynamic_v1.js` | 111 | `4` | `5);U+000A    ctx.closePath();U+000A    ctx.fill(` |
| `js/zimageturbo_lora_dynamic_v1.js` | 122 | `U+25C0ymbo` | `leftarrow (drawn a a triange)` |
| `js/zimageturbo_lora_dynamic_v1.js` | 123 | `fll` | `begnPath();U+000A    ctx.move` |
| `js/zimageturbo_lora_dynamic_v1.js` | 124 | `ext"U+25C0", ` | `o + 4, midY - 5);U+000A    ctx.lineTo(valueCenter - arrowGap - 4, midY);U+000A    ctx.lineTo(valueCenter - arrowGap + 4` |
| `js/zimageturbo_lora_dynamic_v1.js` | 126 | `4` | `5);U+000A    ctx.closePath();U+000A    ctx.fill(` |
| `js/zimageturbo_lora_dynamic_v1.js` | 211 | `U+2795` |  |
| `js/zimageturbo_lora_dynamic_v1.js` | 251 | `U+2795` |  |
| `js/zimageturbo_lora_dynamic_v1.js` | 286 | `U+26ABU+1F7E2` |  |
| `js/zimageturbo_lora_dynamic_v1.js` | 293 | `U+2B06U+FE0F` |  |
| `js/zimageturbo_lora_dynamic_v1.js` | 297 | `U+2B07U+FE0F` |  |
| `js/zimageturbo_lora_dynamic_v1.js` | 301 | `U+1F5D1U+FE0F` |  |
| `js/zimageturbo_lora_dynamic_v3.js` | 3 | `U+2605U+2605U+2605U+2605U+2605U+2605` | `******` |
| `js/zimageturbo_lora_dynamic_v3.js` | 86 | `U+1F522` |  |
| `js/zimageturbo_lora_dynamic_v3.js` | 91 | `U+1F522` |  |
| `js/zimageturbo_lora_dynamic_v3.js` | 200 | `U+1F522` |  |
| `js/zimageturbo_lora_dynamic_v4.js` | 3 | `U+2605U+2605U+2605U+2605U+2605U+2605` | `******` |
| `js/zimageturbo_lora_dynamic_v4.js` | 78 | `U+1F522` |  |
| `js/zimageturbo_lora_dynamic_v4.js` | 83 | `U+1F522` |  |
| `js/zimageturbo_lora_dynamic_v4.js` | 187 | `U+1F522` |  |
| `nodes/controlnet.py` | 446 | `U+2014` | `-` |
| `nodes/controlnet.py` | 572 | `U+2014` | `-` |
| `nodes/controlnet.py` | 775 | `U+2014` | `-` |
| `nodes/lora/qwenimage.py` | 37 | `U+2014` | `-` |
| `nodes/lora/qwenimage.py` | 143 | `U+26A0U+FE0F` | `[warn]` |
| `nodes/lora/qwenimage.py` | 169 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage.py` | 170 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage.py` | 171 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage.py` | 172 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage.py` | 173 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage.py` | 180 | `U+2705` | `[OK]` |
| `nodes/lora/qwenimage.py` | 181 | `U+1F4E6` | `[box]` |
| `nodes/lora/qwenimage.py` | 184 | `U+1F504` | `[reload]` |
| `nodes/lora/qwenimage.py` | 189 | `U+1F527` | `[tool]` |
| `nodes/lora/qwenimage.py` | 192 | `U+1F4E6` | `[box]` |
| `nodes/lora/qwenimage.py` | 207 | `U+274C` | `[X]` |
| `nodes/lora/qwenimage.py` | 254 | `U+2014` | `-` |
| `nodes/lora/qwenimage.py` | 392 | `U+26A0U+FE0F` | `[warn]` |
| `nodes/lora/qwenimage.py` | 418 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage.py` | 419 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage.py` | 420 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage.py` | 421 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage.py` | 422 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage.py` | 429 | `U+2705` | `[OK]` |
| `nodes/lora/qwenimage.py` | 430 | `U+1F4E6` | `[box]` |
| `nodes/lora/qwenimage.py` | 433 | `U+1F504` | `[reload]` |
| `nodes/lora/qwenimage.py` | 438 | `U+1F527` | `[tool]` |
| `nodes/lora/qwenimage.py` | 441 | `U+1F4E6` | `[box]` |
| `nodes/lora/qwenimage.py` | 456 | `U+274C` | `[X]` |
| `nodes/lora/qwenimage_v1.py` | 145 | `U+2192U+2713` | `->OK` |
| `nodes/lora/qwenimage_v1.py` | 148 | `U+2192U+2717` | `->X` |
| `nodes/lora/qwenimage_v1.py` | 185 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v1.py` | 186 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v1.py` | 187 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v1.py` | 188 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v1.py` | 189 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v1.py` | 193 | `U+2705` | `[OK]` |
| `nodes/lora/qwenimage_v1.py` | 194 | `U+1F4E6` | `[box]` |
| `nodes/lora/qwenimage_v1.py` | 196 | `U+1F504` | `[reload]` |
| `nodes/lora/qwenimage_v1.py` | 201 | `U+1F504` | `[reload]` |
| `nodes/lora/qwenimage_v1.py` | 208 | `U+1F527` | `[tool]` |
| `nodes/lora/qwenimage_v1.py` | 209 | `U+1F4E6` | `[box]` |
| `nodes/lora/qwenimage_v1.py` | 223 | `U+274C` | `[X]` |
| `nodes/lora/qwenimage_v2.py` | 179 | `U+26A0U+FE0F` | `[warn]` |
| `nodes/lora/qwenimage_v2.py` | 206 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v2.py` | 207 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v2.py` | 208 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v2.py` | 209 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v2.py` | 210 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v2.py` | 217 | `U+2705` | `[OK]` |
| `nodes/lora/qwenimage_v2.py` | 218 | `U+1F4E6` | `[box]` |
| `nodes/lora/qwenimage_v2.py` | 221 | `U+1F504` | `[reload]` |
| `nodes/lora/qwenimage_v2.py` | 226 | `U+1F504` | `[reload]` |
| `nodes/lora/qwenimage_v2.py` | 231 | `U+1F504` | `[reload]` |
| `nodes/lora/qwenimage_v2.py` | 253 | `U+1F527` | `[tool]` |
| `nodes/lora/qwenimage_v2.py` | 256 | `U+1F4E6` | `[box]` |
| `nodes/lora/qwenimage_v2.py` | 272 | `U+274C` | `[X]` |
| `nodes/lora/qwenimage_v3.py` | 39 | `U+2014` | `-` |
| `nodes/lora/qwenimage_v3.py` | 177 | `U+2192U+2713` | `->OK` |
| `nodes/lora/qwenimage_v3.py` | 180 | `U+2192U+2717` | `->X` |
| `nodes/lora/qwenimage_v3.py` | 221 | `U+26A0U+FE0F` | `[warn]` |
| `nodes/lora/qwenimage_v3.py` | 247 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v3.py` | 248 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v3.py` | 249 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v3.py` | 250 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v3.py` | 251 | `U+1F50D` | `[i]` |
| `nodes/lora/qwenimage_v3.py` | 255 | `U+2705` | `[OK]` |
| `nodes/lora/qwenimage_v3.py` | 256 | `U+1F4E6` | `[box]` |
| `nodes/lora/qwenimage_v3.py` | 258 | `U+1F504` | `[reload]` |
| `nodes/lora/qwenimage_v3.py` | 262 | `U+1F504` | `[reload]` |
| `nodes/lora/qwenimage_v3.py` | 268 | `U+1F527` | `[tool]` |
| `nodes/lora/qwenimage_v3.py` | 269 | `U+1F4E6` | `[box]` |
| `nodes/lora/qwenimage_v3.py` | 283 | `U+274C` | `[X]` |
| `nodes/lora/zimageturbo_v1.py` | 101 | `U+2192U+2713` | `->OK` |
| `nodes/lora/zimageturbo_v1.py` | 104 | `U+2192U+2717` | `->X` |
| `nodes/lora/zimageturbo_v1.py` | 122 | `U+274C` | `[X]` |
| `nodes/lora/zimageturbo_v1.py` | 154 | `U+2705` | `[OK]` |
| `nodes/lora/zimageturbo_v1.py` | 156 | `U+274C` | `[X]` |
| `nodes/lora/zimageturbo_v3.py` | 171 | `U+2192U+2713` | `->OK` |
| `nodes/lora/zimageturbo_v3.py` | 174 | `U+2192U+2717` | `->X` |
| `nodes/lora/zimageturbo_v3.py` | 220 | `U+26A0U+FE0F` | `[warn]` |
| `nodes/lora/zimageturbo_v4.py` | 161 | `U+2192U+2713` | `->OK` |
| `nodes/lora/zimageturbo_v4.py` | 164 | `U+2192U+2717` | `->X` |
| `nodes/lora/zimageturbo_v4.py` | 237 | `U+26A0U+FE0F` | `[warn]` |
| `nodes/lora/zimageturbo_v4.py` | 323 | `U+274C` | `[X]` |
| `nodes/lora/zimageturbo_v4.py` | 325 | `U+26A0U+FE0F` | `[warn]` |
| `nodes/lora/zimageturbo_v4.py` | 372 | `U+274C` | `[X]` |
| `nodes/lora/zimageturbo_v4.py` | 395 | `U+2705` | `[OK]` |
| `nodes/lora/zimageturbo_v4.py` | 397 | `U+274C` | `[X]` |
| `nunchaku_code/lora_cache.py` | 2 | `U+2014` | `-` |
| `nunchaku_code/lora_cache.py` | 5 | `U+2192` | `->` |
| `nunchaku_code/lora_cache.py` | 11 | `U+2190` | `<-` |
| `nunchaku_code/lora_cache.py` | 12 | `U+2190` | `<-` |
| `nunchaku_code/lora_cache.py` | 15 | `U+2192` | `->` |
| `nunchaku_code/lora_cache.py` | 16 | `U+2192` | `->` |
| `nunchaku_code/lora_cache.py` | 17 | `U+2192` | `->` |
| `nunchaku_code/lora_cache.py` | 180 | `U+2014` | `-` |
| `nunchaku_code/lora_cache.py` | 210 | `U+2014` | `-` |
| `nunchaku_code/lora_cache.py` | 220 | `U+2192` | `->` |
| `nunchaku_code/lora_cache.py` | 260 | `U+2014` | `-` |
| `nunchaku_code/lora_cache.py` | 292 | `U+2014` | `-` |
| `nunchaku_code/lora_cache.py` | 305 | `U+2014` | `-` |
| `nunchaku_code/lora_cache.py` | 316 | `U+2190` | `<-` |
| `nunchaku_code/lora_qwen.py` | 130 | `U+2705` | `[OK]` |
| `nunchaku_code/lora_qwen.py` | 132 | `U+274C` | `[X]` |
| `nunchaku_code/lora_qwen.py` | 134 | `U+274C` | `[X]` |
| `nunchaku_code/lora_qwen.py` | 136 | `U+274C` | `[X]` |
| `nunchaku_code/lora_qwen.py` | 139 | `U+274C` | `[X]` |
| `nunchaku_code/lora_qwen.py` | 143 | `U+2705` | `[OK]` |
| `nunchaku_code/lora_qwen.py` | 153 | `U+274C` | `[X]` |
| `nunchaku_code/lora_qwen.py` | 168 | `U+274C` | `[X]` |
| `nunchaku_code/lora_qwen.py` | 177 | `U+274C` | `[X]` |
| `nunchaku_code/lora_qwen.py` | 538 | `U+0394` | `delta` |
| `nunchaku_code/lora_qwen.py` | 539 | `U+0394` | `delta` |
| `nunchaku_code/lora_qwen.py` | 1329 | `U+26A0U+FE0F` | `[warn]` |
| `nunchaku_code/lora_qwen.py` | 1601 | `U+2192` | `->` |
| `nunchaku_code/lora_qwen.py` | 1674 | `U+26A0U+FE0F` | `[warn]` |
| `nunchaku_code/lora_qwen.py` | 1683 | `U+250CU+2500U+2510` | `+-+` |
| `nunchaku_code/lora_qwen.py` | 1684 | `U+251CU+2500U+2524` | `+-+` |
| `nunchaku_code/lora_qwen.py` | 1685 | `U+2514U+2500U+2518` | `+-+` |
| `nunchaku_code/lora_qwen.py` | 1692 | `U+2502U+2502` | `\|\|` |
| `nunchaku_code/lora_qwen.py` | 1697 | `U+2502U+2502` | `\|\|` |
| `nunchaku_code/lora_qwen.py` | 1707 | `U+2502U+2502` | `\|\|` |
| `nunchaku_code/lora_qwen.py` | 1788 | `U+2014` | `-` |
| `patches/transformers_qwen_vl_docstring_patch.py` | 5 | `U+2014` | `-` |
| `patches/transformers_qwen_vl_docstring_patch.py` | 6 | `U+2014` | `-` |
| `patches/transformers_qwen_vl_docstring_patch.py` | 191 | `U+2014` | `-` |
| `patches/transformers_qwen_vl_docstring_patch.py` | 212 | `U+2014` | `-` |
| `patches/transformers_qwen_vl_docstring_patch.py` | 220 | `U+2014` | `-` |
| `prestartup_script.py` | 7 |  | `U+000Aimport importlib.machineryU+000Aimport importlib` |
| `prestartup_script.py` | 9 |  | `U+000Aimport locale` |
| `prestartup_script.py` | 12 |  | `U+000Aimport re` |
| `prestartup_script.py` | 15 |  | `U+000AU+000A# ---------------------------------------------------------------------------U+000A# Source-encoding rescue (upstream support).U+000A#U+000A# Python decodes a .py file as UTF-8 unless it carries a coding cookie. When anU+000A# editor or an installer pipeline re-saves a file in a legacy code page (ANSI,U+000A# cp936/GBK, cp932, cp1252), the file is no longer valid UTF-8 and the importU+000A# dies with "(unicode error) 'utf-8' codec can't decode byte 0xNN". The packU+000A# that owns the file then fails to load, whichever pack it is -- not only thisU+000A# one.U+000A#U+000A# The stock loader still runs first, byte for byte, .pyc caching included. ThisU+000A# hook only steps in when that decode fails: it retries the same bytes with theU+000A# local code page and the common legacy ones, and logs every rescued file withU+000A# the code page it used. Nothing is silenced: the log names the file, the byteU+000A# offset, and the fact that its text may differ from the author's original.U+000A# ---------------------------------------------------------------------------U+000A_RESCUED_SOURCES = []U+000A_RESCUE_CANDIDATES = ("cp936", "cp932", "cp1252", "big5", "shift_jis", "latin-1")U+000AU+000AU+000Adef _decode_failure_offset(message):U+000A    match = re.search(r"position (\d+)", message or "")U+000A    return int(match.group(1)) if match else -1U+000AU+000AU+000Adef _is_decode_failure(exc):U+000A    text = str(exc)U+000A    return "codec can't decode" in text or "unicode error" in textU+000AU+000AU+000Adef _rescue_compile(data, path, exc):U+000A    """Compile bytes the stock UTF-8 decode rejected, using a legacy code page."""U+000A    raw = data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8", "surrogatepass")U+000A    candidates = []U+000A    preferred = locale.getpreferredencoding(False)U+000A    if preferred:U+000A        candidates.append(preferred)U+000A    candidates.extend(_RESCUE_CANDIDATES)U+000A    tried = set()U+000A    for encoding in candidates:U+000A        key = encoding.lower()U+000A        if key in tried:U+000A            continueU+000A        tried.add(key)U+000A        try:U+000A            source = raw.decode(encoding)U+000A        except (UnicodeDecodeError, LookupError):U+000A            continueU+000A        try:U+000A            code = compile(source, path, "exec", dont_inherit=True)U+000A        except SyntaxError:U+000A            continueU+000A        _RESCUED_SOURCES.append((path, encoding))U+000A        logger.warning(U+000A            "ComfyUI-QwenImageLoraLoader prestartup: source rescue: %s is not valid UTF-8 "U+000A            "(first invalid byte at %d); compiled as %s. This file was saved outside UTF-8 "U+000A            "(ANSI / cp936 / cp932 / cp1252). Re-download it or re-save it as UTF-8; text from "U+000A            "this file may differ from the author's original.",U+000A            path, _decode_failure_offset(str(exc)), encoding,U+000A        )U+000A        return codeU+000A    raise excU+000AU+000AU+000Adef _install_source_encoding_rescue():U+000A    """Cover every source import in this process, before ComfyUI loads the packs."""U+000A    disabled = os.environ.get("HSWQ_NO_ENCODING_RESCUE", "").strip()U+000A    if disabled not in ("", "0"):U+000A        logger.warning(U+000A            "ComfyUI-QwenImageLoraLoader prestartup: source rescue disabled by "U+000A            "HSWQ_NO_ENCODING_RESCUE=%s",U+000A            disabled,U+000A        )U+000A        return FalseU+000AU+000A    loader_cls = importlib.machinery.SourceFileLoaderU+000A    if not getattr(loader_cls.get_code, "__hswq_encoding_rescue__", False):U+000A        original_get_code = loader_cls.get_codeU+000AU+000A        def get_code(self, fullname):U+000A            try:U+000A                return original_get_code(self, fullname)U+000A            except SyntaxError as exc:U+000A                if not _is_decode_failure(exc):U+000A                    raiseU+000A                path = self.get_filename(fullname)U+000A                return _rescue_compile(self.get_data(path), path, exc)U+000AU+000A        get_code.__hswq_encoding_rescue__ = TrueU+000A        loader_cls.get_code = get_codeU+000AU+000A    # Subclasses that override get_code (ComfyUI's governance loader does) reachU+000A    # the source through source_to_code instead; wrap the class that actuallyU+000A    # defines it. importlib.abc.SourceLoader is a different class, so resolve itU+000A    # from the real loader's MRO rather than assuming a module.U+000A    source_loader = NoneU+000A    for klass in loader_cls.__mro__:U+000A        if "source_to_code" in klass.__dict__:U+000A            source_loader = klassU+000A            breakU+000A    if source_loader is None:U+000A        return TrueU+000A    if not getattr(source_loader.source_to_code, "__hswq_encoding_rescue__", False):U+000A        original_source_to_code = source_loader.source_to_codeU+000AU+000A        def source_to_code(self, data, path, *, _optimize=-1):U+000A            try:U+000A                return original_source_to_code(self, data, path, _optimize=_optimize)U+000A            except SyntaxError as exc:U+000A                if not _is_decode_failure(exc):U+000A                    raiseU+000A                return _rescue_compile(data, path, exc)U+000AU+000A        source_to_code.__hswq_encoding_rescue__ = TrueU+000A        source_loader.source_to_code = source_to_codeU+000A    return TrueU+000AU+000AU+000A_ENCODING_RESCUE_INSTALLED = FalseU+000Atry:U+000A    _ENCODING_RESCUE_INSTALLED = _install_source_encoding_rescue()U+000Aexcept Exception:U+000A    logger.exception(U+000A        "ComfyUI-QwenImageLoraLoader prestartup: source rescue could not be installed"U+000A    )U+000A` |
| `wrappers/qwenimage.py` | 87 | `U+1F527` | `[tool]` |
| `wrappers/qwenimage.py` | 235 | `U+2014` | `-` |
| `wrappers/qwenimage_v2.py` | 73 | `U+1F527` | `[tool]` |
| `wrappers/qwenimage_v2.py` | 230 | `U+2014` | `-` |
| `wrappers/qwenimage_v2.py` | 248 | `U+1F527` | `[tool]` |
| `wrappers/zimageturbo.py` | 69 | `U+1F527` | `[tool]` |

Per-character counts, over the whole change:

| Removed | Added | Times |
| --- | --- | --- |
| `U+1F50D` | `[i]` | 25 |
| `U+2014` | `-` | 20 |
| `U+274C` | `[X]` | 17 |
| `U+1F522` |  | 12 |
| `U+1F4E6` | `[box]` | 10 |
| `U+26A0U+FE0F` | `[warn]` | 9 |
| `U+2705` | `[OK]` | 9 |
| `U+1F504` | `[reload]` | 9 |
| `U+1F527` | `[tool]` | 9 |
| `U+2192` | `->` | 6 |
| `U+2605U+2605U+2605U+2605U+2605U+2605` | `******` | 6 |
| `U+2192U+2713` | `->OK` | 5 |
| `U+2192U+2717` | `->X` | 5 |
| `4` | `5);U+000A    ctx.closePath();U+000A    ctx.fill(` | 4 |
| `fll` | `begnPath();U+000A    ctx.move` | 4 |
| `U+2795` |  | 4 |
| `U+2190` | `<-` | 3 |
| `U+2502U+2502` | `\|\|` | 3 |
| `ext"U+25B6", ` | `o - 4, midY - 5);U+000A    ctx.lineTo(valueCenter + arrowGap + 4, midY);U+000A    ctx.lineTo(valueCenter + arrowGap - 4` | 2 |
| `ext"U+25C0", ` | `o + 4, midY - 5);U+000A    ctx.lineTo(valueCenter - arrowGap - 4, midY);U+000A    ctx.lineTo(valueCenter - arrowGap + 4` | 2 |
| `U+0394` | `delta` | 2 |
| `U+25B6ymbo` | `rightarrow (drawn a a triange)` | 2 |
| `U+25C0ymbo` | `leftarrow (drawn a a triange)` | 2 |
| `U+26ABU+1F7E2` |  | 2 |
| `U+2B06U+FE0F` |  | 2 |
| `U+2B07U+FE0F` |  | 2 |
| `U+1F5D1U+FE0F` |  | 2 |
| `U+250CU+2500U+2510` | `+-+` | 1 |
| `U+2514U+2500U+2518` | `+-+` | 1 |
| `U+251CU+2500U+2524` | `+-+` | 1 |

### 4.3 The two canvas arrows

The V1 stacks drew the strength arrows with the glyphs `U+25B6` and `U+25C0` through
`ctx.fillText`. They are now drawn as filled triangles, so the source stays ASCII and the
picture is unchanged. Before (one of the two, in `js/z_qwen_lora_dynamic_v1.js` and
`js/zimageturbo_lora_dynamic_v1.js`):

```javascript
ctx.fillText("U+25B6", valueCenter + arrowGap, midY + 4);
```

After:

```javascript
ctx.beginPath();
      ctx.moveTo(valueCenter + arrowGap - 4, midY - 5);
      ctx.lineTo(valueCenter + arrowGap + 4, midY);
      ctx.lineTo(valueCenter + arrowGap - 4, midY + 5);
      ctx.closePath();
      ctx.fill();
```

The mirrored left arrow is the same with `valueCenter - arrowGap` and the triangle pointing the
other way.

## 5. What it means

### 5.1 What is guaranteed now

| Guarantee | How it is enforced |
| --- | --- |
| This package cannot be broken by a code-page conversion | every shipped `.py` / `.js` is ASCII, and ASCII is identical in every code page |
| Another pack saved in a legacy code page still loads | the prestartup hook retries the decode with the local code page and five legacy ones |
| Submodules are covered, not just entry files | the hook wraps `SourceFileLoader.get_code`, which is what ordinary imports use |
| A loader that overrides `get_code` is covered | the hook also wraps `source_to_code` of the class that defines it (resolved from the MRO, not from `importlib.abc`, which is a different class) |
| A real syntax error is never hidden | only messages containing `codec can't decode` or `unicode error` are retried; anything else is re-raised unchanged |
| Nothing is silenced | each rescue logs the file, the offset of the first invalid byte and the code page used |
| The normal path is unchanged | the stock loader runs first; `.pyc` caching and byte-for-byte behaviour are untouched |
| The hook can be turned off | `HSWQ_NO_ENCODING_RESCUE=1`, and disabling it is logged |

### 5.2 Measured verification

Run with the installation's interpreter, loading `prestartup_script.py` as a module and then
importing fixtures written in temporary directories:

| Check | Result |
| --- | --- |
| Hook installed, `torch` still not imported | True / not in `sys.modules` |
| Plain UTF-8 module | loads unchanged |
| Module saved as cp936 | loads; text restored |
| Submodule of a package saved as cp936 | loads |
| Loader that overrides `get_code` (governance shape) | loads |
| Installing the hook twice | no second wrap |
| Genuine `SyntaxError` (`def f(:`) | still raised |
| Rescue registry after the three fixtures | three entries, all resolved as `cp936` |

Repository-wide checks after the change:

| Check | Result |
| --- | --- |
| Non-ASCII bytes in shipped `.py` / `.js` | 0 of 31 files |
| `compile` over tracked `.py` | 23 of 23 clean |
| `node --check` over tracked `.js` | 8 of 8 clean |

### 5.3 What it costs

Nothing on the normal path: one extra `try` around the stock loader call, and the wrapper only
does work when that call raises a decode error. On the rescue path the cost is one decode and one
compile per affected file, once per process.

### 5.4 Limits, stated plainly

- A rescued file keeps its bytes. Characters that the code page could not represent were already
  replaced when the file was saved, so rescued text can differ from the author's original. The
  log says this on every rescue.
- `latin-1` decodes any byte sequence, so it sits last: a file that is not in any of the earlier
  code pages still compiles, with mojibake in strings. This is deliberate - the alternative is a
  pack that does not load at all - but it is why the log line is mandatory.
- A prestartup script of another pack can run before this hook is installed, because ComfyUI
  iterates the custom-node directory in listing order. Entry files and everything imported after
  the hook is installed are covered; a prestartup script that ran earlier is not.
- Files that are valid UTF-8 but semantically wrong (for example cp1252 text that happens to
  decode) are not detectable here and are not the target of this change.

### 5.5 How to reproduce the verification

```python
# fixture: a module saved in a legacy code page
open(path, "wb").write('VALUE = "..."\n'.encode("cp936"))

# load the hook, then import the fixture
spec = importlib.util.spec_from_file_location("ps", "prestartup_script.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import fixture_module  # succeeds, and one warning line names the file and the code page
```

The warning line has this shape:

```
ComfyUI-QwenImageLoraLoader prestartup: source rescue: <path> is not valid UTF-8
(first invalid byte at <offset>); compiled as <code page>. This file was saved outside UTF-8
(ANSI / cp936 / cp932 / cp1252). Re-download it or re-save it as UTF-8; text from this file may
differ from the author's original.
```