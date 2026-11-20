<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader/blob/main/zhmd/v2.7.0.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

This document records one change: v2.7.0 restores the preventive patch that silenced the
`[ERROR] ... loss/logits ... but not documented` lines printed by `transformers` when
Qwen3-VL / Qwen2.5-VL modelings are imported. The patch was removed in v2.6.9; it is back
because that removal only held for transformers 5.23.0 and later.

## 1. What this release changes

| item | value |
| --- | --- |
| restored file | `patches/transformers_qwen_vl_docstring_patch.py` (248 lines, blob `7751b492f53c`) |
| restored wiring | `prestartup_script.py` (+15 lines: 1 constant, 14-line block) |
| version | `2.6.9` to `2.7.0` in `pyproject.toml` and `__init__.py` |
| changelog | v2.7.0 entry in `RELEASE_NOTES/changelog.md` and `zhmd/changelogzh.md` |
| tag | `v2.7.0` (annotated), commit `c7804f3` |

v2.6.9 is not withdrawn. Its tag (`e4490e3`), its release page and its changelog entry stay
as they are. v2.6.7 and v2.6.8 are untouched. The two releases are the two sides of one
decision: v2.6.9 removed the patch on the evidence of transformers 5.23.0, and v2.7.0 puts
it back for the stacks that evidence did not cover.

## 2. Why the patch is back

### 2.1 The line still appears on older transformers

The removed patch recorded the mechanism: the Qwen VL modelings declared their own causal-LM
output class, inherited `CausalLMOutputWithPast`, and merged the parent documentation through
the shared `ModelOutputArgs` argument block. `ModelOutputArgs` does not carry `loss` and
`logits`, so the merge leaves two annotations undocumented and the validator prints two
`[ERROR]` lines at import of:

- `transformers.models.qwen3_vl.modeling_qwen3_vl`
- `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl`

Nothing fails; inference is unaffected. The damage is to the log, which then contains two
expected errors on every start that loads a Qwen VL text encoder.

### 2.2 What v2.6.9's evidence covered

The removal rested on a measurement of transformers 5.23.0: that version no longer uses
`ModelOutputArgs` in those modelings at all. `qwen3_vl` returns `CausalLMOutputWithPast`
directly (annotation L1298, construction L1363) and its only custom output class is
`BaseModelOutputWithDeepstackFeatures`, a plain dataclass. With the patch absent, importing
both modelings emits 0 `[ERROR]` lines on stdout and 0 on stderr.

That result is correct and it still holds. What it does not cover is every earlier release,
where the modelings still merge parent docs and the line still prints.

### 2.3 Who is affected

Users who run an older `transformers` - portable ComfyUI builds that ship the 5.15.x line,
or environments pinned by another custom node - still see the two lines. Those users are the
reason this release exists. On 5.23.0 and later the restored patch decides, by measurement,
to install nothing.

## 3. What the patch does

The patch wraps `transformers.utils.auto_docstring.get_args_doc_from_source` and, for
`ModelOutputArgs`, merges `loss` and `logits` entries parsed from
`CausalLMOutputWithPast.__doc__`. Two gates decide whether the wrapper is installed:

1. native gate - does `ModelOutputArgs` already document `loss` and `logits`;
2. import gate - a subprocess import of both modelings, with stdout captured, counting lines
   that contain `[ERROR]` and one of `loss` / `logits`.

| stack | gate 1 | gate 2 | result |
| --- | --- | --- | --- |
| transformers 5.23.0 and later | False (fields still absent) | True (import is clean) | wrapper **not** installed, no behaviour change |
| transformers before 5.23 | False | False (the lines appear) | wrapper installed before the modelings import, lines prevented |
| probe cannot run (timeout, non-zero exit) | False | None | wrapper installed (safe direction) |

The wrapper is tagged and idempotent, so a second call cannot wrap twice. It runs from
`prestartup_script.py`, before any custom node `__init__`, because ComfyUI-nunchaku imports its
`models/qwenimage.py` before this package's `__init__.py` runs.

## 4. Cost

On transformers 5.23.0 the patch takes the skip branch, but the probe has already run: one
subprocess import of both modelings per ComfyUI start. Measured on the development machine:
**29.82 s** for that call. That cost is accepted deliberately - it is what keeps a working
guard for older stacks without pinning `transformers` - and it is the trade v2.6.9 tried to
avoid and this release reverses.

## 5. Verification (measured)

| check | result |
| --- | --- |
| restored file identical to v2.6.8 | blob `7751b492f53c13998ffedceaa925681a7760635c`, `git diff v2.6.8` on both files = 0 lines |
| restored state identical to the removal's parent | `8ec3c00^` = `beeaeb7`, diff on both files = 0 lines |
| import probe with the patch absent (5.23.0) | stdout 0 `[ERROR]`, stderr 0 `[ERROR]` |
| probe cost, one call | 29.82 s |
| startup with the patch present (5.23.0) | log line `CausalLM ModelOutput docstring patch skipped: ... (upstream fixed - patch not installed)`; no wrapper installed |

## 6. Removing it again

The condition is version-based, not opinion-based: remove the patch once the supported minimum
`transformers` is 5.23 or later. At that point the same measurement that justified v2.6.9
applies to every supported stack, and `git revert 811b226` removes both the file and the
wiring. Until then, removing it costs users on older releases a log that carries two expected
errors - which is exactly what happened between v2.6.9 and this release.
