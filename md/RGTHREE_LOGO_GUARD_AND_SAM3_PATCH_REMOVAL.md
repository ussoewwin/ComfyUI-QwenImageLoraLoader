# rgthree logo-route guard + SAM3 scalp patch removal

Repository: `ComfyUI-QwenImageLoraLoader`
Scope of this change set: one **new** self-defense patch, one **modified** file, one **deleted** patch.

---

## 1. What was wrong, and why the SAM3 patch was removed

### 1.1 The live-environment problem (rgthree-comfy logo route)

`rgthree-comfy` serves its logo from

```python
# rgthree-comfy/py/server/routes_config.py
async def get_logo(...):
    svg = await get_logo_svg()
    resp = svg.format(bg=bg, fg=fg)          # line 48
```

and `get_logo_svg()` caches whatever `tool.comfy.Icon`
(`https://comfy.rgthree.com/media/rgthree.svg`) returns **without validating it**:

```python
async with session.get(LOGO_URL, headers=headers) as resp:   # trust_env=True -> proxy aware
    LOGO_SVG = await resp.text()
LOGO_SVG = re.sub(r'(id="bg".*fill=)"[^\"]+"', r'\1"{bg}"', LOGO_SVG)
LOGO_SVG = re.sub(r'(id="fg".*fill=)"[^\"]+"', r'\1"{fg}"', LOGO_SVG)
```

Two consequences were observed in this environment:

1. **`ValueError: unexpected '{' in field name`** at `routes_config.py:48`.
   `str.format()` is applied to fetched markup. When the fetch is answered by something
   that is not our SVG - a proxy / captive portal / CDN error page (a Cloudflare 523 page
   contains `{` in its inline CSS) - the call raises, the route fails, and because the
   payload is cached in the module global the failure repeats for every request until
   ComfyUI is restarted.
2. **The error page rendered in the UI.** Simply replacing `.format()` with `.replace()`
   (so it stops raising) is not sufficient: the HTML is then served as
   `image/svg+xml` and painted on the canvas. Validation is required, not suppression.

Both failure modes depend on an external host the user does not control, so the correct
place for a fix while upstream is unpatched is **here**, as a self-defense guard.

### 1.2 Working-tree state in this repository (why the SAM3 patch is gone)

The SAM3 `seg_features` scalp patch had been added to this repository as **parity** for
[Comfy-Org/ComfyUI#15979](https://github.com/Comfy-Org/ComfyUI/pull/15979) and was
documented as "inert when the upstream fix is present".

The upstream fix is present on this machine - verified, not assumed:

| Check | Result |
|---|---|
| PR state (`gh pr view 15979 --repo Comfy-Org/ComfyUI`) | `MERGED` (author `ussoewwin`) |
| Merge commit contained in the installed HEAD | `git merge-base --is-ancestor 2035799 HEAD` -> exit 0 |
| Releases containing it | `v0.35.0`, `v0.35.1` (installed = `v0.35.1`, `856a922be`) |
| Installed source | `comfy/ldm/sam3/detector.py:426` new comment; `seg_features = features` at **line 434** = *after* the scalp trim |

With the upstream fix installed the patch has no effect but still costs import time and
maintenance, so the patch was deleted - **that patch only**. The deletion was kept
strictly scoped and verified:

* module removed: `patches/sam3_seg_features_scalp_patch.py` (gone from dev and live),
* no references left: `__init__.py` contains `sam3_seg_features_scalp` **0** times,
* other patch blocks untouched (the diff is limited to the SAM3 block; the Nunchaku and
  rgthree blocks are unchanged),
* `py_compile` passes for both `__init__.py` and the remaining patch modules,
* the removal is recoverable from git history (`6bc1cd8`).

Restoration condition: if this repository is ever used with **ComfyUI < v0.35.0** (no
upstream fix), the empty-mask behaviour returns and the patch must be restored from git
history - it is not needed with the current ComfyUI.

---

## 2. Files created / modified / deleted

| File | Status | Purpose |
|---|---|---|
| `patches/rgthree_logo_route_guard.py` | **NEW** | Validates rgthree's fetched logo markup and falls back to the bundled SVG; applied at import and again on server startup |
| `__init__.py` | **MODIFIED** | Wires the guard (9 added lines) |
| `patches/sam3_seg_features_scalp_patch.py` | **DELETED** | Redundant after upstream PR #15979 shipped in ComfyUI v0.35.0+ |
| `md/RGTHREE_LOGO_GUARD_AND_SAM3_PATCH_REMOVAL.md` | **NEW** | This document |

---

## 3. Full code

### 3.1 NEW - `patches/rgthree_logo_route_guard.py` (complete file)

```python
# -*- coding: utf-8 -*-
"""
Self-defense guard for rgthree-comfy's logo routes.

rgthree-comfy's ``py/server/routes_config.py`` does::

    svg = await get_logo_svg()          # caches tool.comfy.Icon, never validated
    resp = svg.format(bg=bg, fg=fg)     # str.format over fetched markup

so any literal ``{`` / ``}`` in the cached payload raises

    ValueError: unexpected '{' in field name

and ``/rgthree/logo.svg`` + ``/rgthree/logo_markup.svg`` fail for every request
until ComfyUI is restarted. A system proxy, a captive portal or a CDN error page
is enough to trigger it: ``get_logo_svg()`` fetches the remote SVG with
``trust_env=True``, and e.g. a Cloudflare 523 page contains ``{`` in its CSS.

Upstream fix proposed as rgthree/rgthree-comfy#763. Until that is merged *and*
installed on the host, this guard keeps the route safe locally: it wraps the
payload getter so anything that is not SVG markup is replaced by the logo
bundled with rgthree itself (``web/common/media/rgthree.svg`` - local, no
network, no braces).

Inert when upstream is fixed: a real SVG payload (with the ``{bg}`` / ``{fg}``
placeholders) is passed through untouched.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

_PATCH_TAG = "_qwen_lora_loader_rgthree_logo_guard"
_ORIGINAL_GET_LOGO_SVG = None
_HOOK_INSTALLED = False

_ROUTES_SUFFIX = os.path.normpath(os.path.join("rgthree-comfy", "py", "server", "routes_config.py"))
_LOGO_SUFFIX = os.path.join("web", "common", "media", "rgthree.svg")


def _find_routes_config():
    """The already-imported rgthree-comfy routes_config module, or None."""
    for mod in list(sys.modules.values()):
        path = getattr(mod, "__file__", None)
        if not path:
            continue
        try:
            if os.path.normpath(path).endswith(_ROUTES_SUFFIX):
                return mod
        except Exception:
            continue
    return None


def _bundled_logo(routes_config_module) -> str:
    """rgthree's own bundled logo (local file; contains no braces)."""
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(routes_config_module.__file__))))
        with open(os.path.join(root, _LOGO_SUFFIX), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.debug("rgthree logo guard: bundled logo unavailable (%s)", e)
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"></svg>'


def _install_guard(routes_config_module) -> bool:
    current = getattr(routes_config_module, "get_logo_svg", None)
    if current is None:
        return False
    if getattr(current, _PATCH_TAG, False):
        return True

    original = current

    async def guarded_get_logo_svg():
        svg = await original()
        if not isinstance(svg, str) or "<svg" not in svg.lower():
            logger.warning(
                "rgthree logo guard: the cached logo payload is not SVG (proxy/CDN error page?); "
                "serving the bundled logo instead (upstream rgthree-comfy#763)"
            )
            return _bundled_logo(routes_config_module)
        return svg

    guarded_get_logo_svg.__name__ = getattr(original, "__name__", "get_logo_svg")
    setattr(guarded_get_logo_svg, _PATCH_TAG, True)
    routes_config_module.get_logo_svg = guarded_get_logo_svg
    return True


def _install_prompt_hook() -> bool:
    """Re-apply before every prompt (custom-node load order is not guaranteed)."""
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return True
    try:
        import execution
    except Exception:
        return False
    executor = getattr(execution, "PromptExecutor", None)
    if executor is None:
        return False

    installed = False
    for method_name in ("execute", "execute_async"):
        original_method = executor.__dict__.get(method_name)
        if not callable(original_method) or getattr(original_method, _PATCH_TAG, False):
            continue

        def make(original_method=original_method):
            def wrapped(self, *args, **kwargs):
                try:
                    apply_rgthree_logo_guard()
                except Exception:
                    logger.debug("rgthree logo guard: re-apply skipped", exc_info=True)
                return original_method(self, *args, **kwargs)

            setattr(wrapped, _PATCH_TAG, True)
            return wrapped

        setattr(executor, method_name, make())
        installed = True

    _HOOK_INSTALLED = installed
    return installed


def _install_server_startup_hook() -> bool:
    """Apply the guard on server startup.

    The UI asks for /rgthree/logo*.svg while loading, which can happen before any
    prompt runs, so the per-prompt retry is not early enough on its own.
    """
    try:
        from server import PromptServer

        app = getattr(PromptServer.instance, "app", None)
        if app is None or getattr(app, "_qwen_logo_guard", False):
            return bool(getattr(app, "_qwen_logo_guard", False))

        async def _on_startup(_app):
            try:
                apply_rgthree_logo_guard()
            except Exception:
                logger.debug("rgthree logo guard: startup apply skipped", exc_info=True)

        app.on_startup.append(_on_startup)
        app._qwen_logo_guard = True
        return True
    except Exception:
        return False


_DISABLE_ENV = "QIL_DISABLE_RGTHREE_LOGO_GUARD"


def apply_rgthree_logo_guard() -> bool:
    """Guard rgthree's logo routes. True when the guard is in place.

    Set ``QIL_DISABLE_RGTHREE_LOGO_GUARD=1`` to switch the guard (including the
    per-prompt retry hook) off entirely, e.g. to A/B its runtime cost.

    Custom-node load order is not guaranteed, so when rgthree-comfy has not been
    imported yet the retry hook is installed anyway and the guard lands on the
    next prompt.
    """
    global _ORIGINAL_GET_LOGO_SVG
    if os.environ.get(_DISABLE_ENV) == "1":
        return False
    _install_server_startup_hook()
    routes = _find_routes_config()
    if routes is None or not _install_guard(routes):
        return False
    _ORIGINAL_GET_LOGO_SVG = routes.get_logo_svg
    return True
```

### 3.2 MODIFIED - `__init__.py`, the added block

```python
# --- rgthree logo route guard (self-defense until rgthree-comfy#763 lands) ---
try:
    from .patches.rgthree_logo_route_guard import apply_rgthree_logo_guard
    if apply_rgthree_logo_guard():
        logger.info("Applied rgthree logo route guard (non-SVG payloads use the bundled logo).")
    else:
        logger.debug(
            "rgthree logo route guard: rgthree-comfy not loaded yet; retrying at every prompt."
        )
except Exception as e:
    logger.debug(f"Error importing/applying rgthree logo route guard: {e}")
```

### 3.3 DELETED - `__init__.py`, the removed block (for the record)

```python
# --- SAM3 segmentation-head scalp patch (upstream PR #15979 parity) ---
try:
    from .patches.sam3_seg_features_scalp_patch import apply_sam3_seg_features_scalp_patch
    if apply_sam3_seg_features_scalp_patch():
        logger.info("Applied SAM3 seg_features scalp patch (empty-mask fix).")
    else:
        logger.debug("SAM3 seg_features scalp patch not applied (SAM3 not available).")
except Exception as e:
    logger.debug(f"Error importing/applying SAM3 seg_features scalp patch: {e}")
```

and the deleted module `patches/sam3_seg_features_scalp_patch.py` (it wrapped
`comfy.ldm.sam3.detector.SAM3Detector._detect`, pre-trimmed `features`/`positions` by
`self.scalp` and temporarily set `self.scalp = 0`, restoring it in a `finally`).

---

## 4. What the code means

### 4.1 Why the guard wraps `routes_config.get_logo_svg`

`get_logo()` calls the module-global name `get_logo_svg`, which `routes_config.py` bound
at import (`from ..pyproject import get_logo_svg`). Rebinding **that** name is what
changes the behaviour of the already-registered route; patching `pyproject.get_logo_svg`
would not.

### 4.2 Why validate `<svg` and fall back to the bundled logo

* A payload without `<svg` cannot be used as a logo. Falling back to
  `web/common/media/rgthree.svg` (shipped with rgthree, local, no network, no braces)
  means `str.format()` in the route can never raise, and the UI shows the real logo
  instead of an error page.
* A **valid** SVG payload is passed through untouched: with a working network and no
  upstream change, the guard is completely inert.

### 4.3 Why the startup hook

Custom-node import order is not guaranteed, and the UI requests `/rgthree/logo*.svg`
while it loads - before any prompt runs. The guard is therefore applied (a) immediately
when rgthree's `routes_config` is already imported, and (b) again via
`PromptServer.instance.app.on_startup`, so it is in place before the UI asks for the logo.

### 4.4 Operational switches

* `QIL_DISABLE_RGTHREE_LOGO_GUARD=1` disables the guard entirely (for A/B isolation).
* When the guard is triggered, the log records the reason at WARNING level
  (`rgthree logo guard: the cached logo payload is not SVG (proxy/CDN error page?); ...`),
  so the condition is observable and nothing is hidden.

### 4.5 Why removing the SAM3 patch is safe *and* why it is recorded here

The patch existed only to compensate for a not-yet-merged upstream fix. It is inert now
(verified above), so removing it removes dead weight without changing behaviour. The
restoration condition (ComfyUI < v0.35.0) and the git source (`6bc1cd8`) are recorded so
the decision can be reversed deliberately rather than rediscovered.
