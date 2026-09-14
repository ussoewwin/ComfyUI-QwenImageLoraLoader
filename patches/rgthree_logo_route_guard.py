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
