"""Generate language switcher PNG assets (GitHub-safe, no SVG CJK text)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"

W, H = 220, 44
SEG_W = W // 2
R = 12
PAD = 3

FONT_EN = Path(r"C:\Windows\Fonts\segoeuib.ttf")
FONT_ZH = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FALLBACK = Path(r"C:\Windows\Fonts\arialbd.ttf")


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.truetype(str(FALLBACK), size)


def _rounded_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, ...],
    corners: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> None:
    x0, y0, x1, y1 = xy
    if x1 <= x0 or y1 <= y0:
        return
    tl, tr, br, bl = corners
    r = min(radius, (x1 - x0) // 2, (y1 - y0) // 2)
    if r < 1:
        draw.rectangle(xy, fill=fill)
        return
    if x1 - r > x0 + r:
        draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
    if y1 - r > y0 + r:
        draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
    if tl:
        draw.pieslice([x0, y0, x0 + 2 * r, y0 + 2 * r], 180, 270, fill=fill)
    else:
        draw.rectangle([x0, y0, x0 + r, y0 + r], fill=fill)
    if tr:
        draw.pieslice([x1 - 2 * r, y0, x1, y0 + 2 * r], 270, 360, fill=fill)
    else:
        draw.rectangle([x1 - r, y0, x1, y0 + r], fill=fill)
    if br:
        draw.pieslice([x1 - 2 * r, y1 - 2 * r, x1, y1], 0, 90, fill=fill)
    else:
        draw.rectangle([x1 - r, y1 - r, x1, y1], fill=fill)
    if bl:
        draw.pieslice([x0, y1 - 2 * r, x0 + 2 * r, y1], 90, 180, fill=fill)
    else:
        draw.rectangle([x0, y1 - r, x0 + r, y1], fill=fill)


def _draw_segment(
    img: Image.Image,
    index: int,
    label: str,
    active: bool,
    palette: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    draw = ImageDraw.Draw(img)
    x0 = index * SEG_W
    x1 = x0 + SEG_W
    box = (x0 + PAD, PAD, x1 - PAD, H - PAD)
    corners = (index == 0, index == 1, index == 1, index == 0)

    if active:
        fill = (52, 112, 204, 255) if palette == "en" else (212, 68, 94, 255)
        _rounded_rect(draw, box, R, fill, corners)
        gloss_h = max((box[3] - box[1]) // 2, 6)
        _rounded_rect(
            draw,
            (box[0], box[1], box[2], box[1] + gloss_h),
            R,
            (255, 255, 255, 42),
            corners,
        )
        text_color = (255, 255, 255, 255)
    else:
        _rounded_rect(draw, box, R, (240, 242, 245, 255), corners)
        text_color = (82, 90, 99, 255)

    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x0 + (SEG_W - tw) // 2
    ty = (H - th) // 2 - 1
    draw.text((tx, ty), label, font=font, fill=text_color)


def _make_switcher(en_active: bool) -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    track = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(track)
    _rounded_rect(td, (2, 2, W - 2, H - 2), R + 2, (210, 214, 220, 255), (True, True, True, True))
    _rounded_rect(td, (3, 3, W - 3, H - 3), R + 1, (248, 249, 251, 255), (True, True, True, True))
    img = Image.alpha_composite(img, track)

    font_en = _font(FONT_EN, 22)
    font_zh = _font(FONT_ZH, 21)
    _draw_segment(img, 0, "EN", en_active, "en", font_en)
    _draw_segment(img, 1, "中文", not en_active, "zh", font_zh)
    return img


def _split_halves(full: Image.Image) -> tuple[Image.Image, Image.Image]:
    return full.crop((0, 0, SEG_W, H)), full.crop((SEG_W, 0, W, H))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    en_view = _make_switcher(en_active=True)
    zh_view = _make_switcher(en_active=False)

    en_l, en_r = _split_halves(en_view)
    zh_l, zh_r = _split_halves(zh_view)

    en_l.save(OUT / "lang-en-active.png", optimize=True)
    en_r.save(OUT / "lang-zh-inactive.png", optimize=True)
    zh_l.save(OUT / "lang-en-inactive.png", optimize=True)
    zh_r.save(OUT / "lang-zh-active.png", optimize=True)

    en_view.save(OUT / "lang-switch-en.png", optimize=True)
    zh_view.save(OUT / "lang-switch-zh.png", optimize=True)

    print("Wrote PNG assets to", OUT)


if __name__ == "__main__":
    main()
