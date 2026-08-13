"""Crop the two real logo marks (psychscanner, psychscanner-primal) out of
the original source artwork instead of recreating them from scratch.

Source: ps_ps-primal_logo.jpeg (1536x1024, top half = psychscanner,
bottom half = psychscanner-primal). Autocrops each half to its content
bbox and writes PNG + PDF next to the other generated logos, replacing
the matplotlib-recreated wordmarks branding_assets.py used to make for
these two specifically (the library/docker marks stay as-is -- there is
no "real" source art for those, they were designed fresh).
"""
import os
from PIL import Image

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ps_ps-primal_logo.jpeg")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "logos")


def autocrop(im, bg=(253, 253, 253), tol=12, pad=24):
    im = im.convert("RGB")
    px = im.load()
    w, h = im.size

    def is_bg(x, y):
        r, g, b = px[x, y]
        return abs(r - bg[0]) < tol and abs(g - bg[1]) < tol and abs(b - bg[2]) < tol

    left, right, top, bottom = w, 0, h, 0
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            if not is_bg(x, y):
                left, right = min(left, x), max(right, x)
                top, bottom = min(top, y), max(bottom, y)
    left, top = max(0, left - pad), max(0, top - pad)
    right, bottom = min(w, right + pad), min(h, bottom + pad)
    return im.crop((left, top, right, bottom))


def main():
    src = Image.open(SRC).convert("RGB")
    w, h = src.size
    top_half = src.crop((0, 0, w, h // 2))
    bottom_half = src.crop((0, h // 2, w, h))

    psychscanner = autocrop(top_half)
    primal = autocrop(bottom_half)

    for name, im in [("psychscanner", psychscanner), ("psychscanner-primal", primal)]:
        im.save(os.path.join(OUT, f"{name}.png"))
        im.save(os.path.join(OUT, f"{name}.pdf"), "PDF", resolution=300)
        print(name, im.size)


if __name__ == "__main__":
    main()
