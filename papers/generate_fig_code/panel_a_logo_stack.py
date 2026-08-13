"""Panel A of Figure 1: real logo images, cropped and stacked top-to-bottom.

Not TikZ -- this loads the actual logo files (the real cropped source art
for psychscanner/psychscanner-primal via crop_source_logos.py, and the
generated marks for the library/docker repos via branding_assets.py),
autocrops each tightly to its content bounding box, and stacks them as
plain boxes with no connecting lines -- relation is conveyed by grouping
and vertical order only. Run: `python3 panel_a_logo_stack.py`.
"""
import os
from PIL import Image, ImageDraw

LOGO_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "logos")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")

RULE = (210, 210, 212)


def autocrop(im, pad=6):
    im = im.convert("RGBA")
    bbox = im.getbbox()
    if bbox is None:
        return im
    l, t, r, b = bbox
    l, t = max(0, l - pad), max(0, t - pad)
    r, b = min(im.width, r + pad), min(im.height, b + pad)
    return im.crop((l, t, r, b))


def whiten_to_alpha(im, thresh=248):
    """The two real-photo crops (psychscanner/psychscanner-primal) come from
    a flat-white JPEG with no alpha channel; key the near-white background
    out so it blends into the panel canvas instead of showing as a box."""
    im = im.convert("RGBA")
    px = im.load()
    for y in range(im.height):
        for x in range(im.width):
            r, g, b, a = px[x, y]
            if r >= thresh and g >= thresh and b >= thresh:
                px[x, y] = (r, g, b, 0)
    return im


def load(name):
    """raw_*.png files come from `pdftoppm`, which always flattens PDF
    transparency onto opaque white -- key that white back out so every
    logo blends into the panel canvas with no visible box."""
    raw = os.path.join(LOGO_DIR, f"raw_{name}-1.png")
    if os.path.exists(raw):
        return whiten_to_alpha(autocrop(Image.open(raw)))
    im = autocrop(Image.open(os.path.join(LOGO_DIR, f"{name}.png")))
    if name in ("psychscanner", "psychscanner-primal"):
        im = whiten_to_alpha(im)
    return im


def scale_to_width(im, w):
    h = int(im.height * w / im.width)
    return im.resize((w, h), Image.LANCZOS)


def build():
    root = scale_to_width(load("psychscanner"), 1400)
    primal = scale_to_width(load("psychscanner-primal"), 1400)
    lib = scale_to_width(load("psyscan-library"), 1150)
    libp = scale_to_width(load("psyscan-library-primal"), 1150)
    dgen = scale_to_width(load("docker_ps-general"), 340)
    dneuro = scale_to_width(load("docker_ps-neuroscanner"), 340)
    dprim = scale_to_width(load("docker_ps-primal"), 340)

    gap = 34
    group_gap = 46
    docker_row_h = max(dgen.height, dneuro.height, dprim.height)
    total_w = 1400 + 200
    total_h = (root.height + gap + primal.height + group_gap + lib.height + gap + libp.height
               + group_gap + docker_row_h + 20)

    canvas = Image.new("RGBA", (total_w, total_h), (250, 250, 250, 255))
    draw = ImageDraw.Draw(canvas)

    def cx(im):
        return (total_w - im.width) // 2

    def rule(y):
        draw.line([(60, y), (total_w - 60, y)], fill=RULE, width=2)

    y = 10
    canvas.alpha_composite(root, (cx(root), y))
    y += root.height + gap
    canvas.alpha_composite(primal, (cx(primal), y))
    y += primal.height + group_gap // 2
    rule(y)
    y += group_gap // 2

    canvas.alpha_composite(lib, (cx(lib), y))
    y += lib.height + gap
    canvas.alpha_composite(libp, (cx(libp), y))
    y += libp.height + group_gap // 2
    rule(y)
    y += group_gap // 2

    row_w = dgen.width + 60 + dneuro.width + 60 + dprim.width
    x0 = (total_w - row_w) // 2
    canvas.alpha_composite(dgen, (x0, y + (docker_row_h - dgen.height) // 2))
    canvas.alpha_composite(dneuro, (x0 + dgen.width + 60, y + (docker_row_h - dneuro.height) // 2))
    canvas.alpha_composite(dprim, (x0 + dgen.width + 60 + dneuro.width + 60,
                                    y + (docker_row_h - dprim.height) // 2))
    y += docker_row_h

    out = canvas.crop((0, 0, total_w, y + 15))
    out.save(os.path.join(OUT, "logos", "panel_a_stack.png"))
    return out


if __name__ == "__main__":
    im = build()
    bg = Image.new("RGB", im.size, (250, 250, 250))
    bg.paste(im.convert("RGB"), (0, 0))
    bg.save(os.path.join(OUT, "logos", "panel_a_stack_rgb.pdf"), "PDF", resolution=600)
    print("wrote panel_a_stack.png /.pdf", im.size)
