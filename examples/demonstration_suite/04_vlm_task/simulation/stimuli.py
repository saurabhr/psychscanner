"""Generate the 6 shape/color stimulus images used by run_vlm_task.py.

No reusable image stimuli existed in the repo (examples/_tool_multimodal_tutorial_run/pixel.png
and examples/_supervisor_tutorial_run/pixel.png are 1x1 test pixels; the sibling
img_vivid_task/ directory has a text-only VVIQ task, no images) so these are drawn
with PIL: 6 distinct color+shape pairs, each with one unmistakable correct answer,
suitable for a "what color and shape is this?" VQA-style trial.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

STIMULI_DIR = Path(__file__).parent / "stimuli"

# (name, color, shape) -- color/shape are the ground-truth answer tokens.
SHAPES: list[tuple[str, tuple[int, int, int], str]] = [
    ("img1", (220, 30, 30), "circle"),
    ("img2", (30, 90, 220), "square"),
    ("img3", (30, 160, 60), "triangle"),
    ("img4", (230, 180, 20), "diamond"),
    ("img5", (150, 40, 200), "square"),
    ("img6", (240, 120, 20), "circle"),
]


def _draw(color: tuple[int, int, int], shape: str) -> Image.Image:
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    if shape == "circle":
        d.ellipse([40, 40, 160, 160], fill=color)
    elif shape == "square":
        d.rectangle([40, 40, 160, 160], fill=color)
    elif shape == "triangle":
        d.polygon([(100, 30), (170, 160), (30, 160)], fill=color)
    elif shape == "diamond":
        d.polygon([(100, 20), (180, 100), (100, 180), (20, 100)], fill=color)
    else:
        raise ValueError(f"unknown shape {shape}")
    return img


def generate() -> list[dict]:
    """Write the 6 PNGs (idempotent) and return their metadata."""
    STIMULI_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for name, color, shape in SHAPES:
        path = STIMULI_DIR / f"{name}.png"
        if not path.exists():
            _draw(color, shape).save(path)
        color_name = {
            (220, 30, 30): "red", (30, 90, 220): "blue", (30, 160, 60): "green",
            (230, 180, 20): "yellow", (150, 40, 200): "purple", (240, 120, 20): "orange",
        }[color]
        items.append({"name": name, "path": path, "color": color_name, "shape": shape})
    return items


if __name__ == "__main__":
    for item in generate():
        print(item["name"], "->", item["color"], item["shape"], item["path"])
