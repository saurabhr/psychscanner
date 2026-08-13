"""Set 4 VISER experiments — shared utilities.

Stimulus generation for visual reasoning tasks: counting, visual search, scene description, spatial relationships.

Based on VISER method: augment visual inputs with low-level spatial structures and sequential prompts.

Design
------
Multiple tasks, each with sequences of trials.
For counting: scenes with 10-20 objects, count specific type.
VISER: overlay grid on image, prompt for sequential scanning.
Non-VISER: plain image or text description.

OCR model: receives PIL-rendered PNG images (multimodal)
Non-OCR: receives text descriptions of scenes
"""
from __future__ import annotations

import base64
import io
import json
import random
from typing import List, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False

from psychscanner.feedback import FeedbackBase


# ── Object types for scenes ───────────────────────────────────────────────────

OBJECT_TYPES = [
    ("red circle", (255, 0, 0), "circle"),
    ("blue square", (0, 0, 255), "square"),
    ("green triangle", (0, 255, 0), "triangle"),
    ("yellow star", (255, 255, 0), "star"),
    ("purple hexagon", (128, 0, 128), "hexagon"),
]

_RNG_SEED = 4  # distinct seed


def _generate_scenes() -> List[dict]:
    """Generate 10 scenes, each with 10-20 objects."""
    rng = np.random.default_rng(_RNG_SEED)
    scenes = []
    for scene_i in range(10):
        num_objects = rng.integers(10, 21)
        objects = []
        for _ in range(num_objects):
            obj_type = random.choice(OBJECT_TYPES)
            x = rng.integers(50, 350)
            y = rng.integers(50, 250)
            objects.append({
                "type": obj_type[0],
                "color": obj_type[1],
                "shape": obj_type[2],
                "x": x,
                "y": y,
            })
        scenes.append(objects)
    return scenes


SCENES: List[List[dict]] = _generate_scenes()


# ── Image generation ──────────────────────────────────────────────────────────

def _draw_object(draw: ImageDraw, obj: dict) -> None:
    """Draw an object on the image."""
    x, y = obj["x"], obj["y"]
    if obj["shape"] == "circle":
        draw.ellipse((x-20, y-20, x+20, y+20), fill=obj["color"])
    elif obj["shape"] == "square":
        draw.rectangle((x-20, y-20, x+20, y+20), fill=obj["color"])
    elif obj["shape"] == "triangle":
        draw.polygon([(x, y-20), (x-20, y+20), (x+20, y+20)], fill=obj["color"])
    elif obj["shape"] == "star":
        # Simple star
        points = []
        for i in range(10):
            angle = i * 36
            radius = 20 if i % 2 == 0 else 10
            points.append((x + radius * np.cos(np.radians(angle)), y + radius * np.sin(np.radians(angle))))
        draw.polygon(points, fill=obj["color"])
    elif obj["shape"] == "hexagon":
        points = []
        for i in range(6):
            angle = i * 60
            points.append((x + 20 * np.cos(np.radians(angle)), y + 20 * np.sin(np.radians(angle))))
        draw.polygon(points, fill=obj["color"])


def _render_scene_png(scene: List[dict], viser: bool = False) -> str:
    """Render scene as PNG, with VISER grid if enabled."""
    if not _PIL:
        raise RuntimeError("Pillow required for image generation.")
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for obj in scene:
        _draw_object(draw, obj)
    if viser:
        # Add grid lines for VISER
        for i in range(0, 400, 50):
            draw.line((i, 0, i, 300), fill=(0, 0, 0), width=1)
        for j in range(0, 300, 50):
            draw.line((0, j, 400, j), fill=(0, 0, 0), width=1)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_image_stimulus(scene: List[dict], viser: bool) -> List[dict]:
    """Multimodal content: PNG image + instruction."""
    b64 = _render_scene_png(scene, viser)
    instruction = "Count the objects by scanning the grid cells sequentially from top-left to bottom-right." if viser else "Count the objects in the image."
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": instruction},
    ]


def make_text_description(scene: List[dict], viser: bool) -> str:
    """Text description of the scene, with VISER grid description if enabled."""
    descriptions = []
    for obj in scene:
        descriptions.append(f"a {obj['type']} at position ({obj['x']}, {obj['y']})")
    scene_desc = "Scene: " + ", ".join(descriptions) + "."
    if viser:
        scene_desc += " The scene is overlaid with a grid of 50x50 cells. Count by scanning cells sequentially."
    return scene_desc


def make_probe(target_type: str) -> str:
    return f"How many {target_type} are there in the scene? Respond with the count."


# ── Feedback handler ──────────────────────────────────────────────────────────

class VISERFeedback(FeedbackBase):
    """Feedback for counting task."""

    def on_response(self, trial: dict, response: dict) -> str | None:
        if trial.get("phase") != "probe":
            return None
        target_count = trial.get("target_count", 0)
        resp_count = response.get("count")
        if resp_count is None:
            return json.dumps({"result": "parse_error", "correct": target_count})
        try:
            count = int(resp_count)
        except (ValueError, TypeError):
            return json.dumps({"result": "parse_error", "correct": target_count})
        correct = count == target_count
        if correct:
            return json.dumps({"result": "Correct"})
        return json.dumps({"result": "Incorrect", "your_count": count, "correct": target_count})


# ── Task builder ─────────────────────────────────────────────────────────────

_INSTRUCTIONS = (
    "You will be shown scenes with various objects. For each scene, count the number of a specific type of object. "
    "Respond with the exact count."
)


def build_task(ocr: bool, viser: bool) -> dict:
    """Build task dict for counting with OCR/non-OCR and VISER/no-VISER."""
    rng = np.random.default_rng(_RNG_SEED + (1 if ocr else 0) + (2 if viser else 0))  # different seed per condition
    items = {}
    for seq_i, scene in enumerate(SCENES):
        trcode = f"scene_{seq_i + 1:02d}"
        # Choose a target type present in the scene
        present_types = list(set(obj["type"] for obj in scene))
        target_type = random.choice(present_types)
        target_count = sum(1 for obj in scene if obj["type"] == target_type)

        stimuli = []
        # Study: show the scene
        if ocr:
            stimulus = make_image_stimulus(scene, viser)
        else:
            stimulus = make_text_description(scene, viser)
        stimuli.append({
            "trcode": trcode,
            "stimulus": stimulus,
            "phase": "study",
            "scene_idx": seq_i,
            "target_type": target_type,
            "target_count": target_count,
            "parser": None,
            "fb": False,
        })
        # Probe: ask for count
        stimuli.append({
            "trcode": trcode,
            "stimulus": make_probe(target_type),
            "phase": "probe",
            "scene_idx": seq_i,
            "target_type": target_type,
            "target_count": target_count,
            "parser": "SerialProbeResponse",  # Assuming parser for count
            "fb": True,  # Enable feedback
        })
        items[trcode] = stimuli

    return {
        "tasktype": "visual_reasoning",
        "taskname": "viser_counting",
        "instructions": _INSTRUCTIONS,
        "contexts": ["Scene counting"],
        "contexts_id": ["scene"],
        "context_present": False,
        "items": items,
        "parser": None,
        "chain_type": "task",
    }