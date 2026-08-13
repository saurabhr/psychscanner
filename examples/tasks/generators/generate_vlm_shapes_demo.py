"""Build a standalone, task_library()-discoverable VLM task card from the
real stimulus images already used by
examples/demonstration_suite/04_vlm_task/simulation/run_vlm_task.py.

That demo's task_file is built inline in Python and never saved as a
reusable task card, so it isn't fetchable via task_library() -- this script
extracts the same 6 real images (color+shape ground truth from stimuli.py,
not fabricated) into one, following this repo's standard `items` schema
instead of the advanced factorial design's ad-hoc block/memory-condition
setup. For the full 2x2 memory x feedback factorial, use
run_vlm_task.py directly.

Images are embedded as base64 (via psychscanner.datasets.prompts.multimodal
.image_block), matching exactly how run_vlm_task.py embeds them at call
time -- not a new convention, the same one already verified working there.

Run from the psychscanner project root:
    python examples/tasks/generators/generate_vlm_shapes_demo.py
"""
from __future__ import annotations

from pathlib import Path

from psychscanner import save_task_card
from psychscanner.datasets.prompts.multimodal import image_block

STIMULI_DIR = Path(__file__).parents[1] / "stimuli"
OUT_DIR = Path(__file__).parents[1]

# Ground truth from examples/demonstration_suite/04_vlm_task/simulation/
# stimuli.py's SHAPES list -- same 6 images, copied into examples/tasks/stimuli/.
SHAPES: list[tuple[str, str, str]] = [
    ("img1", "red", "circle"),
    ("img2", "blue", "square"),
    ("img3", "green", "triangle"),
    ("img4", "yellow", "diamond"),
    ("img5", "purple", "square"),
    ("img6", "orange", "circle"),
]

_INSTRUCTIONS = (
    "You will see a simple image. State the color and shape you see, in "
    "the exact format 'color shape' (e.g. 'red circle')."
)


def _build_card() -> dict:
    # trcode must be "{contexts_id value}_<anything>" -- task_prompts.py's
    # gen_trial_promptdata does trcode.split("_")[0] to look up the trial's
    # context in contexts_id. Found by a real ValueError during the mock-llm
    # verification run below, not assumed from reading the schema docs.
    items = {}
    for name, color, shape in SHAPES:
        path = STIMULI_DIR / f"{name}.png"
        trcode = f"shapes_{name}"
        items[trcode] = [
            {
                "trcode": trcode,
                "stimulus": [image_block(path)],
                "corrAns": f"{color} {shape}",
            }
        ]
    return {
        "tasktype": "sc",
        "taskname": "vlm_shapes_demo",
        "instructions": {"definition": [_INSTRUCTIONS]},
        "contexts": ["shapes"],
        "contexts_id": ["shapes"],
        "context_present": False,
        "chain_type": "item",
        "parser": None,
        "items": items,
    }


def demo() -> None:
    """Self-check: every referenced stimulus image actually exists and the
    card has the expected number of items. No pytest needed."""
    for name, _, _ in SHAPES:
        assert (STIMULI_DIR / f"{name}.png").is_file(), f"missing stimulus: {name}.png"
    card = _build_card()
    assert len(card["items"]) == len(SHAPES)
    assert card["taskname"] == "vlm_shapes_demo"
    print("demo() OK —", len(card["items"]), "items, all stimulus files present")


if __name__ == "__main__":
    demo()
    card = _build_card()
    path = save_task_card(card, OUT_DIR / f"{card['taskname']}.tcard.psyscan")
    print(f"wrote {path} ({len(card['items'])} items)")
