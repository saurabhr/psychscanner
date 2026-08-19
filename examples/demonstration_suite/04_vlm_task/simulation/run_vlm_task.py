"""VLM shape-naming task: Single-Turn vs Summary memory x Reward vs Accuracy feedback.

Each trial shows the model one PIL-drawn shape image and asks for its color and
shape ("color shape", e.g. "red circle"). The 6 images (stimuli.py) are shown
twice per participant (block 1, block 2, independently shuffled) so a repeated
image can show whether feedback from block 1 carries into block 2 -- and
whether that carry-over depends on memory condition.

Conditions (2x2, run separately, same vision model):
  memory:   SingleTurn (stateless call each trial) vs Summary (Convo memory
            with a rolling summary once history exceeds memory_k, i.e. the
            model can recall earlier images/feedback in the conversation)
  feedback: Reward (a +1/-1 score after each answer, no answer revealed) vs
            Accuracy (explicit correct/incorrect + the right answer)

Model: google/gemma-4-26b-a4b-it:free via OpenRouter (free, vision-capable).
nvidia/nemotron-nano-12b-v2-vl:free (used in examples/demonstration_suite/04_vlm_task/advanced/set3_multimodal)
was tried first but its free-tier route repeatedly 504'd ("upstream idle
timeout") after ~2 minutes; gemma answered correctly in ~3s in the same
smoke test, so this run uses gemma instead -- see reports/report.md.

Run with a real OpenRouter API key (OPENROUTER_API_KEY in .env at repo root):

    source .venv/bin/activate
    python examples/demonstration_suite/04_vlm_task/simulation/run_vlm_task.py
"""
from __future__ import annotations

import random
import re
from pathlib import Path

from dotenv import load_dotenv

import psychscanner as psy
from psychscanner import ExpCard, ExpCardInit, ScannerModel, to_csv
from psychscanner.datasets.prompts.multimodal import image_block
from psychscanner.feedback import FeedbackBase

from stimuli import generate

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

DEMO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DEMO_DIR / "data"
RAW_DIR = DATA_DIR / "_runs"

MODEL = "google/gemma-4-26b-a4b-it:free"
FAMILY = "openrouter"
PARAMS = {"temperature": 0, "timeout": 30}
NSIM = 2          # simulated participants per condition
N_BLOCKS = 2       # each of the 6 images shown once per block

MEMORY_CONDITIONS = ["SingleTurn", "Summary"]
FEEDBACK_CONDITIONS = ["reward", "accuracy"]

INSTRUCTIONS = (
    "You will see a series of simple images, one at a time. For each image, "
    "state the color and shape you see, in the exact format 'color shape' "
    "(e.g. 'red circle'). Some images repeat later in the session."
)


def _parse_answer(text: str, color: str, shape: str) -> bool:
    """Loose match: both the expected color word and shape word appear."""
    t = text.lower()
    return bool(re.search(rf"\b{color}\b", t)) and bool(re.search(rf"\b{shape}\b", t))


class RewardFeedback(FeedbackBase):
    """Numeric reward only (+1 correct / -1 incorrect); correct answer withheld."""

    def __init__(self):
        self.total = 0

    def on_response(self, trial: dict, response: dict) -> str | None:
        correct = _parse_answer(response.get("content", ""), trial["color"], trial["shape"])
        self.total += 1 if correct else -1
        return f"Score: {'+1' if correct else '-1'} (running total: {self.total})."


class AccuracyFeedback(FeedbackBase):
    """Explicit correct/incorrect feedback that also reveals the right answer."""

    def on_response(self, trial: dict, response: dict) -> str | None:
        correct = _parse_answer(response.get("content", ""), trial["color"], trial["shape"])
        answer = f"{trial['color']} {trial['shape']}"
        if correct:
            return f"Correct! It was indeed a {answer}."
        return f"Incorrect. It was actually a {answer}."


FEEDBACK_FNS = {"reward": RewardFeedback, "accuracy": AccuracyFeedback}


def build_task(taskname: str, seed: int) -> dict:
    items = generate()
    rng = random.Random(seed)
    trials = []
    for block in range(1, N_BLOCKS + 1):
        order = items[:]
        rng.shuffle(order)
        for pos, item in enumerate(order, start=1):
            trials.append({
                "trcode": f"shapes_{item['name']}_b{block}",
                "stimulus": [
                    image_block(item["path"]),
                    {"type": "text", "text": "What color and shape is this? Answer as 'color shape'."},
                ],
                "block": block,
                "position_in_block": pos,
                "image": item["name"],
                "color": item["color"],
                "shape": item["shape"],
                "parser": None,
                "fb": True,
            })
    return {
        "tasktype": "vqa", "taskname": taskname,
        "instructions": {"definition": [INSTRUCTIONS]},
        "contexts": ["shape naming"], "contexts_id": ["shapes"], "context_present": False,
        "chain_type": "task", "parser": "0",
        "items": {"shapes": trials},
    }


def run_condition(memory: str, feedback_kind: str, attempts: int = 3) -> Path:
    label = f"{memory.lower()}_{feedback_kind}"
    task = build_task(label, seed=hash(label) % 10_000)

    # ponytail: the free OpenRouter model occasionally 504s ("upstream idle
    # timeout") outside the library's built-in 429-only retry; just redo the
    # whole condition a couple of times rather than patching library retry logic.
    last_exc = None
    for attempt in range(1, attempts + 1):
        card = ExpCardInit(
            model=MODEL, family=FAMILY, parameters=PARAMS,
            cogtype="no", nsim=NSIM,
            task_file=task,
            parser=None, chain_type="task",
            memory="SingleTurn" if memory == "SingleTurn" else "Convo",
            memory_k=-1 if memory == "SingleTurn" else 6,
            summary_k=0 if memory == "SingleTurn" else 3,
            feedback=True, feedback_fn=FEEDBACK_FNS[feedback_kind],
            projectname=label, proj_dir=RAW_DIR, tunnel_status="0",
        )
        try:
            scanner = ScannerModel(expcard=ExpCard(card))
            scanner.run()
            out_csv = DATA_DIR / f"{label}.csv"
            df = to_csv(scanner, path=out_csv)
            print(f"  {label}: saved {len(df)} rows -> {out_csv.name}")
            return out_csv
        except Exception as exc:  # noqa: BLE001 - transient API errors, just retry the condition
            last_exc = exc
            print(f"  {label}: attempt {attempt}/{attempts} failed ({exc}); retrying" if attempt < attempts else f"  {label}: giving up after {attempts} attempts")
    raise last_exc


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    generate()  # ensure stimuli exist
    for memory in MEMORY_CONDITIONS:
        for feedback_kind in FEEDBACK_CONDITIONS:
            print(f"\n==== running memory={memory} feedback={feedback_kind} ====")
            run_condition(memory, feedback_kind)
