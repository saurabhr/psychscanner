"""Set 3 multimodal experiments — shared utilities.

Stimulus generation (image / text), word-number pairs, BPTFeedback handler,
and the task-dict builder used by all four Exp 3.x run.py files.

Design
------
10 sequences × 10 study items + 1 probe each (chain_type="task").
Probe targets serial positions 1-10 (one per sequence), covering the full
primacy-to-recency span.

Within each sequence, study items alternate between image-slots (positions
1,3,5,7,9) and text-slots (positions 2,4,6,8,10).

  OCR model  — image slots rendered as PIL PNG (multimodal HumanMessage)
  non-OCR    — image slots rendered as "[IMAGE TEXT] WORD → NUMBER" strings

This supports two behavioral measures:
  Lost-in-the-Middle   → accuracy by serial_position (1-10)
  Attentional Residue  → accuracy by probe_target_modality (image vs. text)
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False

from psychscanner.feedback import FeedbackBase


# ── 100 unique common 4-6 letter words ───────────────────────────────────────

WORDS_100: list[str] = [
    "APPLE", "BEACH", "CHAIR", "DRAMA", "EAGLE", "FENCE", "GRAPE", "HOTEL",
    "IMAGE", "JUDGE", "KNIFE", "LEMON", "MAGIC", "NIGHT", "OCEAN", "PIANO",
    "QUEEN", "RADAR", "STORM", "TIGER", "ULTRA", "VOICE", "WITCH", "XENON",
    "YACHT", "ZEBRA", "AMBER", "BONUS", "CEDAR", "DELTA", "EMBER", "FLAME",
    "GLOBE", "HAVEN", "IDEAL", "JOKER", "KARMA", "LASER", "MANOR", "NOBLE",
    "ORBIT", "PEARL", "QUEST", "RIVER", "SOLAR", "TOTEM", "URBAN", "VAPOR",
    "WHEEL", "XYLEM", "YIELD", "ZONAL", "ACORN", "BLOOM", "CORAL", "DIVER",
    "EPOCH", "FLORA", "GLOOM", "HELIX", "IVORY", "JEWEL", "KNEEL", "LUNAR",
    "MAPLE", "NAVAL", "OZONE", "PILOT", "QUILL", "ROCKY", "SIREN", "TORCH",
    "ULNAR", "VENOM", "WEAVE", "EXACT", "ASTER", "BLAZE", "CREST", "DRAKE",
    "EVENT", "FABLE", "GLARE", "HASTE", "INDEX", "JOUST", "KNACK", "LEVEL",
    "MARSH", "NERVE", "OPTIC", "PRISM", "QUOTA", "REIGN", "SCOUT", "TIDAL",
    "UMBRA", "VISOR", "WOVEN", "YOUNG",
]

_RNG_SEED = 3  # distinct from PAL seed (42) and zhang seed (7)


def _generate_sequences() -> list[list[tuple[str, int]]]:
    """10 sequences × 10 (word, number) pairs, fixed seed."""
    rng = np.random.default_rng(_RNG_SEED)
    words = list(WORDS_100)
    rng.shuffle(words)
    numbers = rng.integers(10, 100, size=100).tolist()
    return [
        [(words[i * 10 + j], int(numbers[i * 10 + j])) for j in range(10)]
        for i in range(10)
    ]


SEQUENCES: list[list[tuple[str, int]]] = _generate_sequences()


# ── Image stimulus ────────────────────────────────────────────────────────────

def _render_png_b64(word: str, number: int) -> str:
    """Render 'WORD → NUMBER' as a 400×100 white-background PNG → base64."""
    if not _PIL:
        raise RuntimeError("Pillow is required for OCR stimulus generation.")
    img = Image.new("RGB", (400, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = None
    for fp in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(fp, 36)
            break
        except (IOError, OSError):
            pass
    if font is None:
        font = ImageFont.load_default()
    draw.text((40, 30), f"{word}  →  {number}", fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_image_stimulus(word: str, number: int) -> list:
    """Multimodal content list: PNG image + memorisation instruction."""
    b64 = _render_png_b64(word, number)
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Memorize the word–number pair shown in this image."},
    ]


def make_text_direct(word: str, number: int) -> str:
    return f"Memorize: {word} → {number}"


def make_text_described(word: str, number: int) -> str:
    """Verbatim text description matching what an OCR pass would extract."""
    return f"[IMAGE TEXT] {word} → {number}  (memorize this pair)"


def make_probe(word: str) -> str:
    return (
        f"Recall test: What number was paired with the word {word}? "
        "Respond with the number you recall and your confidence (1–6)."
    )


# ── Feedback handler ──────────────────────────────────────────────────────────

class BPTFeedback(FeedbackBase):
    """Feedback for the behavioral profiling task — fires on probe trials only."""

    def on_response(self, trial: dict, response: dict) -> str | None:
        if trial.get("phase") != "probe":
            return None
        target = int(trial.get("number", -1))
        recalled_raw = response.get("recalled_number")
        if recalled_raw is None:
            return json.dumps({"result": "parse_error", "correct_answer": target})
        try:
            recalled = int(recalled_raw)
        except (ValueError, TypeError):
            return json.dumps({"result": "parse_error", "correct_answer": target})
        correct = recalled == target
        if correct:
            return json.dumps({
                "result": "Correct",
                "feedback": f"You correctly recalled {target}.",
            })
        return json.dumps({
            "result": "Incorrect",
            "your_recall": recalled,
            "correct_answer": target,
            "feedback": f"The correct number was {target}, not {recalled}.",
        })


# ── Task dict builder ─────────────────────────────────────────────────────────

_INSTRUCTIONS = (
    "You will study sequences of word–number pairs one at a time. "
    "Pay close attention to each pair. "
    "After studying all pairs in a sequence, you will be tested on one specific pair. "
    "Respond with the number you recall and your confidence "
    "(1 = not at all confident, 6 = completely confident)."
)


def build_task(ocr: bool, feedback: bool) -> dict:
    """Return the BPT-10 task dict for the given VLM type and feedback setting.

    Parameters
    ----------
    ocr : bool
        True → image-slot items are PIL-rendered PNGs (multimodal stimulus list).
        False → image-slot items are verbatim ``[IMAGE TEXT]`` text strings.
    feedback : bool
        True → probe items carry ``"fb": True`` so BPTFeedback fires.
    """
    items: dict[str, list] = {}

    for seq_i, seq in enumerate(SEQUENCES):
        trcode = f"seq_{seq_i + 1:02d}"
        probe_pos = seq_i  # seq i probes position i (0-indexed): covers 0..9

        stimuli: list[dict] = []
        for item_pos, (word, number) in enumerate(seq):
            is_image_slot = (item_pos % 2 == 0)  # positions 0,2,4,6,8

            if ocr and is_image_slot:
                stimulus = make_image_stimulus(word, number)
                modality = "image"
            elif ocr:
                stimulus = make_text_direct(word, number)
                modality = "text"
            elif is_image_slot:
                stimulus = make_text_described(word, number)
                modality = "described"
            else:
                stimulus = make_text_direct(word, number)
                modality = "direct"

            stimuli.append({
                "trcode": trcode,
                "stimulus": stimulus,
                "phase": "study",
                "seq_idx": seq_i,
                "serial_position": item_pos + 1,
                "word": word,
                "number": number,
                "modality": modality,
                "parser": None,
                "fb": False,
            })

        probe_word, probe_number = seq[probe_pos]
        probe_modality = stimuli[probe_pos]["modality"]
        prev_modality = stimuli[probe_pos - 1]["modality"] if probe_pos > 0 else None

        stimuli.append({
            "trcode": trcode,
            "stimulus": make_probe(probe_word),
            "phase": "probe",
            "seq_idx": seq_i,
            "serial_position": probe_pos + 1,
            "word": probe_word,
            "number": probe_number,
            "modality": "text",
            "probe_target_modality": probe_modality,
            "prev_modality": prev_modality,
            "parser": "SerialProbeResponse",
            "fb": feedback,
        })

        items[trcode] = stimuli

    return {
        "tasktype":       "episodic",
        "taskname":       "bpt10",
        "instructions":   _INSTRUCTIONS,
        "contexts":       ["Sequence recall"],
        "contexts_id":    ["seq"],
        "context_present": False,
        "items":          items,
        "parser":         None,
        "chain_type":     "task",
    }
