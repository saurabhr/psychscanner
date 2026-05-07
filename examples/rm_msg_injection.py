"""Reality Monitoring feedback handler.

Subclasses FeedbackBase to provide trial-by-trial feedback for the RM
word-pair task.  Tracks words used across all trials within one participant
simulation via self.all_words_in_use — no module-level globals needed.

Usage::

    from rm_msg_injection import Stim_Trial_Injection

    card_in = ExpCard(
        ...
        feedback=True,
        feedback_fn=Stim_Trial_Injection,   # pass the class, not an instance
    )
"""
from __future__ import annotations

import json

from nltk.corpus import words as nltk_words

from psychscanner.feedback import FeedbackBase


# ---------------------------------------------------------------------------
# Helper: extract (response_str, rating_str) from a pre-parsed response dict.
# Uses the trial's 'parser' key for explicit field lookup so that:
#   Response_part_1_rm  →  Word_2 (str),            Rating (0–100)
#   Response_part_2_rm  →  Judgment (internal/ext),  Confidence (1–6)
# ---------------------------------------------------------------------------

def _extract_fields(response: dict, parser_name: str | None) -> tuple[str, str]:
    if parser_name == "Response_part_1_rm":
        return str(response.get("Word_2", "")), str(response.get("Rating", ""))
    if parser_name in ("Response_part_2_rm", "Response_part_2_rmrevo"):
        return str(response.get("Judgment", "")), str(response.get("Confidence", ""))
    # Unstructured / unknown parser — best-effort fallback
    values = list(response.values())
    response_str = str(response.get("content") or (values[0] if values else ""))
    rating_str   = str(values[1] if len(values) > 1 else "")
    return response_str, rating_str


# ---------------------------------------------------------------------------
# Standalone feedback logic (pure function — easy to unit-test)
# ---------------------------------------------------------------------------

def generate_fb_response(
    trial: dict,
    response: dict,
    words_in_use: list,
) -> str | None:
    """Return a JSON feedback string, or None when no feedback is needed.

    Parameters
    ----------
    trial:
        Full trial dict from the task JSON.
    response:
        Pre-parsed model response dict (from TaskRunner).
    words_in_use:
        Accumulating list of all words seen so far in this simulation.
        Modified in-place so the caller's reference stays in sync.
    """
    parser_name = trial.get("parser")  # 'Response_part_1_rm' or 'Response_part_2_rm'
    response_str, rating_str = _extract_fields(response, parser_name)
    stim = trial.get("stimulus", {})

    # ── task-instruction trials ────────────────────────────────────────────
    if "task_instruction" in trial["trcode"]:
        if "ready" not in response_str.lower():
            return "Correct answer is READY. Please carefully follow the instructions."
        return None

    # ── test/recognition trials ────────────────────────────────────────────
    if "test" in trial["trcode"]:
        corr_ans = trial.get("corrAns", "").strip()
        if (
            response_str.lower().replace(".", "").strip()
            == corr_ans.lower().replace(".", "").strip()
        ):
            response_fb = (
                f"**CORRECT: The correct 'Judgment' was '{corr_ans}'. "
                "Your 'Judgment' followed the task instructions.**"
            )
        else:
            response_fb = (
                f"**INCORRECT: The correct 'Judgment' was '{corr_ans}'. "
                "It might also be incorrect due to poor formatting. "
                "Carefully follow all trial instructions about the source-judgment options.**"
            )

        try:
            rating = float(rating_str)
            rate_fb = (
                "**CORRECT: Your 'Confidence' value is in the instructed rating scale range (1–6).**"
                if 1 <= rating <= 6
                else "**INCORRECT: Your 'Confidence' value is not in the instructed range (1–6).**"
            )
        except (ValueError, TypeError):
            rate_fb = "**INCORRECT: Your 'Confidence' value is incorrect due to formatting problems.**"

    # ── encoding trials ────────────────────────────────────────────────────
    else:
        word1 = stim.get("Word_Pair", {}).get("word_1", "").strip()
        word2 = stim.get("Word_Pair", {}).get("word_2", "").strip()
        words_in_use.append(word1)

        if "__" in word2:  # imagined trial
            if response_str.lower() in [w.lower() for w in words_in_use]:
                words_in_use.append(response_str)
                response_fb = (
                    "**INCORRECT: Last trial was an imagined trial type. Your 'Word_2' has "
                    "already been used in a previous trial. Always imagine a novel second word.**"
                )
            elif response_str.lower() not in nltk_words.words():
                words_in_use.append(response_str)
                response_fb = (
                    "**INCORRECT: Last trial was an imagined trial type. Your 'Word_2' is not "
                    "in the English dictionary. Always imagine a novel English word.**"
                )
            else:
                words_in_use.append(response_str)
                response_fb = (
                    "**CORRECT: Last trial was an imagined trial type. "
                    "Your 'Word_2' correctly followed the given instructions.**"
                )
        else:  # perceived trial
            words_in_use.append(word2)
            words_in_use.append(response_str)
            if word2.lower() != response_str.lower():
                response_fb = (
                    "**INCORRECT: Last trial was a perceived trial type. Your 'Word_2' does "
                    "not match the second word in the pair.**"
                )
            else:
                response_fb = (
                    "**CORRECT: Last trial was a perceived trial type. "
                    "Your 'Word_2' correctly followed the trial instruction.**"
                )

        try:
            rating = float(rating_str)
            rate_fb = (
                "**CORRECT: Your 'rating' value is in the instructed rating scale range.**"
                if 0 <= rating <= 100
                else "**INCORRECT: Your 'rating' value is not in the instructed range.**"
            )
        except (ValueError, TypeError):
            rate_fb = "**INCORRECT: Your 'rating' value is incorrect due to formatting problems.**"

    overall_fb = (
        "**INCORRECT Answer!** Follow all trial instructions more accurately."
        if "INCORRECT" in response_fb or "INCORRECT" in rate_fb
        else "**CORRECT Answer!** Well answered, good job!"
    )

    return json.dumps(
        {
            "Feedback on previous response": {
                "response feedback": response_fb,
                "rating feedback": rate_fb,
                "overall feedback": overall_fb,
            }
        },
        indent=4,
    )


# ---------------------------------------------------------------------------
# FeedbackBase subclass
# ---------------------------------------------------------------------------

class Stim_Trial_Injection(FeedbackBase):
    """Feedback handler for the Reality Monitoring word-pair task.

    Instantiated once per participant simulation by psychscanner, so
    ``self.all_words_in_use`` correctly tracks words across all trials.
    """

    def __init__(self):
        self.all_words_in_use: list[str] = []

    def on_response(self, trial: dict, response: dict) -> str | None:
        """Generate feedback based on the trial type and model response."""
        return generate_fb_response(trial, response, self.all_words_in_use)
