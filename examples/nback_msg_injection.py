"""N-back feedback handler.

Subclasses FeedbackBase to provide trial-by-trial corrective feedback for
the SweetPea-generated n-back task (examples/tasks/nback_sweetpea_n*.tcard.psyscan
et al.) -- the successor to the Reality Monitoring word-pair task's
rm_msg_injection.py in the tutorial notebooks.

N-back has no encoding/test phase split the way the RM task did: every
trial is a single "is this letter a match?" judgment, with ground truth
already computed and stored on the trial dict's ``corrAns`` field by
examples/tasks/generators/generate_nback_sweetpea.py. So feedback here is
a single string comparison, not per-trial-type branching -- there's no
cross-trial state to track either (no used-word list, no rating scale),
so unlike Stim_Trial_Injection this class carries no instance state at all.

Usage::

    from nback_msg_injection import NbackFeedback

    card_in = ExpCardInit(
        ...,
        feedback=True,
        feedback_fn=NbackFeedback,   # pass the class, not an instance
    )
"""
from __future__ import annotations

import json

from psychscanner.feedback import FeedbackBase


def generate_fb_response(trial: dict, response_text: str) -> str:
    """Return a JSON feedback string for one n-back trial.

    Parameters
    ----------
    trial:
        Full trial dict from the task JSON (carries ``corrAns``: ``"match"``
        or ``"no-match"``).
    response_text:
        The model's raw response text for this trial (the n-back task card
        has no structured parser -- see generate_nback_sweetpea.py -- so
        this is free text, matched case-insensitively against "match" /
        "no-match" rather than requiring an exact clean string).
    """
    corr_ans = trial["corrAns"].strip().lower()
    resp = response_text.strip().lower()

    if "no-match" in resp or "no match" in resp:
        judged = "no-match"
    elif "match" in resp:
        judged = "match"
    else:
        judged = None

    if judged is None:
        response_fb = (
            f"**INCORRECT: Could not find 'match' or 'no-match' in your response "
            f"('{response_text.strip()[:60]}'). The correct answer was "
            f"'{corr_ans}'. Reply with only 'match' or 'no-match'.**"
        )
    elif judged == corr_ans:
        response_fb = f"**CORRECT: '{corr_ans}' was the right judgment.**"
    else:
        response_fb = (
            f"**INCORRECT: You answered '{judged}', but the correct judgment "
            f"was '{corr_ans}'.**"
        )

    return json.dumps(
        {"Feedback on previous response": {"response feedback": response_fb}},
        indent=4,
    )


class NbackFeedback(FeedbackBase):
    """Feedback handler for the SweetPea n-back task.

    Stateless (no cross-trial tracking needed, unlike the RM handler's
    used-word list), so instantiation is a formality kept only to match
    the FeedbackBase contract TaskRunner expects.
    """

    def on_response(self, trial: dict, response: dict) -> str:
        """Generate feedback for one n-back trial.

        ``response`` is whatever TaskRunner captured for this trial. The
        n-back card has no structured parser (``"parser": null``), so this
        is normally ``{"content": "<raw text>"}`` or similar rather than a
        parsed Pydantic-schema dict -- extract the raw text defensively.
        """
        if isinstance(response, dict):
            text = str(response.get("content") or next(iter(response.values()), ""))
        else:
            text = str(response)
        return generate_fb_response(trial, text)


def demo() -> None:
    """Self-check: both judgment outcomes and the unparseable case."""
    trial = {"trcode": "nback_sweetpea_n2_05", "corrAns": "match"}

    correct = json.loads(generate_fb_response(trial, "match"))
    assert "CORRECT" in correct["Feedback on previous response"]["response feedback"]
    assert "INCORRECT" not in correct["Feedback on previous response"]["response feedback"]

    wrong = json.loads(generate_fb_response(trial, "no-match"))
    assert "INCORRECT" in wrong["Feedback on previous response"]["response feedback"]

    trial_nm = {"trcode": "nback_sweetpea_n2_06", "corrAns": "no-match"}
    correct_nm = json.loads(generate_fb_response(trial_nm, "It's a no-match."))
    assert "CORRECT" in correct_nm["Feedback on previous response"]["response feedback"]

    unparseable = json.loads(generate_fb_response(trial, "I think it's the letter B"))
    assert "INCORRECT" in unparseable["Feedback on previous response"]["response feedback"]

    handler = NbackFeedback()
    fb = handler.on_response(trial, {"content": "match"})
    assert "CORRECT" in json.loads(fb)["Feedback on previous response"]["response feedback"]

    print("demo() OK — match/no-match/unparseable feedback and NbackFeedback.on_response verified")


if __name__ == "__main__":
    demo()
