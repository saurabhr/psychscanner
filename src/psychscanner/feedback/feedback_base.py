from __future__ import annotations

import ast
import json
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage


class FeedbackBase(ABC):
    """Abstract base class for trial-level feedback handlers.

    Subclass this, override ``on_response``, and pass the **class** (not an
    instance) as ``feedback_fn`` in your ``ExpCard``.  psychscanner creates one
    instance per participant simulation, so cross-trial state is safely kept in
    ``self``.

    Minimal example::

        import json
        from psychscanner.feedback import FeedbackBase

        class MyFeedback(FeedbackBase):
            def on_response(self, trial, response):
                if int(response.get("rating", 0)) < 3:
                    return json.dumps({"hint": "Try to be more specific."})
                return None
    """

    @abstractmethod
    def on_response(self, trial: dict, response: dict) -> str | None:
        """Generate feedback for a completed trial.

        Called by ``TaskRunner`` after every trial where the trial's ``fb``
        flag is ``True`` (or absent, which defaults to ``True``).

        Parameters
        ----------
        trial:
            The full trial dict from the task JSON (``trcode``, ``stimulus``,
            ``corrAns``, …).
        response:
            Model response as a plain Python ``dict``.  For structured output
            this is the pydantic model's ``.model_dump()``.  For unstructured
            output this is ``{"content": <raw_string>}``.

        Returns
        -------
        str or None
            Feedback string (typically a JSON string) to inject before the
            *next* trial, or ``None`` to skip feedback for this trial.
        """

    def inject_feedback(self, input_dict: dict, fb_str: str) -> dict:
        """Merge previous-trial feedback into the next trial's ``input_dict``.

        Override this to change the injection format.  The default wraps the
        feedback JSON and the current trial stimulus into a single
        ``HumanMessage`` so the model sees both together.

        Parameters
        ----------
        input_dict:
            The ``{"inputs": [HumanMessage(…)], "system_message": …, …}``
            dict that will be passed to the LLM for the upcoming trial.
        fb_str:
            The string returned by the previous call to ``on_response``.

        Returns
        -------
        dict
            Updated ``input_dict`` with feedback injected into ``inputs``.
        """
        trial_content = input_dict["inputs"][0].content

        if isinstance(trial_content, list):
            # Multimodal content blocks: prepend feedback as a text block
            # instead of json.dumps-flattening the list (which would turn
            # image/audio blocks into inert text).
            input_dict["inputs"] = [
                HumanMessage(content=[{"type": "text", "text": fb_str}, *trial_content])
            ]
            return input_dict

        try:
            trial_dict = json.loads(trial_content)
        except (json.JSONDecodeError, TypeError):
            trial_dict = {"stimulus": trial_content}

        try:
            fb_data = json.loads(fb_str)
        except (json.JSONDecodeError, TypeError):
            fb_data = {"feedback": fb_str}

        merged = {**fb_data, "current_trial": trial_dict}
        input_dict["inputs"] = [HumanMessage(json.dumps(merged, indent=4))]
        return input_dict
