from __future__ import annotations

from abc import ABC, abstractmethod


class NextTrialBase(ABC):
    """Abstract base class for conditional intermediate-trial handlers.

    Subclass this, override ``next_trial``, and pass the **class** (not an
    instance) as ``next_trial_fn`` in your ``ExpCard``. psychscanner creates
    one instance per participant simulation, so cross-trial state is safely
    kept in ``self`` — same lifecycle as ``FeedbackBase``.

    After every trial, ``TaskRunner`` calls ``next_trial`` with the trial that
    just ran and its parsed response. Returning a trial dict inserts it into
    the run immediately, *before* the next trial from the task card; returning
    ``None`` proceeds straight to that next trial. The returned dict runs
    exactly like a task's ``items`` entry (``trcode``/``stimulus`` required,
    ``fb``/``tools``/``parser`` optional).

    ``next_trial`` can itself keep returning new trials, e.g. to implement
    staircase or adaptive-testing logic. To avoid looping forever if the
    handler keeps proposing the same stimulus, ``TaskRunner`` stops asking
    once the *same* stimulus has been returned ``max_repeat`` times in a row
    and resumes the task card's own trial sequence. Override ``max_repeat``
    to change the threshold.

    Minimal example::

        class RepeatUntilCorrect(NextTrialBase):
            def next_trial(self, trial, response):
                if response.get("rating") is None:
                    return {**trial, "trcode": trial["trcode"] + "_retry"}
                return None
    """

    max_repeat: int = 3

    @abstractmethod
    def next_trial(self, trial: dict, response: dict) -> dict | None:
        """Decide whether to insert an intermediate trial after ``trial``.

        Parameters
        ----------
        trial:
            The trial dict that was just executed (``trcode``, ``stimulus``, …).
        response:
            Model response as a plain Python ``dict`` — same shape passed to
            ``FeedbackBase.on_response``.

        Returns
        -------
        dict or None
            A new trial dict to run next (same schema as a task JSON trial),
            or ``None`` to continue with the next trial from the task card.
        """
