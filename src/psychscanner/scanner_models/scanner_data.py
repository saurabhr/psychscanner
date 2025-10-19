from typing import TYPE_CHECKING, Any, Callable
from psychscanner.datasets import task_prompts


def scanner_data(expcard: Any = None) -> dict:
    """Process the experimental card data and generate prompt data.

    Parameters:
    ----------
    expcard : Any, optional
        The experimental card object containing input data. If None, the
        instance's experimental card is used.

    Returns:
    -------
    dict
        A dictionary containing system prompt data, trial prompt data, and
        the experimental card.
    """
    if expcard is None:
        raise ValueError("Exp Card is required.")

    system_prompt_data = task_prompts.gen_symsg_promptdata(expcard)
    trial_prompt_data = task_prompts.gen_trial_promptdata(expcard)

    return {
        "system_prompts": system_prompt_data["system_prompts"],
        "task_prompts": trial_prompt_data,
        "expcard": expcard,
        "system_template": system_prompt_data["system_template"],
        "chain_type": system_prompt_data["chain_type"],
    }
