"""This module defines chat prompt templates for use in the psychscanner project.

It includes:
- CONV_PROMPT_STIM: A chat prompt template for system messages and input stimuli.
- CONV_PROMPT_PLACE_HISTORY: A chat prompt template for system messages, history, and input stimuli.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CONV_PROMPT_STIM = ChatPromptTemplate.from_messages(
    [("system", "{system_message}"), MessagesPlaceholder(variable_name="input_stim")]
)

CONV_PROMPT_STIM_CANDIDATE = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_message}.                                                   Reflect and grade the assistant response to the user input below."),
        MessagesPlaceholder(variable_name="input_stim"),
        MessagesPlaceholder(variable_name="candidate"),
    ]
)

CONV_PROMPT_PLACE_HISTORY = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "{system_message}",
        ),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="input_stim"),
    ]
)


CONV_PROMPT_MEMORY_MAP = {
    "SingleTurn": CONV_PROMPT_STIM,
    "Convo": CONV_PROMPT_PLACE_HISTORY,
    "Summary": CONV_PROMPT_STIM,
    "ConvoSummaryK": CONV_PROMPT_STIM,
    "MessagePassingHistory": CONV_PROMPT_STIM,
    "RunnableHistory": CONV_PROMPT_PLACE_HISTORY,
}


TOT_CONV_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "{system_message}"
            "Submit exactly {k} guesses for this round.",
        ),
        MessagesPlaceholder(variable_name="input_stim"),
        MessagesPlaceholder(variable_name="candidate")
    ],
).partial(candidate="")



def get_chat_template(agentconfig):

    memory_type = agentconfig.memory_type
    thought_expansion = agentconfig.thought_expansion

def get_trial_state_chat_template(agentconfig,instructions):
    pass

