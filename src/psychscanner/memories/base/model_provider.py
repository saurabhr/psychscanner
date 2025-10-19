"""Get AI model for the psychscanner.

This module provides functionality to initialize and return chat models.
It supports multiple model families such as 'ollama' and 'huggingface',
and includes error handling for unavailable models.
"""

from langchain.chat_models import init_chat_model
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
import click
from .mock_llm import ChatMockModel

NOT_FOUND_LLM_MSG = "Requested llm not available. Check your model and family."


def llm_chat_model(model: str, family: str, parameters: dict) -> object:
    """Initialize and return a chat model based on the specified model, family, and parameters.

    Parameters:
    ----------
    model : str
        The name or identifier of the model to use.
    family : str
        The family or provider of the model (e.g., 'ollama', 'huggingface').
    parameters : dict
        Additional parameters required for initializing the model.

    Returns:
    -------
    object
        An initialized chat model instance.

    Raises:
    ------
    ValueError
        If the requested model or family is not available.
    """
    if family == "mock-llm":
        if parameters:
            chat_model = ChatMockModel(
                model=model, repeat_buffer_length=3, **parameters
            )
        else:
            chat_model = ChatMockModel(model=model, repeat_buffer_length=3)
    elif family == "ollama":
        try:
            if parameters:
                llm = ChatOllama(model=model, **parameters)
            else:
                llm = ChatOllama(model=model)
            chat_model = llm
        except Exception as exc:
            raise ValueError(NOT_FOUND_LLM_MSG) from exc

    elif family == "huggingface":
        try:
            if parameters:
                llm = HuggingFaceEndpoint(
                    repo_id=model, task="text-generation", **parameters
                )
            else:
                llm = HuggingFaceEndpoint(
                    repo_id=model,
                    task="text-generation",
                )
            chat_model = ChatHuggingFace(llm=llm)

        except Exception as exc:
            raise ValueError(NOT_FOUND_LLM_MSG) from exc
    elif parameters:
        chat_model = init_chat_model(model, model_provider=family, **parameters)
    else:
        chat_model = init_chat_model(model, model_provider=family)

    click.echo(f"--<chat model>-- {chat_model}")
    return chat_model
