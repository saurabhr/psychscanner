"""Get AI model for the psychscanner.

This module provides functionality to initialize and return chat models.
It supports multiple model families and automatically resolves API keys
from environment variables for cloud providers.
"""

import os

import click
from langchain.chat_models import init_chat_model

from .mock_llm import ChatMockModel

NOT_FOUND_LLM_MSG = "Requested llm not available. Check your model and family."

# Canonical env-var name for each family that needs an API key.
_API_KEY_ENV: dict[str, str] = {
    "openai":       "OPENAI_API_KEY",
    "anthropic":    "ANTHROPIC_API_KEY",
    "groq":         "GROQ_API_KEY",
    "together":     "TOGETHER_API_KEY",
    "mistral":      "MISTRAL_API_KEY",
    "cohere":       "COHERE_API_KEY",
    "google":       "GOOGLE_API_KEY",
    "google-genai": "GOOGLE_API_KEY",
    "gemini":       "GOOGLE_API_KEY",
    "fireworks":    "FIREWORKS_API_KEY",
    "azure":        "AZURE_OPENAI_API_KEY",
    "huggingface":  "HUGGINGFACEHUB_API_TOKEN",
    "ollama":       "OLLAMA_API_KEY",
}

# Kwarg name the provider SDK expects for the key (defaults to "api_key").
_API_KEY_KWARG: dict[str, str] = {
    "huggingface": "huggingfacehub_api_token",
}


def _resolve_api_key(family: str, parameters: dict) -> dict:
    """Return a copy of *parameters* with the provider's API key injected.

    Checks *parameters* first (caller wins), then the canonical env var.
    Uses the provider-specific kwarg name from ``_API_KEY_KWARG`` (defaults
    to ``"api_key"``).
    """
    kwarg = _API_KEY_KWARG.get(family.lower(), "api_key")

    if kwarg in parameters or "api_key" in parameters:
        return parameters

    env_var = _API_KEY_ENV.get(family.lower())
    if env_var is None:
        return parameters

    key = os.getenv(env_var)
    if key:
        click.echo(f"--<api key>-- loaded {env_var} from environment")
        return {**parameters, kwarg: key}

    click.echo(
        f"--<api key>-- warning: {env_var} not set; "
        f"proceeding without explicit api_key for family '{family}'"
    )
    return parameters


def llm_chat_model(
    model: str,
    family: str,
    parameters: dict | None = None,
) -> object:
    """Initialize and return a chat model for the given model and family.

    ``mock-llm`` is the only special case (no real provider).  All other
    families — including ``ollama`` and ``huggingface`` — are routed through
    ``init_chat_model`` after the API key is resolved from *parameters* or
    the canonical environment variable for that family.

    Pass ``base_url`` in *parameters* to point ``ollama`` at a remote server.
    For ``huggingface`` the key is injected as ``huggingfacehub_api_token``
    (the name expected by the HuggingFace SDK).

    Parameters
    ----------
    model:
        Model name or identifier (e.g. ``"gpt-4o"``,
        ``"smollm2:360m-instruct-fp16"``).
    family:
        Provider / family string (e.g. ``"openai"``, ``"ollama"``,
        ``"groq"``, ``"huggingface"``).
    parameters:
        Optional dict of extra kwargs forwarded to the model constructor
        (e.g. ``temperature``, ``api_key``, ``base_url``).

    Returns
    -------
    object
        An initialized LangChain chat model instance.

    Raises
    ------
    ValueError
        If the model or family is unavailable.
    """
    if parameters is None:
        parameters = {}

    family_lower = family.lower()

    if family_lower == "mock-llm":
        chat_model = ChatMockModel(
            model=model,
            repeat_buffer_length=10,
            **parameters,
        )
    else:
        params = _resolve_api_key(family_lower, parameters)
        try:
            chat_model = init_chat_model(
                model, model_provider=family, **params
            )
        except Exception as exc:
            raise ValueError(NOT_FOUND_LLM_MSG) from exc

    click.echo(f"--<chat model>-- {chat_model}")
    return chat_model
