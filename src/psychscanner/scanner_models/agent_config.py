from pydantic import BaseModel, ConfigDict, Field
from typing import TYPE_CHECKING, Any, Callable, Literal, Type
from psychscanner.staging import factory_settings
from psychscanner.datasets.prompts import chat_prompts

class AgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    modelname: str | None
    familyname: str | None
    parameters: dict | None
    modelobject: Any | None
    memory_type: Literal["SingleTurn", "Convo"]
    memory_k: int | None
    summary_k: int | None = 0

    chain_type: Literal["item", "trial", "task"]
    chain_config: Any | None = None
    trace_cfg: Any | None = None

    system_msg: str | None
    agent_model: Any | None = None

    # Resolved parser: a BaseModel subclass, a dispatch callable, or None.
    parser: Type[BaseModel] | Callable | None
    parser_raw: bool | None
    parser_config: dict | None
    trial_parsers: list[Any] | None = None

    tools: list[Any] | None = None

    feedback: Any | None = None
    feedback_fn: Any | None = None
    agent_prompt: None = None
