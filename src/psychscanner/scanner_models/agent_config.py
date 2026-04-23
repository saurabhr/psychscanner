from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Any, Callable,Literal
from psychscanner.staging import factory_settings
from psychscanner.datasets.prompts import chat_prompts

class AgentConfig(BaseModel):
    modelname: str | None
    familyname: str | None
    parameters: dict | None
    modelobject: Any|None
    memory_type: Literal["SingleTurn", "Convo"]
    memory_k: int | None

    chain_type: Literal["item","trial","task"]
    chain_config: Any|None = None
    trace_cfg: Any | None = None

    system_msg: str | None
    agent_model: Any | None = None

    parser: Any | None
    parser_raw: Any | None
    parser_config: Any | None
    trial_parsers: list[Any]|None = None

    feedback: Any|None =None
    feedback_fn: Any|None=None
    agent_prompt: None=None
