from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import click
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages
)
import click
from langchain_core.messages.utils import count_tokens_approximately
#from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel
from langgraph.graph import START, MessagesState, StateGraph
from psychscanner.datasets.prompts import chat_prompts


def chat_template_parser_chain(
    chat_model: Any,

    parser: BaseModel | None,
    parser_config: dict[str, Any],
    *,
    parser_raw: bool,
    convo_template: Any,
):
    pass

class AgentInitializer:
    def __init__(self,agent_cfg):
        self.cfg = agent_cfg
        self.ai_memory = MemorySaver()
        self.store = InMemoryStore()
        self.cpsave = InMemorySaver()
        self.memoryname = self.cfg.memory_type
        self.memory_minimize = False
        self.memory_k = self.cfg.memory_k
        self.ai_memory = MemorySaver()
        self.memory_minimize = False
        self.chain_type = self.cfg.chain_type
        self.parser = self.cfg.parser
        self.trial_parsers = self.cfg.trial_parsers
        self.tools = self.cfg.tools
        self.workflow = None
        self.ai_app = None




