from langgraph.graph import StateGraph
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langgraph.graph import START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from .base.base_agent import AgentInitializer
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

# Import your parser classes here, or adjust as needed
from psychscanner.datasets.prompts.parser import Response_part_1_rm, Response_part_2_rm

# Example dynamic parser function
def get_dynamic_outparser(state):
    trcode = state.trcode
    # Replace with your actual logic and parser classes
    if "test" in trcode:
        return Response_part_2_rm
    else:
        return Response_part_1_rm


def single_turn_convo_node(
    agent_cfg, workflow=None, nodename="sigconvo",compile_graph = True,
    add_start =True,

):
    runnable1 = None
    runnable2 = None
    runnable = None

    prompt = agent_cfg.agent_prompt
    if prompt is None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_message}"),
                MessagesPlaceholder(variable_name="inputs"),
            ]
        )
    if agent_cfg.parser == "0":
        runnable = prompt | agent_cfg.modelobject


    elif agent_cfg.parser == "dynamic":
        # Dynamic parser selection based on state['trcode']
        runnable1 = prompt | agent_cfg.modelobject.with_structured_output(
            Response_part_1_rm,
            include_raw=agent_cfg.parser_raw,
            **agent_cfg.parser_config,
        )
        runnable2 = prompt | agent_cfg.modelobject.with_structured_output(
            Response_part_2_rm,
            include_raw=agent_cfg.parser_raw,
            **agent_cfg.parser_config,
        )
        

    else:
        # Dynamic parser selection based on state['trcode']
        runnable = prompt | agent_cfg.modelobject.with_structured_output(
            agent_cfg.parser,
            include_raw=agent_cfg.parser_raw,
            **agent_cfg.parser_config,
        )

    class State(TypedDict):
        inputs: Annotated[Sequence[BaseMessage], add_messages]
        system_message: str
        trcode: str
    

    def call_model(state: State):


        response = runnable.invoke(state)

        if agent_cfg.parser_raw:
                response = response["raw"]
        else:
            response = str(response.model_dump())
            response = AIMessage(response)

        return {"inputs": [response]}
    
    def runnable_resp1node(state: State):


        response = runnable1.invoke(state)

        if agent_cfg.parser_raw:
                response = response["raw"]
        else:
            response = str(response.model_dump())
            response = AIMessage(response)

        return {"inputs": [response]}
    def runnable_resp2node(state: State):


        response = runnable2.invoke(state)

        if agent_cfg.parser_raw:
                response = response["raw"]
        else:
            response = str(response.model_dump())
            response = AIMessage(response)

        return {"inputs": [response]}

    def parser_selector(state: State):
        trcode = state['trcode']
        if "test" in trcode:
            return "runnable_resp2node"
        else:
            return "runnable_resp1node"

    if workflow is None:
        workflow = StateGraph(state_schema=State)

    if agent_cfg.parser == "dynamic":
        workflow.add_node("runnable_resp1node", runnable_resp1node)
        workflow.add_node("runnable_resp2node", runnable_resp2node)
        workflow.add_edge("runnable_resp1node", END)
        workflow.add_edge("runnable_resp2node", END)
        workflow.add_conditional_edges(START, parser_selector)
    else: 
        workflow.add_node(nodename, call_model)
        if add_start:
            workflow.add_edge(START,nodename)

    graph = workflow
    
    if compile_graph:
        if agent_cfg.memory_type == "SingleTurn":
            graph = graph.compile()
        elif agent_cfg.memory_type == "Convo":
            graph = graph.compile(checkpointer=MemorySaver())
        return graph
    return workflow
