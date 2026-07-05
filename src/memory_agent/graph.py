"""LangGraph ReAct memory agent graph — standard ainvoke compatible."""

import asyncio
import logging
import os
from datetime import datetime
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore

from memory_agent import prompts, tools, utils
from memory_agent.state import State

logger = logging.getLogger(__name__)


async def call_model(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Call the LLM with injected memories from the store."""
    cfg         = config.get("configurable", {})
    user_id     = cfg.get("user_id", "default")
    model       = cfg.get("model", os.getenv("MODEL", "groq/llama-3.3-70b-versatile"))
    system_tmpl = cfg.get("system_prompt", prompts.SYSTEM_PROMPT)

    # Fetch top-10 relevant memories
    memories = await store.asearch(
        ("memories", user_id),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

    formatted = "\n".join(
        f"[{mem.key}]: {mem.value}" for mem in memories
    )
    user_info = f"\n<memories>\n{formatted}\n</memories>" if formatted else "No memories yet."

    sys_msg = system_tmpl.format(user_info=user_info, time=datetime.now().isoformat())
    llm     = utils.load_chat_model(model)

    msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
        [{"role": "system", "content": sys_msg}, *state.messages]
    )
    return {"messages": [msg]}


async def store_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    """Execute all upsert_memory tool calls from the last message."""
    cfg      = config.get("configurable", {})
    user_id  = cfg.get("user_id", "default")
    tool_calls = getattr(state.messages[-1], "tool_calls", [])

    saved = await asyncio.gather(
        *(
            tools.upsert_memory(
                **tc["args"],
                user_id=user_id,
                store=store,
            )
            for tc in tool_calls
        )
    )

    results = [
        {"role": "tool", "content": mem, "tool_call_id": tc["id"]}
        for tc, mem in zip(tool_calls, saved)
    ]
    return {"messages": results}


def route_message(state: State):
    """Route to store_memory if there are tool calls, otherwise END."""
    if getattr(state.messages[-1], "tool_calls", None):
        return "store_memory"
    return END


# Build the graph
builder = StateGraph(State)
builder.add_node(call_model)
builder.add_node(store_memory)
builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
builder.add_edge("store_memory", "call_model")

graph = builder.compile()
graph.name = "MemoryAgent"

__all__ = ["graph"]
