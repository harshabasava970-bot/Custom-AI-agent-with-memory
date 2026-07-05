"""LangGraph ReAct memory agent graph."""

import asyncio
import logging
from datetime import datetime
from typing import cast

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from memory_agent import tools, utils
from memory_agent.context import Context
from memory_agent.state import State

logger = logging.getLogger(__name__)


async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    """Call the LLM, injecting relevant memories from the store."""
    user_id = runtime.context.user_id
    model = runtime.context.model
    system_prompt = runtime.context.system_prompt

    # Retrieve the top-10 most semantically relevant memories
    memories = await cast(BaseStore, runtime.store).asearch(
        ("memories", user_id),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

    formatted = "\n".join(
        f"[{mem.key}]: {mem.value} (similarity: {mem.score})" for mem in memories
    )
    if formatted:
        formatted = f"\n<memories>\n{formatted}\n</memories>"

    sys = system_prompt.format(user_info=formatted or "No memories yet.", time=datetime.now().isoformat())
    llm = utils.load_chat_model(model)

    msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
        [{"role": "system", "content": sys}, *state.messages]
    )
    return {"messages": [msg]}


async def store_memory(state: State, runtime: Runtime[Context]):
    """Execute all upsert_memory tool calls concurrently."""
    tool_calls = getattr(state.messages[-1], "tool_calls", [])

    saved = await asyncio.gather(
        *(
            tools.upsert_memory(
                **tc["args"],
                user_id=runtime.context.user_id,
                store=cast(BaseStore, runtime.store),
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
    msg = state.messages[-1]
    if getattr(msg, "tool_calls", None):
        return "store_memory"
    return END


# Build the graph
builder = StateGraph(State, context_schema=Context)
builder.add_node(call_model)
builder.add_node(store_memory)
builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
builder.add_edge("store_memory", "call_model")

graph = builder.compile()
graph.name = "MemoryAgent"

__all__ = ["graph"]
