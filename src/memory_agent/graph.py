"""LangGraph ReAct memory agent graph — self-contained with built-in store."""

import asyncio
import logging
import os
import uuid
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.store.memory import InMemoryStore

from memory_agent import prompts, tools, utils
from memory_agent.state import State

logger = logging.getLogger(__name__)

# ── Module-level in-memory store (shared across all requests in one process) ──
_store = InMemoryStore()


async def call_model(state: State, config: RunnableConfig) -> dict:
    """Call the LLM with memories injected into the system prompt."""
    cfg         = config.get("configurable", {})
    user_id     = cfg.get("user_id", "default")
    model       = cfg.get("model", os.getenv("MODEL", "groq/llama-3.3-70b-versatile"))
    system_tmpl = cfg.get("system_prompt", prompts.SYSTEM_PROMPT)

    # Fetch relevant memories
    try:
        memories = await _store.asearch(
            ("memories", user_id),
            query=str([m.content for m in state.messages[-3:]]),
            limit=10,
        )
        formatted = "\n".join(f"[{m.key}]: {m.value}" for m in memories)
        user_info = f"\n<memories>\n{formatted}\n</memories>" if formatted else "No memories yet."
    except Exception as e:
        logger.warning(f"Memory fetch failed: {e}")
        user_info = "No memories yet."

    sys_msg = system_tmpl.format(user_info=user_info, time=datetime.now().isoformat())

    llm = utils.load_chat_model(model)
    msg = await llm.bind_tools([tools.upsert_memory]).ainvoke(
        [{"role": "system", "content": sys_msg}, *state.messages]
    )

    # Strip any raw <function=...> text that some models leak into content
    if hasattr(msg, "content") and msg.content:
        import re
        msg.content = re.sub(r"<function=\w+>.*?</function>", "", msg.content, flags=re.DOTALL).strip()

    return {"messages": [msg]}


async def store_memory(state: State, config: RunnableConfig):
    """Save tool call results (memories) into the store."""
    cfg        = config.get("configurable", {})
    user_id    = cfg.get("user_id", "default")
    tool_calls = getattr(state.messages[-1], "tool_calls", [])

    saved = await asyncio.gather(
        *(
            tools.upsert_memory(
                **tc["args"],
                user_id=user_id,
                store=_store,
            )
            for tc in tool_calls
        )
    )

    return {
        "messages": [
            {"role": "tool", "content": mem, "tool_call_id": tc["id"]}
            for tc, mem in zip(tool_calls, saved)
        ]
    }


def route_message(state: State):
    if getattr(state.messages[-1], "tool_calls", None):
        return "store_memory"
    return END


builder = StateGraph(State)
builder.add_node(call_model)
builder.add_node(store_memory)
builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
builder.add_edge("store_memory", "call_model")

graph = builder.compile()
graph.name = "MemoryAgent"

__all__ = ["graph"]
