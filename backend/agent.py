"""Agent runner — wraps the LangGraph memory agent for the API."""

import logging
import pathlib
import sys
from typing import Any

from langchain_core.messages import HumanMessage

from backend.memory_store import MemoryStore

logger = logging.getLogger(__name__)

# Ensure src/ is on path
_src = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


class AgentRunner:
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self._graph = None

    def _get_graph(self):
        if self._graph is None:
            from memory_agent.graph import graph
            self._graph = graph
        return self._graph

    async def chat(
        self,
        message: str,
        user_id: str = "default",
        thread_id: str = "default",
        model: str = "groq/llama-3.3-70b-versatile",
    ) -> str:
        graph = self._get_graph()

        config: dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "model": model,
            }
        }

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

        messages = result.get("messages", [])
        for msg in reversed(messages):
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            if role in ("ai", "assistant"):
                return msg.content

        return "I'm not sure how to respond to that."
