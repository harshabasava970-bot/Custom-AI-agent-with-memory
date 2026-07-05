"""Agent runner — wraps the LangGraph memory agent for the API."""

import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage

from backend.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class AgentRunner:
    """Bridges the FastAPI layer and the LangGraph memory agent."""

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self._graph = None

    def _get_graph(self):
        """Lazy-load the graph so import errors surface clearly at runtime."""
        if self._graph is None:
            # Add src to path so the package resolves correctly
            import sys, pathlib
            src = pathlib.Path(__file__).resolve().parents[1] / "src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))

            from memory_agent.graph import graph
            self._graph = graph
        return self._graph

    async def chat(
        self,
        message: str,
        user_id: str = "default",
        thread_id: str = "default",
        model: str = "openai/gpt-4o-mini",
    ) -> str:
        """Run one turn of the memory agent and return the assistant reply."""
        graph = self._get_graph()

        # LangGraph config: thread_id keeps conversation history,
        # user_id scopes memories, store is provided by the graph compile-time store.
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "model": model,
            }
        }

        store = self.memory_store.get_langgraph_store()

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            store=store,
        )

        # The last AIMessage is the assistant reply
        messages = result.get("messages", [])
        for msg in reversed(messages):
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            if role in ("ai", "assistant"):
                return msg.content

        return "I'm not sure how to respond to that."
