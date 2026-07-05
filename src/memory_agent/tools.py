"""Agent tools for memory storage."""

import uuid
from typing import Annotated

from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore


async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: uuid.UUID | None = None,
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a memory in the database.

    If a memory conflicts with an existing one, UPDATE it by passing memory_id.
    Do not create duplicate memories.

    Args:
        content: The main content of the memory.
            Example: "User's name is Harsha and they love Python."
        context: Additional context for when/why this was noted.
            Example: "Mentioned during onboarding conversation."
        memory_id: Only provide when UPDATING an existing memory.
    """
    mem_id = memory_id or uuid.uuid4()
    await store.aput(
        ("memories", user_id),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"
