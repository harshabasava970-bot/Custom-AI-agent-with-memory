"""Runtime context for the memory agent."""

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated

from memory_agent import prompts


@dataclass(kw_only=True)
class Context:
    """Main context for the memory graph."""

    user_id: str = "default"
    """Unique user ID — memories are scoped per user."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="groq/llama-3.3-70b-versatile",
        metadata={
            "description": "LLM to use in 'provider/model' format. "
            "Examples: groq/llama-3.3-70b-versatile, groq/mixtral-8x7b-32768, "
            "openai/gpt-4o-mini, anthropic/claude-3-5-sonnet-20241022"
        },
    )

    system_prompt: str = prompts.SYSTEM_PROMPT

    def __post_init__(self):
        """Override defaults with environment variables if set."""
        for f in fields(self):
            if not f.init:
                continue
            if getattr(self, f.name) == f.default:
                env_val = os.environ.get(f.name.upper())
                if env_val:
                    setattr(self, f.name, env_val)
