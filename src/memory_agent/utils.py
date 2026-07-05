"""Utility functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a 'provider/model' string.

    Args:
        fully_specified_name: e.g. 'openai/gpt-4o-mini'
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)
