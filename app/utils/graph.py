"""This file contains the graph utilities for the application."""

from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from typing import List
import json
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


def dump_messages(messages: list[BaseMessage]) -> list[dict]:
    """Dump the messages to a list of dictionaries.

    Args:
        messages (list[BaseMessage]): The messages to dump.

    Returns:
        list[dict]: The dumped messages.
    """
    return [message.model_dump() for message in messages]


def prepare_messages(
    messages: list[BaseMessage], system_prompt: str
) -> list[BaseMessage]:
    """Prepare the messages for the LLM.

    Args:
        messages (list[BaseMessage]): The messages to prepare.
        system_prompt (str): The system prompt to use.

    Returns:
        list[BaseMessage]: The prepared messages.
    """

    return [SystemMessage(content=system_prompt)] + messages[
        -settings.MAX_CONTEXT_MESSAGES :
    ]


def format_messages(messages: List[BaseMessage]) -> str:
    """Generate a readable execution log from a list of messages including user input, AI responses, and tool calls."""
    lines = []

    for msg in messages:

        if isinstance(msg, HumanMessage):
            lines.append(f"USER: {msg.content}")
            continue

        if isinstance(msg, AIMessage):
            if msg.content:
                lines.append(f"SYSTEM: {msg.content}")

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "unknown_tool")
                    tool_args = tool_call.get("args", {})
                    pretty_args = json.dumps(tool_args, indent=2, ensure_ascii=False)
                    lines.append(f"SYSTEM TOOL_CALL: {tool_name}(\n{pretty_args}\n)")

        # Handle tool result
        elif getattr(msg, "type", None) == "tool":
            lines.append(f"SYSTEM TOOL_RESULT: {msg.content}")

    return "\n\n".join(lines)
