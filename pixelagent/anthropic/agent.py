from typing import Optional

import pixeltable as pxt

from ..core.base import BaseAgent
from ..core.spec import ProviderSpec
from .utils import create_messages, merge_system

try:
    from pixeltable.functions.anthropic import invoke_tools, messages
except ImportError:
    raise ImportError("anthropic not found; run `pip install anthropic`")


def _call(model, prompt, system_prompt, model_kwargs, max_tokens, tools):
    # `model` first: matching pixeltable's stored kwarg order keeps a re-applied
    # schema a no-op. max_tokens is required by the Anthropic API.
    args = {
        "model": model,
        "messages": prompt,
        "max_tokens": max_tokens,
        "model_kwargs": merge_system(system_prompt, model_kwargs),
    }
    if tools is not None:
        args["tools"] = tools
    return messages(**args)


SPEC = ProviderSpec(
    name="anthropic",
    default_model="claude-3-5-sonnet-latest",
    # Anthropic keeps the system prompt out of the message list, so the
    # system_prompt column is dropped here and travels via merge_system.
    build_prompt=lambda system_prompt, memory_context, user_message, image: (
        create_messages(memory_context, user_message, image)
    ),
    call=_call,
    extract=lambda response: response.content[0].text,
    wrap_user_text=lambda text: [{"role": "user", "content": text}],
    invoke=invoke_tools,
)


class Agent(BaseAgent):
    """An Anthropic agent with persistent memory and tool execution."""

    spec = SPEC

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = SPEC.default_model,
        n_latest_messages: Optional[int] = 10,
        tools: pxt.Tools | None = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,
        tool_kwargs: Optional[dict] = None,
        max_tokens: int = 4096,
    ):
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            model=model,
            n_latest_messages=n_latest_messages,
            tools=tools,
            reset=reset,
            chat_kwargs=chat_kwargs,
            tool_kwargs=tool_kwargs,
            max_tokens=max_tokens,
        )
