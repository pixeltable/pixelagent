from typing import Optional

import pixeltable as pxt

from ..core.base import BaseAgent
from ..core.limits import openai_model_kwargs
from ..core.spec import ProviderSpec
from .utils import create_messages

try:
    from pixeltable.functions.openai import chat_completions, invoke_tools
except ImportError:
    raise ImportError("openai not found; run `pip install openai`")


def _call(model, prompt, system_prompt, model_kwargs, max_tokens, tools):
    # `model` is written first deliberately: pixeltable stores a computed
    # column's call as JSONB, and matching the stored kwarg order keeps a
    # re-applied schema a no-op.
    args = {
        "model": model,
        "messages": prompt,
        # variant=1 on the follow-up call (tools is None) so it does not share
        # the dict chat_completions mutates while advertising tools.
        "model_kwargs": openai_model_kwargs(model_kwargs, max_tokens, 0 if tools is not None else 1),
    }
    if tools is not None:
        args["tools"] = tools
    return chat_completions(**args)


SPEC = ProviderSpec(
    name="openai",
    default_model="gpt-4o-mini",
    # OpenAI carries the system prompt as the first message, so build_prompt
    # takes it; the other providers pass it out of band.
    build_prompt=create_messages,
    call=_call,
    extract=lambda response: response.choices[0].message.content,
    wrap_user_text=lambda text: [{"role": "user", "content": text}],
    invoke=invoke_tools,
)


class Agent(BaseAgent):
    """An OpenAI agent with persistent memory and tool execution."""

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
