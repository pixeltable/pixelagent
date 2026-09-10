from typing import Optional

import pixeltable as pxt

from ..core.base import BaseAgent
from ..core.spec import ProviderSpec
from .utils import create_content, merge_system_instruction

try:
    from pixeltable.functions.gemini import generate_content, invoke_tools
except ImportError:
    raise ImportError("google.genai not found; run `pip install google-genai`")


def _call(model, prompt, system_prompt, model_kwargs, max_tokens, tools):
    # Gemini takes both the system prompt and any extra options inside `config`.
    args = {
        "model": model,
        "contents": prompt,
        "config": merge_system_instruction(system_prompt, model_kwargs),
    }
    if tools is not None:
        args["tools"] = tools
    return generate_content(**args)


SPEC = ProviderSpec(
    name="gemini",
    default_model="gemini-2.0-flash",
    # create_content flattens the conversation to a single string, and Gemini's
    # pixeltable binding has no image parameter, so images are not supported
    # here. Passing one to chat() is accepted and ignored.
    build_prompt=lambda system_prompt, memory_context, user_message, image: (
        create_content(memory_context, user_message)
    ),
    call=_call,
    extract=lambda response: response["candidates"][0]["content"]["parts"][0]["text"],
    wrap_user_text=lambda text: text,
    invoke=invoke_tools,
)


class Agent(BaseAgent):
    """A Google Gemini agent with persistent memory and tool execution."""

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
