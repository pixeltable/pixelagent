from typing import Optional

import pixeltable as pxt

from ..core.base import BaseAgent
from ..core.spec import ProviderSpec
from .utils import create_messages

try:
    from pixeltable.functions.bedrock import converse, invoke_tools
except ImportError:
    raise ImportError("boto3 not found; run `pip install boto3`")


def _call(model, prompt, system_prompt, model_kwargs, max_tokens, tools):
    # Bedrock is the one provider whose signature still takes a top-level
    # `system=`, and it accepts a column expression there. Extra options go
    # through `inference_config=` rather than being splatted.
    args = {
        "model_id": model,
        "messages": prompt,
        "system": [{"text": system_prompt}],
        "inference_config": model_kwargs,
    }
    if tools is not None:
        args["tool_config"] = tools
    return converse(**args)


SPEC = ProviderSpec(
    name="bedrock",
    default_model="amazon.nova-pro-v1:0",
    build_prompt=lambda system_prompt, memory_context, user_message, image: (
        create_messages(memory_context, user_message, image)
    ),
    call=_call,
    extract=lambda response: response.output.message.content[0].text,
    # Bedrock nests message content one level deeper than the others.
    wrap_user_text=lambda text: [{"role": "user", "content": [{"text": text}]}],
    invoke=invoke_tools,
)


class Agent(BaseAgent):
    """An AWS Bedrock agent with persistent memory and tool execution."""

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
