from dataclasses import dataclass
from typing import Callable, Optional

import pixeltable as pxt


@dataclass(frozen=True)
class ProviderSpec:
    """
    Everything that differs between providers, in one value.

    Each field is a callable that receives Pixeltable column expressions and
    returns an expression. They are evaluated once, inside the schema factory's
    class body, so what they build becomes a computed column definition rather
    than a per-row Python call.

    `model` and `tools` arrive as plain Python values rather than columns, and
    that is a Pixeltable requirement, not a design choice:

    - `model` must be a compile-time literal. openai.chat_completions,
      anthropic.messages and gemini.generate_content each declare a
      `@fn.resource_pool` keyed on the model, and a non-Literal model makes
      pixeltable raise `Could not determine resource pool`. (bedrock.converse
      declares no resource pool and would accept a column, but the four
      providers are kept uniform.)
    - `tools` is compiled into the column by `invoke_tools`, so the executable
      toolset is fixed per column and therefore per table.

    Everything else -- system prompt, model_kwargs, max_tokens, history depth --
    is row data. See pixelagent/core/schema.py.
    """

    name: str
    default_model: str

    # (system_prompt, memory_context, user_message, image) -> messages expression
    build_prompt: Callable[..., pxt.exprs.Expr]

    # (model, prompt, system_prompt, model_kwargs, max_tokens, tools) -> response expression
    call: Callable[..., pxt.exprs.Expr]

    # response -> the assistant's text
    extract: Callable[[pxt.exprs.Expr], pxt.exprs.Expr]

    # a single user-turn text expression -> a messages expression.
    # Providers disagree on the shape: OpenAI and Anthropic take
    # {"role": "user", "content": <str>}, Bedrock nests it as
    # {"role": "user", "content": [{"text": <str>}]}, and Gemini takes the bare
    # text. Used by the tool-calling handshake, which has no history to fold in.
    wrap_user_text: Callable[[pxt.exprs.Expr], pxt.exprs.Expr]

    # (tools, response) -> tool invocation results; None for providers without tool support
    invoke: Optional[Callable[..., pxt.exprs.Expr]] = None
