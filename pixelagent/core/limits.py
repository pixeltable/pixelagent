"""Fold the per-turn token cap into provider option dicts.

Each provider spells the cap differently, and a caller who already set it in
`chat_kwargs` / `tool_kwargs` keeps that value.

`openai_model_kwargs` is non-deterministic on purpose. `chat_completions`
writes `tools` into the dict it receives, so a cached dict would leak that
list into a later call.
"""

from typing import Optional

import pixeltable as pxt


@pxt.udf(is_deterministic=False)
def openai_model_kwargs(options: Optional[dict], max_tokens: int, variant: int = 0) -> dict:
    # `variant` is unused. The tools pipeline calls this twice per row, and
    # Pixeltable reuses one result object for identical calls. A distinct
    # variant keeps the follow-up call's dict separate from the one
    # chat_completions mutates.
    merged = dict(options or {})
    if "max_tokens" not in merged and "max_completion_tokens" not in merged:
        merged["max_tokens"] = max_tokens
    return merged


@pxt.udf
def gemini_config(config: Optional[dict], max_tokens: int) -> dict:
    merged = dict(config or {})
    merged.setdefault("max_output_tokens", max_tokens)
    return merged


@pxt.udf
def bedrock_inference_config(options: Optional[dict], max_tokens: int) -> dict:
    merged = dict(options or {})
    merged.setdefault("maxTokens", max_tokens)
    return merged
