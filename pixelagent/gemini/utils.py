import base64
import io
from typing import Optional

import PIL
import pixeltable as pxt


@pxt.udf
def create_content(
    memory_context: list[dict],
    current_message: str,
    image: Optional[PIL.Image.Image] = None,
) -> list:
    """Flatten the conversation to Gemini content parts.

    Text-only turns are a one-element list. An image turn prepends an
    `inline_data` part. `generate_content` accepts that list as `contents`.
    """
    context = ""
    for msg in memory_context:
        context += f"{msg['role'].title()}: {msg['content']}\n"
    context += f"User: {current_message}\n"
    context += "Assistant: "
    if image is None:
        return [context]

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return [
        {"inline_data": {"mime_type": "image/jpeg", "data": encoded}},
        context,
    ]


@pxt.udf
def merge_system_instruction(
    system_prompt: str, model_kwargs: Optional[dict] = None
) -> dict:
    """Gemini takes the system prompt as `system_instruction` inside `config`."""
    return {"system_instruction": system_prompt, **(model_kwargs or {})}
