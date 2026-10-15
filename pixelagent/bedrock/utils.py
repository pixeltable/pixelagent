import io
from typing import Optional

import PIL
import pixeltable as pxt


@pxt.udf
def create_messages(
    memory_context: list[dict],
    current_message: str,
    image: Optional[PIL.Image.Image] = None,
) -> list[dict]:
    """Format history as Bedrock Converse messages.

    Text content is `{"text": ...}`. An image is
    `{"image": {"format": "jpeg", "source": {"bytes": ...}}}`.
    """
    # Create a copy to avoid modifying the original and format for Bedrock
    messages = []
    
    # Get messages in oldest-first order
    reversed_memory = list(reversed(memory_context))
    
    # Ensure the conversation starts with a user message
    # If the first message is from the assistant, skip it
    start_idx = 0
    if reversed_memory and reversed_memory[0]["role"] == "assistant":
        start_idx = 1
    
    # Format previous messages for Bedrock
    for msg in reversed_memory[start_idx:]:
        # Convert string content to the required list format
        if isinstance(msg["content"], str):
            messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}]
            })
        else:
            # If it's already in the correct format, keep it as is
            messages.append(msg)

    # For text-only messages
    if not image:
        messages.append({"role": "user", "content": [{"text": current_message}]})
        return messages

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    # Converse converts a PIL image or a media path in source.bytes. Raw JPEG
    # bytes are already the wire format, so they pass through that walk.
    content_blocks = [
        {"text": current_message},
        {
            "image": {
                "format": "jpeg",
                "source": {"bytes": buf.getvalue()},
            },
        },
    ]

    messages.append({"role": "user", "content": content_blocks})

    return messages
