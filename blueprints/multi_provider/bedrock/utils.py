import base64
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
    """
    Format messages for Bedrock models (similar to Anthropic's format).
    
    Args:
        memory_context: Previous conversation history
        current_message: Current user message
        image: Optional image to include with the message
        
    Returns:
        List of message dictionaries formatted for Bedrock
    """
    # Create a copy to avoid modifying the original
    messages = memory_context.copy()

    # For text-only messages
    if not image:
        messages.append({"role": "user", "content": current_message})
        return messages

    # Convert image to base64
    bytes_arr = io.BytesIO()
    image.save(bytes_arr, format="JPEG")
    b64_bytes = base64.b64encode(bytes_arr.getvalue())
    b64_encoded_image = b64_bytes.decode("utf-8")

    # Create content blocks with text and image
    content_blocks = [
        {"type": "text", "text": current_message},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64_encoded_image,
            },
        },
    ]

    messages.append({"role": "user", "content": content_blocks})

    return messages
